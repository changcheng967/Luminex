# Box test of NNUE_FT_ACT (branch fused-act): the feature layer's bias, SCReLU and side-to-move write fused into the
# Triton kernels. ONE visible cell, ~25 minutes, no training run. Writes only under /marimo/fatest (delete it
# afterwards); /marimo/data and /marimo/out are only read. It refuses to start while a run is alive or the GPU holds
# more than 2 GB.
#  1. check_fused.py --act on the first t91 part with the float checkpoint: the activations h and the gradient that
#     goes into the table's kernel bit for bit against forward()'s elementwise ops, in float and under QAT, for one
#     consumer and for two (the draw head's case); then fwd_loss end to end, ACT=0 against ACT=1.
#  2. The trainer's own profile (NNUE_PROFILE: 100 steps after 300 of warm-up), the settings of the 2026-11-15
#     profile except 0.6B positions instead of 1.5B (0.5B on the card, 0.1B in host RAM): the float phase (L1 768,
#     NNUE_FT_DENSE=3, the draw head, the C++ feeder) with NNUE_FT_ACT=0 and =1; QAT from the qatstart checkpoint on
#     NNUE_FT_DENSE=2 (dense fp32, as today), =3 (fused exact, PR #86) and =3 with NNUE_FT_ACT=1.
# The summary at the end: per run, ms a step (wall and GPU) and the time by kind; the losses of the ACT=0 and =1 runs
# side by side (equal up to the table kernel's atomics, ~1e-7 a step); the PASS/FAIL lines.
import collections
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time

import marimo as mo

COMMIT = "e6da953cd6375ddf672c924e300182b2a14c4d37"   # branch fused-act (on PR #86's cb44cca)
HOME = "/marimo"
TEST = HOME + "/fatest"
LOG = TEST + "/fatest.log"
FLOAT_CK = HOME + "/out/luminex_gen768s_02.pt"           # the float phase's checkpoint (with the draw head)
QAT_CK = HOME + "/out/luminex_gen768s_02_qatstart.pt"    # the same without the head: QAT's start
REFRESH_SECONDS = 5

# The cells' draw head (v5r2.py's PATCH_DRAW, verbatim), applied to this branch's trainer for the float runs.
PATCH_DRAW = textwrap.dedent('''
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    rt = open(src, encoding="utf-8").read()


    def patch(old, new, what):
        global rt
        if rt.count(old) != 1:
            sys.exit(f"draw patch: {what} appears {rt.count(old)} times")
        rt = rt.replace(old, new)


    patch("def forward(model, wf, bf, stm, cnt, acc=None):",
          "def forward(model, wf, bf, stm, cnt, acc=None, want_h=False):", "forward()")
    patch("    return F.linear(tail_act(z), Wo, bo).gather(1, bucket).view(n).float() * 300.0\\n",
          "    _o = F.linear(tail_act(z), Wo, bo).gather(1, bucket).view(n).float() * 300.0\\n"
          "    return (_o, h) if want_h else _o\\n", "the value's return")
    patch('def fwd_loss(model, rec):\\n'
          '    wf, bf, stm, tgt, flag, cnt = expand(rec)\\n'
          '    acc = ft_fused(wf, bf, model.ft.weight) if FT_DENSE == 3 else None\\n'
          '    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\\n'
          '        pred = forward(model, wf, bf, stm, cnt, acc)\\n'
          '    return batch_loss(pred.float(), tgt.clamp(-TARGET_CLIP, TARGET_CLIP), flag)\\n',
          'DRAW_W = float(os.environ.get("NNUE_DRAW_W", "0"))\\n'
          '\\n'
          '\\n'
          'def draw_logit(model, h, pred, cnt):\\n'
          '    """The draw head: one logit per output bucket from the activations and the value\\'s size\\n'
          '    (detached: the draw loss shapes the accumulator, not the value\\'s output directly)."""\\n'
          '    n = h.shape[0]\\n'
          '    bucket = ((cnt - 1) * model.num_buckets // 32).clamp(0, model.num_buckets - 1).view(n, 1)\\n'
          '    z = model.dhead(h).float().gather(1, bucket).view(n)\\n'
          '    return z + model.dhead_p[bucket.view(n)] * (pred.detach().float().abs() / 400.0)\\n'
          '\\n'
          '\\n'
          'def fwd_loss(model, rec):\\n'
          '    wf, bf, stm, tgt, flag, cnt = expand(rec)\\n'
          '    acc = ft_fused(wf, bf, model.ft.weight) if FT_DENSE == 3 else None\\n'
          '    dh = getattr(model, "dhead", None)\\n'
          '    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\\n'
          '        if dh is None:\\n'
          '            pred = forward(model, wf, bf, stm, cnt, acc)\\n'
          '        else:\\n'
          '            pred, h = forward(model, wf, bf, stm, cnt, acc, want_h=True)\\n'
          '            dlogit = draw_logit(model, h, pred, cnt)\\n'
          '    vloss = batch_loss(pred.float(), tgt.clamp(-TARGET_CLIP, TARGET_CLIP), flag)\\n'
          '    if dh is None:\\n'
          '        return vloss\\n'
          '    fl = rec[:, 26].to(torch.int64)\\n'
          '    known = ((fl >> 3) & 1).float()\\n'
          '    drawn = (((fl >> 4) & 3) == 1).float()\\n'
          '    bce = F.binary_cross_entropy_with_logits(dlogit, drawn, reduction="none")\\n'
          '    return vloss + DRAW_W * (bce * known).sum() / known.sum().clamp(min=1.0)\\n', "fwd_loss()")
    patch("    model = LNNUE(L1=L1).to(dev)\\n",
          "    model = LNNUE(L1=L1).to(dev)\\n"
          "    if DRAW_W > 0:\\n"
          "        model.dhead = torch.nn.Linear(2 * L1, model.num_buckets).to(dev)\\n"
          "        torch.nn.init.zeros_(model.dhead.weight)\\n"
          "        torch.nn.init.zeros_(model.dhead.bias)\\n"
          "        model.dhead_p = torch.nn.Parameter(torch.zeros(model.num_buckets, device=dev))\\n"
          '        print(f"[draw] draw head on, loss weight {DRAW_W:g}", flush=True)\\n', "model construction")
    patch("    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\\n",
          '    if hasattr(model, "dhead"):   # its own group, the last: no weight decay, the tail learning rate\\n'
          '        groups.append({"params": list(model.dhead.parameters()) + [model.dhead_p], "weight_decay": 0.0,\\n'
          '                       "lr": LR * TAIL_LR_MULT})\\n'
          "    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\\n", "the optimizer line")
    open(dst, "w", encoding="utf-8").write(rt)
    print(f"  draw head: patched trainer {dst}")
''')

SCRIPT = r"""
set -uo pipefail
HOME_DIR="$1"; TEST="$2"; COMMIT="$3"; FLOAT_CK="$4"; QAT_CK="$5"; PY="$6"
mkdir -p "$TEST/tools" "$TEST/out"
sha() { sha256sum "$1" | cut -d' ' -f1; }
ok() { [ -f "$1" ] && [ "$(sha "$1")" = "$2" ]; }
echo "== 1/4: environment"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader || { echo "FAIL: no GPU"; exit 1; }
"$PY" -c 'import torch, triton; print("  torch", torch.__version__, "| triton", triton.__version__, "|", torch.cuda.get_device_name(0))'
FREE_GB=$(df -BG --output=avail "$TEST" | tail -n 1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 5 ] || { echo "FAIL: $TEST has ${FREE_GB} GB free; the test writes ~2 GB"; exit 1; }
DATA="$HOME_DIR/data"
ls $DATA/t91_2026-0[567].*.lcr > /dev/null 2>&1 || { echo "FAIL: no $DATA/t91_2026-0[567].*.lcr"; exit 1; }
ls $DATA/jantail.*.lcr > /dev/null 2>&1 || { echo "FAIL: no $DATA/jantail.*.lcr"; exit 1; }
[ -f "$FLOAT_CK" ] || { echo "FAIL: no $FLOAT_CK"; exit 1; }
echo "  data: $(ls $DATA/t91_2026-0[567].*.lcr | wc -l) LCR1 parts of May-Jul, the January tail; float start $FLOAT_CK"
echo "== 2/4: code at ${COMMIT:0:7} (branch fused-act)"
GH=https://raw.githubusercontent.com/changcheng967/Luminex/$COMMIT/nnue/openi_upload
for pair in "resident_train.py 62b59768bda410ccc7bbb05463471c139898d13a9de6d1f7437c931782c96539" "ft_fused.py dccb4415a925f4c0625e484cf8b68e1b92f232e923aac3399a2217fe0622ebef" "check_fused.py a918ecd79de2f99cf488a8df5771aaf320e004f6c40ee451108cd0fb826238a7" "luminex_nnue_train.py ea5b94f2026c82b7a111d54d75fcf87fc30a8986290332b3dd446bb64f1222a8" "feeder.cpp e01bbbf99c8bd712b3fe28cdb90287e91f310b3eba8ebd922ece92aabfed8779"; do
    set -- $pair
    ok "$TEST/tools/$1" "$2" || curl -fsSL --retry 5 -o "$TEST/tools/$1" "$GH/$1" || curl -fsSL --retry 5 -o "$TEST/tools/$1" "https://gh-proxy.com/$GH/$1" || true
    ok "$TEST/tools/$1" "$2" || { echo "FAIL: $1 from GitHub does not match sha256 $2"; exit 1; }
done
echo "  code: sha256 ok"
"$PY" "$TEST/tools/patch_draw.py" "$TEST/tools/resident_train.py" "$TEST/tools/resident_train_dh.py" || { echo "FAIL: the draw patch"; exit 1; }
echo "== 3/4: check_fused.py --act (bit for bit, float and QAT)"
"$PY" "$TEST/tools/check_fused.py" --act "$DATA" "$FLOAT_CK" || echo "FAIL: check_fused.py --act (exit $?)"
echo "== 4/4: the trainer's profile, 100 steps after 300"
BASE=("NNUE_DATA=$DATA/t91_2026-0[567].*.lcr" "NNUE_VAL_DATA=$DATA/jantail.*.lcr" NNUE_CONTINUE=0 NNUE_FEED_THREADS=8
      NNUE_BUCKETS=8 NNUE_TAIL_LR_MULT=4 NNUE_L1=768 NNUE_BS=131072 NNUE_T_MAX_STEPS=100000 NNUE_BUDGET_SEC=0
      NNUE_LOSS_POWER=3.0 NNUE_QUIET_LOSS=1,0.3,0 NNUE_GRAD_CLIP=1.0 NNUE_TARGET_CLIP=1500 NNUE_VAL_FRAC=0.002
      NNUE_EVAL_EVERY=1000000 NNUE_SAVE_EVERY=1000000 NNUE_KEEP_EVERY=0 NNUE_COMPILE=1 NNUE_VRAM_POS=500000000
      NNUE_MAX_POS=600000000 NNUE_SEED=6 "NNUE_OUT_DIR=$TEST/out" NNUE_PROFILE=100 NNUE_PROFILE_WARMUP=300)
run() {   # title, script, settings...
    local title="$1" script="$2"; shift 2
    echo "== RUN $title"
    env -u NNUE_BUCKET_MAP -u NNUE_PTOWER PYTHONUNBUFFERED=1 "${BASE[@]}" "$@" "$PY" -u "$TEST/tools/$script" || echo "FAIL: $title (exit $?)"
}
FLOAT=("NNUE_RESUME=$FLOAT_CK" NNUE_LR=5e-5 NNUE_QAT=0 NNUE_FT_DENSE=3 NNUE_DRAW_W=1e-3)
run "float, NNUE_FT_ACT=0" resident_train_dh.py "${FLOAT[@]}" NNUE_FT_ACT=0 NNUE_OUT_NAME=fa_float0.nnue
run "float, NNUE_FT_ACT=1" resident_train_dh.py "${FLOAT[@]}" NNUE_FT_ACT=1 NNUE_OUT_NAME=fa_float1.nnue
if [ -f "$QAT_CK" ]; then
    QAT=("NNUE_RESUME=$QAT_CK" NNUE_LR=2e-5 NNUE_QAT=1 NNUE_QAT_FT8=0 NNUE_DRAW_W=0)
    run "QAT, NNUE_FT_DENSE=2" resident_train.py "${QAT[@]}" NNUE_FT_DENSE=2 NNUE_OUT_NAME=fa_qat2.nnue
    run "QAT, NNUE_FT_DENSE=3, NNUE_FT_ACT=0" resident_train.py "${QAT[@]}" NNUE_FT_DENSE=3 NNUE_FT_ACT=0 NNUE_OUT_NAME=fa_qat30.nnue
    run "QAT, NNUE_FT_DENSE=3, NNUE_FT_ACT=1" resident_train.py "${QAT[@]}" NNUE_FT_DENSE=3 NNUE_FT_ACT=1 NNUE_OUT_NAME=fa_qat31.nnue
else echo "FAIL: no $QAT_CK: the QAT runs were skipped"; fi
echo "== FINISHED (delete $TEST when the numbers are copied)"
"""


def busy():
    """A training or conversion process, or a GPU in use: the reason not to start, else None."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith((b"resident_train.py", b"resident_train_dh.py", b"c500_train_stream.py")) for a in argv) or \
                any(a.endswith((b"binpack2lcr", b"luminex-featurize")) for a in argv[:1]):
            return "a run is alive: " + b" ".join(argv).decode(errors="replace")[:200]
    try:
        used = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=20).stdout.split()[0])
        if used > 2048:
            return f"the GPU holds {used:.0f} MiB"
    except Exception as e:
        return f"nvidia-smi failed ({e.__class__.__name__})"
    return None


def summary(lines):
    """Per run: ms a step and the time by kind; ACT=0 against ACT=1 losses; PASS/FAIL lines."""
    out, run, prof, losses, health, top = [], None, collections.defaultdict(list), {}, {}, collections.defaultdict(list)
    for line in lines:
        if line.startswith("== RUN "):
            run = line[7:]
            losses[run] = {}
        m = re.search(r" step (\d+) loss=([0-9.]+)", line)
        if m and run:
            losses[run][int(m.group(1))] = float(m.group(2))
        m = re.match(r"\[HEALTH step (\d+)\] val loss=([0-9.]+)", line)
        if m and run:
            health.setdefault(run, []).append(f"{m.group(1)}: {m.group(2)}")
        if line.startswith("[profile]") and run:
            body = line[9:].rstrip()
            if "top kernels" in body:
                top[run] = []
            elif run in top:
                if len(top[run]) < 12:
                    top[run].append(body.strip())
            else:
                prof[run].append(body.strip())
        if not line.startswith("== ") and any(k in line for k in ("ACT_PASS", "ACT_FAIL", "bit for bit", "fwd_loss:",
                                                                  "FAIL", "DONE (")):
            out.append(line.strip())
    for r in prof:
        out.append(f"-- {r}")
        out.extend("   " + x for x in prof[r])
        out.extend("   top: " + x for x in top.get(r, []))
        if r in health:
            out.append("   [HEALTH] " + " | ".join(health[r]))
    for a, b in (("float, NNUE_FT_ACT=0", "float, NNUE_FT_ACT=1"),
                 ("QAT, NNUE_FT_DENSE=3, NNUE_FT_ACT=0", "QAT, NNUE_FT_DENSE=3, NNUE_FT_ACT=1"),
                 ("QAT, NNUE_FT_DENSE=2", "QAT, NNUE_FT_DENSE=3, NNUE_FT_ACT=1")):
        common = sorted(set(losses.get(a, {})) & set(losses.get(b, {})))
        if common:
            worst = max(abs(losses[a][s] - losses[b][s]) for s in common)
            out.append(f"losses, {a} vs {b}: {len(common)} steps, largest difference {worst:.5f} "
                       f"(step {common[-1]}: {losses[a][common[-1]]:.5f} vs {losses[b][common[-1]]:.5f})")
    return out


os.makedirs(TEST + "/tools", exist_ok=True)
why = busy()
if why:
    mo.output.replace(mo.plain_text(f"NOT STARTED: {why}. Run this cell when the box is free."))
else:
    with open(TEST + "/tools/patch_draw.py", "w", encoding="utf-8") as f:
        f.write(PATCH_DRAW)
    state = {"last": time.time(), "phase": "starting"}
    recent, every = collections.deque(maxlen=4000), []
    lock = threading.Lock()

    def read_output(pipe):
        log = open(LOG, "a", buffering=1, errors="replace")
        for line in pipe:
            log.write(line)
            text = line.rstrip("\n")
            with lock:
                state["last"] = time.time()
                if text.startswith("== "):
                    state["phase"] = text[3:]
                recent.append(text)
                every.append(text)
        log.close()

    started = time.time()
    proc = subprocess.Popen(["bash", "-c", SCRIPT, "bash", HOME, TEST, COMMIT, FLOAT_CK, QAT_CK, sys.executable],
                            env=dict(os.environ, PYTHONUNBUFFERED="1"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace", start_new_session=True)
    reader = threading.Thread(target=read_output, args=(proc.stdout,), daemon=True)
    reader.start()

    def panel(status):
        with lock:
            shown, s = list(recent)[-400:], dict(state)
        up = int(time.time() - started)
        return mo.plain_text(f"fused-act box test: {status} {up // 60}m {up % 60:02d}s · phase: {s['phase']} · "
                             f"last output {int(time.time() - s['last'])} s ago · log {LOG}\n" + "─" * 46 + "\n"
                             + "\n".join(shown))

    try:
        while True:
            mo.output.replace(panel("running"))
            try:
                proc.wait(timeout=REFRESH_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
        reader.join(timeout=5)
        with lock:
            lines = list(every)
        result = summary(lines)
        with open(LOG, "a") as f:
            f.write("== SUMMARY\n" + "\n".join(result) + "\n")
        for text in result:
            print(text, flush=True)
        mo.output.replace(mo.plain_text(f"fused-act box test finished (exit {proc.returncode}) after "
                                        f"{int(time.time() - started) // 60} min\n" + "\n".join(result)))
    finally:
        if proc.poll() is None:   # the cell was stopped: stop the test
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
