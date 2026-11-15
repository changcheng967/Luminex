# Box test of the binpack feeder (PR #85) and the two QAT commits (PR #86, draft): ONE visible cell,
# ~20 minutes, no training run. Writes only under /marimo/bftest (delete it afterwards); /marimo/data,
# /marimo/tools, /marimo/out and /marimo/ckpt are only read. It refuses to start while a run is alive
# (resident_train, binpack2lcr, the featurizer) or the GPU holds more than 2 GB.
#  1. check_binfeed.py on the August 2026 BT4 binpack: exactness against binpack2lcr, the reservoir on the GPU,
#     the box's read and decode rates at 1-20 threads, the stream's delivery rate.
#  2. Speed, from gen768s_01's float net on that month: 1,500 float steps on NNUE_FT_DENSE=3, then 1,500 QAT
#     steps in the same process (NNUE_QAT_AFTER, the fused exact path; the reservoir is the whole month,
#     ~4.4 GB of VRAM, a >2^31-element tensor); then 1,500 QAT steps as today (a separate process on =2).
#  3. check_fused.py --qat on the month's records with the float phase's .pt: the fused exact forward against
#     the dense fp32 one, bit for bit.
# The summary at the end: steps/s per phase, the switch's cost, the QAT start readings, PASS/FAIL lines.
import collections
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

import marimo as mo

COMMIT = "cb44ccae505a80b5789ef2d9785f0a301d25e3ad"   # branch binpack-qat (PR #86, on PR #85's 0cb8a8f)
NP_COMMIT = "ed84cc1fd2255faa0f33dcfa7839c9b6b4bb49b8"  # nnue-pytorch: the binpack reader binpack2lcr was verified with
MONTH = "2026-08"     # BT4: 432,909,354 bytes, ~168M positions
STEPS = 1500          # steps of each timed phase
CONNECTIONS = 8
HOME = "/marimo"
TEST = HOME + "/bftest"
LOG = TEST + "/bftest.log"
REFRESH_SECONDS = 5

SCRIPT = r"""
set -uo pipefail
HOME_DIR="$1"; TEST="$2"; COMMIT="$3"; NP_COMMIT="$4"; MONTH="$5"; STEPS="$6"; CONN="$7"; PY="$8"
mkdir -p "$TEST/tools" "$TEST/data" "$TEST/out" "$TEST/lcr" "$TEST/work"
sha() { sha256sum "$1" | cut -d' ' -f1; }
ok() { [ -f "$1" ] && [ "$(sha "$1")" = "$2" ]; }
echo "== 1/5: environment"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader || { echo "FAIL: no GPU"; exit 1; }
echo "  CPU threads: $(nproc), RAM: $(free -g | awk '/^Mem/ {print $2}') GB, $(g++ --version | head -n 1)"
"$PY" -c 'import torch, triton; print("  torch", torch.__version__, "| triton", triton.__version__, "|", torch.cuda.get_device_name(0))'
FREE_GB=$(df -BG --output=avail "$TEST" | tail -n 1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 15 ] || { echo "FAIL: $TEST has ${FREE_GB} GB free; the test needs ~10 GB at its peak"; exit 1; }
echo "== 2/5: code at ${COMMIT:0:7} (branch binpack-qat), nnue-pytorch ${NP_COMMIT:0:7}"
GH=https://raw.githubusercontent.com/changcheng967/Luminex/$COMMIT/nnue/openi_upload
for pair in "resident_train.py 978617664877793aab2d6b4885396172c2e9234c48f5c4bfbfeb691c804d316a" "ft_fused.py 6db6cfc39c388b01b42fd9fe1f0d82e806e7964a9d2d2c327ca7de08f418968c" "check_fused.py a0e5d71f0eb52e8435be08e309886a648721554a0809360645af1670a256d837" "binfeed.cpp 0a0432fe17b2b7a1f9042d7490c1f450245851da31820cc85f65391bde62d850" "check_binfeed.py 642506187a58d0e9cb19c006acaa76068ba88719336104fc96f769af342d6d93" "feeder.cpp e01bbbf99c8bd712b3fe28cdb90287e91f310b3eba8ebd922ece92aabfed8779" "luminex_nnue_train.py ea5b94f2026c82b7a111d54d75fcf87fc30a8986290332b3dd446bb64f1222a8" "binpack2lcr.cpp 542f7448adc8bea54c6ab31ce78ed59dba2922b2d568f86149fce661a2ae12d7" "pfetch.py cf24acc2e7482c8afb2862528e2aed6a1b7e4ce8e0e5f7e244e3e98e117a0ccb"; do
    set -- $pair
    ok "$TEST/tools/$1" "$2" || curl -fsSL --retry 5 -o "$TEST/tools/$1" "$GH/$1" || curl -fsSL --retry 5 -o "$TEST/tools/$1" "https://gh-proxy.com/$GH/$1" || true
    ok "$TEST/tools/$1" "$2" || { echo "FAIL: $1 from GitHub does not match sha256 $2"; exit 1; }
done
echo "  code: sha256 ok"
NP="$HOME_DIR/tools/nnue-pytorch"
if [ ! -f "$NP/data_loader/cpp/lib/binpack.h" ]; then
    NP="$TEST/tools/nnue-pytorch"; mkdir -p "$NP"
    curl -fsSL --retry 5 "https://codeload.github.com/official-stockfish/nnue-pytorch/tar.gz/$NP_COMMIT" | tar xz -C "$NP" --strip-components=1
fi
[ -f "$NP/data_loader/cpp/lib/binpack.h" ] || { echo "FAIL: nnue-pytorch headers missing"; exit 1; }
export NNUE_PYTORCH="$NP"
g++ -O2 -march=native -std=c++20 -pthread -I "$NP/data_loader/cpp" -o "$TEST/tools/binpack2lcr" "$TEST/tools/binpack2lcr.cpp" || { echo "FAIL: binpack2lcr build"; exit 1; }
echo "  binpack2lcr: built; binfeed.cpp builds on first use"
echo "== 3/5: the $MONTH BT4 binpack"
case ${MONTH#*-} in 01) mn=jan;; 02) mn=feb;; 03) mn=mar;; 04) mn=apr;; 05) mn=may;; 06) mn=jun;; 07) mn=jul;; 08) mn=aug;; 09) mn=sep;; 10) mn=oct;; 11) mn=nov;; 12) mn=dec;; esac
F="test91-$MONTH-$mn-150gb6p.v6.relabel-BT4-tf13tune-histfix.binpack"
read -r size oid < <("$PY" -c "import json, sys, urllib.parse, urllib.request; print(*(lambda e: (e['size'], e['lfs']['oid']))(json.load(urllib.request.urlopen(urllib.request.Request('https://huggingface.co/api/datasets/xushawn/t91-lgs/paths-info/main', data=urllib.parse.urlencode([('paths', 'binpacks/v6-bt4/' + sys.argv[1]), ('expand', 'true')]).encode()), timeout=60))[0]))" "$F")
if [ "$(stat -c %s "$TEST/data/$F" 2> /dev/null || echo 0)" != "$size" ] || [ -f "$TEST/data/$F.pieces" ]; then
    "$PY" "$TEST/tools/pfetch.py" "https://huggingface.co/datasets/xushawn/t91-lgs/resolve/main/binpacks/v6-bt4/$F" "$TEST/data/$F" "$size" "$CONN" || { echo "FAIL: download"; exit 1; }
    rm -f "$TEST/data/$F.pieces"
fi
[ "$(sha "$TEST/data/$F")" = "$oid" ] || { rm -f "$TEST/data/$F"; echo "FAIL: $F sha256 differs from Hugging Face's"; exit 1; }
echo "  $F: $size bytes, sha256 ok"
echo "== 4/5: check_binfeed.py: exactness, the reservoir on the GPU, read and decode rates"
"$PY" "$TEST/tools/check_binfeed.py" "$TEST/data/$F" --work "$TEST/work" --threads 1,2,4,10,20 || echo "FAIL: check_binfeed.py (exit $?)"
echo "== 5/5: speed, from gen768s_01's float net: float on NNUE_FT_DENSE=3, QAT in the same process (fused exact), then QAT as today (=2)"
CK=""; [ -f "$HOME_DIR/ckpt/luminex_gen768s_01.nnue" ] && CK="$HOME_DIR/ckpt/luminex_gen768s_01.nnue"
echo "  start: ${CK:-random weights}"
COMMON="NNUE_BUCKETS=8 NNUE_TAIL_LR_MULT=4 NNUE_L1=768 NNUE_BS=131072 NNUE_LOSS_POWER=3.0 NNUE_QUIET_LOSS=1,0.3,0 NNUE_GRAD_CLIP=1.0 NNUE_TARGET_CLIP=1500 NNUE_VAL_FRAC=0.002 NNUE_COMPILE=1 NNUE_OUT_DIR=$TEST/out NNUE_BINPACKS=$TEST/data/$F"
echo "== PHASE float + in-process QAT"
env PYTHONUNBUFFERED=1 $COMMON ${CK:+NNUE_RESUME=$CK} NNUE_LR=5e-5 NNUE_T_MAX_STEPS="$STEPS" NNUE_EVAL_EVERY="$STEPS" NNUE_SAVE_EVERY="$STEPS" NNUE_QAT=0 NNUE_FT_DENSE=3 NNUE_QAT_AFTER="$STEPS" NNUE_QAT_LR=2e-5 NNUE_QAT_EVERY="$STEPS" NNUE_SEED=6 NNUE_OUT_NAME=bft.nnue "$PY" -u "$TEST/tools/resident_train.py" || echo "FAIL: float + in-process QAT (exit $?)"
echo "== PHASE separate QAT on NNUE_FT_DENSE=2"
if [ -f "$TEST/out/bft.pt" ]; then
    env PYTHONUNBUFFERED=1 $COMMON NNUE_RESUME="$TEST/out/bft.pt" NNUE_LR=2e-5 NNUE_T_MAX_STEPS="$STEPS" NNUE_EVAL_EVERY="$STEPS" NNUE_SAVE_EVERY="$STEPS" NNUE_QAT=1 NNUE_FT_DENSE=2 NNUE_SEED=4 NNUE_OUT_NAME=bft_q2.nnue "$PY" -u "$TEST/tools/resident_train.py" || echo "FAIL: separate QAT (exit $?)"
else echo "FAIL: no float checkpoint for the separate QAT phase"; fi
echo "== PHASE check_fused.py --qat on the month's records"
"$TEST/tools/binpack2lcr" "$TEST/data/$F" "$TEST/lcr/t91_$MONTH" "$(nproc)" 2> "$TEST/lcr/convert.log" || echo "FAIL: convert"
"$PY" "$TEST/tools/check_fused.py" --qat "$TEST/lcr" $( [ -f "$TEST/out/bft.pt" ] && echo "$TEST/out/bft.pt" ) || echo "FAIL: check_fused.py --qat (exit $?)"
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
    """Steps/s per phase (median of the 50-step readings after the first 300 steps), the switch's
    cost, the QAT start readings, and the PASS/FAIL lines."""
    out, phase, rates, health, switch = [], None, collections.defaultdict(list), {}, {}
    for line in lines:
        if line.startswith("== PHASE"):
            phase = line[9:]
        elif "== QAT PHASE in this run" in line:
            phase, switch["t"] = "in-process QAT (fused, =3)", time.time()
        m = re.search(r"step (\d+) loss=.* \| ([0-9.]+) steps/s", line)
        if m and phase:
            rates[phase].append((int(m.group(1)), float(m.group(2))))
        m = re.match(r"\[HEALTH step (\d+)\] val loss=([0-9.]+) MAE=([0-9.]+)cp", line)
        if m and phase and phase not in health:
            health[phase] = f"step {m.group(1)}: val loss {m.group(2)}, MAE {m.group(3)} cp"
        if any(k in line for k in ("BINFEED_", "GRAD_", "QAT feature layer", "FAIL", "decode,", "stream,", "read into RAM",
                                   "reservoir:", "DONE (")):
            out.append(line.strip())
    for p, r in rates.items():
        first = r[0][0] if r else 0
        late = [x for s, x in r if s >= first + 300] or [x for _, x in r]
        if late:
            out.append(f"steps/s, {p}: median {statistics.median(late):.1f} over {len(late)} readings")
    for p, h in health.items():
        out.append(f"first [HEALTH], {p}: {h}")
    return out


os.makedirs(TEST, exist_ok=True)
why = busy()
if why:
    mo.output.replace(mo.plain_text(f"NOT STARTED: {why}. Run this cell when the box is free."))
else:
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
    proc = subprocess.Popen(["bash", "-c", SCRIPT, "bash", HOME, TEST, COMMIT, NP_COMMIT, MONTH, str(STEPS),
                             str(CONNECTIONS), sys.executable],
                            env=dict(os.environ, PYTHONUNBUFFERED="1"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace", start_new_session=True)
    reader = threading.Thread(target=read_output, args=(proc.stdout,), daemon=True)
    reader.start()

    def panel(status):
        with lock:
            shown, s = list(recent)[-400:], dict(state)
        up = int(time.time() - started)
        return mo.plain_text(f"binfeed box test: {status} {up // 60}m {up % 60:02d}s · phase: {s['phase']} · "
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
        mo.output.replace(mo.plain_text(f"binfeed box test finished (exit {proc.returncode}) after "
                                        f"{int(time.time() - started) // 60} min\n" + "\n".join(result)))
    finally:
        if proc.poll() is None:   # the cell was stopped: stop the test
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
