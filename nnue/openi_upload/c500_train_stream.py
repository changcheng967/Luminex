#!/usr/bin/env python3
"""C500 PER-FRAME trainer (VRAM-resident, OOM-safe). Featurize one frame in <=CAP-position
loads straight to VRAM -> train -> free -> next load (big frames auto-split into parts).
bf16 + grad-clip (v6 fix). Clean per-frame output. No whole-frame VRAM spike (cat frees
each list immediately; PYTORCH_CUDA_ALLOC_CONF=expandable_segments kills fragmentation).

LOSS (v12, cross-checked against SF nnue-pytorch / Berserk / Seer / Leela source):
  |sigma(pred/400) - sigma(target/400)|^NNUE_LOSS_POWER   (default power 2.6)
  + optional SF position weighting via NNUE_POS_W1/NNUE_POS_W2 (default OFF = SF default)
  + lambda is implicitly 1.0 (pure eval): our gamepack stores no game results, and SF's
    own schedule ends at end-lambda 0.75-1.0 with much noisier shallow-search data.

RUN (no args):
  NNUE_L1=512 NNUE_EPOCHS=1 NNUE_BS=131072 NNUE_LR=1e-3 NNUE_FEAT_THREADS=14 NNUE_GRAD_CLIP=1.0
  NNUE_VRAM_POS=180000000   # max positions per VRAM load (big frames split into parts)

PASS-4 COMMAND (Gen768 warm restart on the same 4.29B pack, DOSL trained):
  NNUE_L1=768 NNUE_OUT_NAME=luminex_gen768p4.nnue NNUE_RESUME=/tmp/code/luminex_gen768.pt \
  NNUE_LR=1e-4 NNUE_BS=131072 NNUE_EPOCHS=1 NNUE_T_MAX_STEPS=33000 \
  NNUE_FRAME_SHUFFLE=1 NNUE_SHUFFLE_SEED=43 \
  NNUE_TAIL_LR_MULT=3 \
  NNUE_SWA_START_STEP=24750 NNUE_LOSS_POWER=2.6 python c500_train_stream.py
  # first session prints "optimizer rebuild" (3-group ckpt vs new 4-group opt) — expected
  # after: python quantize_i8.py luminex_gen768p4.nnue luminex_gen768p4_i8.nnue
"""
import os, sys, subprocess, time, glob, copy
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # anti-fragmentation
import numpy as np, torch
try:
    from c2net.context import prepare, upload_output
    ctx = prepare(); out_dir = ctx.output_path
except Exception:   # non-OpenI environment (smoke test / local box)
    class _Ctx: output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    ctx = _Ctx(); out_dir = ctx.output_path
    os.makedirs(out_dir, exist_ok=True)
    upload_output = lambda *a, **k: None

SEARCH_ROOTS = ["/tmp/code", "/tmp/dataset", "/tmp/frames", "/tmp"]
def _walk():
    for root in SEARCH_ROOTS:
        if os.path.isdir(root):
            for dp, dn, fn in os.walk(root, followlinks=True):
                yield dp, fn
def _find_file(name):
    for dp, fn in _walk():
        if name in fn: return os.path.join(dp, name)
    return None

FRAMES_DIR = os.environ.get("NNUE_FRAMES_DIR", "/tmp/code/frames")
FRAMES = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.zst")) + glob.glob(os.path.join(FRAMES_DIR, "frame_*.xz")))
if not FRAMES:
    for dp, _ in _walk():
        FRAMES += sorted(glob.glob(os.path.join(dp, "frame_*.zst")) + glob.glob(os.path.join(dp, "frame_*.xz")))
    FRAMES = sorted(set(FRAMES))
if not FRAMES:
    tar = _find_file("gamepack.tar")
    if tar:
        os.makedirs(FRAMES_DIR, exist_ok=True)
        print(f"  extracting {tar} -> {FRAMES_DIR}", flush=True)
        subprocess.run(f"tar xf '{tar}' -C '{FRAMES_DIR}'", shell=True)
        FRAMES = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.zst")) + glob.glob(os.path.join(FRAMES_DIR, "frame_*.xz")))
# SF-relabel binpack path: NNUE_FENEVAL_FILE points at a fen<TAB>white_cp tsv;
# featurize --fen-eval streams 140B records exactly like the frame pipeline.
_FENEVAL = os.environ.get("NNUE_FENEVAL_FILE")
if _FENEVAL:
    FRAMES = [_FENEVAL]
else:
    assert FRAMES, "no frame_*.zst/.xz found and no gamepack.tar"
# Pass-2+ (warm restart): shuffle frame order to break the temporal bias of the
# newest-first dataset (with cosine annealing, the LAST frames dominate the
# low-LR consolidation phase). Deterministic seed => reproducible order.
# NNUE_SHUFFLE_SEED: vary per pass (pass3=42, pass4=43, ...) — reusing a pass's
# seed replays the identical presentation order and weakens the epoch.
if os.environ.get("NNUE_FRAME_SHUFFLE", "0") == "1":
    import random as _rnd
    _shuf_seed = int(os.environ.get("NNUE_SHUFFLE_SEED", "42"))
    _rnd.Random(_shuf_seed).shuffle(FRAMES)
    print(f"[shuffle] frame order shuffled (seed {_shuf_seed}): first={os.path.basename(FRAMES[0])} last={os.path.basename(FRAMES[-1])}", flush=True)
if any(f.endswith(".zst") for f in FRAMES) and not os.path.exists("/usr/bin/zstd"):
    subprocess.run("apt-get install -y zstd >/dev/null 2>&1 || pip install -q zstandard", shell=True)

train_mod = _find_file("luminex_nnue_train.py")
CODE_DIR = os.environ.get("CODE_DIR") or (os.path.dirname(train_mod) if train_mod else "/tmp/code")
sys.path.insert(0, CODE_DIR)
FEAT = os.environ.get("FEAT") or _find_file("luminex-featurize") or os.path.join(CODE_DIR, "luminex-featurize")
try:  # OpenI's code preparation strips execute bits; restore rather than crash
    if os.path.exists(FEAT) and not os.access(FEAT, os.X_OK):
        os.chmod(FEAT, 0o755)
except OSError:
    pass
assert os.path.exists(FEAT), f"featurizer missing: {FEAT}"
from luminex_nnue_train import LNNUE, save_nnue, NUM_INPUTS
_ANCHOR_W = float(os.environ.get("NNUE_ANCHOR_W", "0"))

L1   = int(os.environ.get("NNUE_L1", "512"))
BS   = int(os.environ.get("NNUE_BS", "131072"))
LR   = float(os.environ.get("NNUE_LR", "1e-3"))
NTH  = int(os.environ.get("NNUE_FEAT_THREADS", "14"))
BUDGET = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.5*3600))))
BUF  = int(os.environ.get("NNUE_BUF", "2000000"))
CAP  = int(os.environ.get("NNUE_VRAM_POS", "180000000"))   # max pos per VRAM load (big frames split)
_gc = float(os.environ.get("NNUE_GRAD_CLIP", "1.0"))   # v6 fix (prevents gradient spikes)
_wc = float(os.environ.get("NNUE_WCLAMP", "0"))          # OFF (fallback only; root cause = tail wd + amsgrad)
# SF position weighting (nnue-pytorch model/nnue.py calculate_sf_loss):
#   w = 1 + (2^w1 - 1) * ((pf-0.5)^2 * pf*(1-pf))^w2, pf = target win probability
# Zero boost at dead-drawn (pf~0.5) and dead-won (pf~0/1); peaks (~1.1-1.4x) at
# decisive-but-uncertain positions — the positions that actually decide games.
# SF's own default is w1=0 (OFF), w2=0.5 (model/config.py LossParams). Kept OFF by
# default here too: only enable with values you intend to A/B.
_POS_W1 = float(os.environ.get("NNUE_POS_W1", "0"))    # 0 = OFF (SF default)
_POS_W2 = float(os.environ.get("NNUE_POS_W2", "0.5"))  # SF default
# Quiet-robust loss (round 3): NNUE_QUIET_LOSS="quiet,capture,check" — weights
# per position class from the dump's flag column. The engine consumes eval at
# quiet nodes only; the tactical tail is label noise (the sf1-vs-full law).
_QUIET_MODE = bool(os.environ.get("NNUE_QUIET_LOSS"))
try:
    _QUIET_WQ, _QUIET_WC, _QUIET_WI = (
        [float(x) for x in os.environ["NNUE_QUIET_LOSS"].split(",")] + [0.0] * 3)[:3]
except (KeyError, ValueError):
    _QUIET_MODE = False
    _QUIET_WQ = _QUIET_WC = _QUIET_WI = 0.0
_QUIET_FIRST_PRINT = True
# DOSL dual-head (Step-0): aux linear head trained alongside the full net so it is
# a standalone-usable qsearch stand-pat. Warmup at lam=1.0 (head alone must be a
# competent eval), then joint lam=0.25 persists as the anchor loss throughout.
# DEFAULT ON since pass-4: the Gen768 pass-3 export shipped a zero-trained LINH
# section (valid format, all-zero weights) that the engine happily used as a 0cp
# qsearch stand-pat — the 0-50 match loss. Export now REFUSES an all-zero head.
# DOSL linear head removed: designed for slow float eval, which no longer exists.
# Pinned cosine horizon (must be defined before the subset-trim below)
_T_MAX_FIXED = int(os.environ["NNUE_T_MAX_STEPS"]) if os.environ.get("NNUE_T_MAX_STEPS") else 0
_CAL_STEPS = 60   # calibration steps to measure throughput
_FEAT_CACHE = os.environ.get("NNUE_FEAT_CACHE", "0") == "1"  # OFF by default: single-pass
# never re-reads a frame, and full-data caching would need ~2.2TB of /tmp. Opt-in
# (still disk-guarded below) only makes sense for small-subset multi-epoch runs.
REC  = 140; SCALE = 400.0   # 140 = w64|b64|stm4|target4|result4 (result: white-rel -1/0/+1, 0 on v1 frames)
# C5 WDL mixing: t' = lam*eval + (1-lam)*RESULT_CP*result_stm, lam annealed
# linearly from NNUE_WDL_LAMBDA to NNUE_WDL_LAMBDA_END over NNUE_WDL_ANNEAL
# steps. Default 1.0 = off (bit-identical to the pre-C5 trainer). result==0
# means draw on labeled (GP2) frames — but ALSO "unlabeled" on v1 frames, so
# only enable on fully-labeled corpora.
_LAM0 = float(os.environ.get("NNUE_WDL_LAMBDA", "1.0"))
_LAM1 = float(os.environ.get("NNUE_WDL_LAMBDA_END", os.environ.get("NNUE_WDL_LAMBDA", "1.0")))
_WDL_ANNEAL = int(os.environ.get("NNUE_WDL_ANNEAL", "40000"))
_RESULT_CP = float(os.environ.get("NNUE_RESULT_CP", "1000.0"))
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("NNUE_OUT_NAME", "luminex_v6.nnue"); OUT_BASE = OUT[:-5] if OUT.endswith(".nnue") else OUT
total_bytes = sum(os.path.getsize(f) for f in FRAMES)
_arch_name = "gen768"
_anchor_note = f" anchor_w={_ANCHOR_W}" if _ANCHOR_W > 0 else ""
print(f"arch: {_arch_name}{_anchor_note}", flush=True)
print(f"per-frame train (OOM-safe): {len(FRAMES)} frames, {total_bytes/1e9:.2f}GB, L1={L1} bs={BS} cap={CAP:,} feat_threads={NTH} grad-clip={_gc} device={device}", flush=True)
print(f"loss: power={os.environ.get('NNUE_LOSS_POWER', '2.6')} in sigmoid(cp/{SCALE:.0f}) space, pos-weight w1={_POS_W1} w2={_POS_W2}{' (OFF)' if _POS_W1 == 0 else ''}, lambda=1.0 (pure eval — gamepack has no game results; SF end-lambda practice)", flush=True)

model = LNNUE(L1=L1).to(device)
gstep = 0  # must exist before resume check (fresh runs would NameError otherwise)
_ck = None  # resume checkpoint dict (kept for SWA state restore further below)
_resume = os.environ.get("NNUE_RESUME") or os.path.join(os.path.dirname(__file__) or ".", OUT_BASE + ".pt")
_opt_state = None  # deferred: optimizer doesn't exist yet at resume time
if os.path.exists(_resume) and os.environ.get("NNUE_RESUME", "1") != "0":
    try:
        if _resume.endswith(".nnue"):
            # Warm start from an engine-format float net (LNN1): rebuild a trainer
            # state_dict so the same expansion/resume paths apply. gstep starts 0.
            from luminex_nnue_train import LNNUE as _LNN
            _ck = {"model": {k: v.to(device) for k, v in _LNN.load_lnn1_state(_resume).items()}, "gstep": 0}
            print(f"[RESUME] warm start from float net {_resume} (gstep fresh)", flush=True)
        else:
            _ck = torch.load(_resume, map_location=device, weights_only=False)
        # arch L1 from the tail's input width (robust to pairwise FTs: the
        # embedding is 2x wide but l2 always sees 2*L1)
        _ck_L1 = _ck["model"]["l2.weight"].shape[1] // 2
        if _ck_L1 != L1:
            # Net2Net widening (Gen v1.3 Step 0): load old-width ckpt into wider model
            from luminex_nnue_train import LNNUE
            LNNUE.net2net_widen(model, _ck["model"], _ck_L1)
            gstep = _ck.get("gstep", 0)
            _opt_state = None  # old moments are shape-incompatible — fresh optimizer
            print(f"[NET2NET] widened {_ck_L1} -> {L1} from {_resume} at gstep={gstep} (optimizer fresh)", flush=True)
        elif (
            _ck["model"]["ft.weight"].shape != tuple(model.ft.weight.shape)
            or _ck["model"]["l2.weight"].shape != tuple(model.l2.weight.shape)
            or (getattr(model, 'num_buckets', 1) > 1 and 'bl2.0.weight' not in _ck["model"])
        ):
            # v2 warm start: pairwise FT duplication and/or tail-width widening
            # and/or bucket seeding — function-preserving (or O(noise^2)) by
            # construction, see LNNUE.v2_expand.
            from luminex_nnue_train import LNNUE
            LNNUE.v2_expand(model, _ck["model"])
            gstep = _ck.get("gstep", 0)
            _opt_state = None
            print(f"[V2] expanded warm start from {_resume} at gstep={gstep} (optimizer fresh)", flush=True)
        else:
            _miss = model.load_state_dict(_ck["model"], strict=False)
            if _miss.missing_keys:
                print(f"[RESUME] new params init fresh: {_miss.missing_keys}", flush=True)
            gstep = _ck.get("gstep", 0)
        # Checkpoints from the pre-padding_idx era may carry a NON-ZERO pad row:
        # export drops that row, so training it = silent Python/engine mismatch.
        try:
            _pad_max = float(model.ft.weight.data[NUM_INPUTS].abs().max().item())
            if _pad_max > 1e-6:
                print(f"[RESUME] pad row non-zero ({_pad_max:.3e}) — forcing zero", flush=True)
                model.ft.weight.data[NUM_INPUTS].zero_()
        except Exception:
            pass
        _opt_state = _ck.get("opt") if _ck_L1 == L1 else None  # only same-width resumes carry moments
        if _opt_state: print(f"[RESUME] + optimizer state ({len(_opt_state['state'])} params) — no warm-up loss", flush=True)
        print(f"[RESUME] loaded {_resume} at gstep={gstep}", flush=True)
    except Exception as _e:
        print(f"[RESUME] FAILED ({_e}) - training from scratch", flush=True)
model.probe_ft(device)   # EmbeddingBag FT
# DOSL warmup baseline: on a CONTINUATION the global gstep already exceeds the
# warmup horizon, so warmup must count from this session's start. A checkpoint
# whose lin head is already trained (non-zero) skips warmup entirely.
print(f"  [LNNUE] ft_mode={model.ft_mode} (compile OFF)", flush=True)
# SWA step-trigger baseline: same global-vs-session trap as the DOSL warmup —
# a resumed run's global gstep dwarfs the threshold, so the trigger must count
# from this session's start (pass-4 averaged its whole pass by accident).
_swa_sess0 = gstep
# Phase 0 root-cause L2 fix: decay ONLY the tail WEIGHTS (where L2/SCReLU feedback
# grows weights), NOT the FT (protects rare king/piece/square buckets from uniform-decay
# undertraining, #45) and NOT any bias (biases are activation operating points).
# Group by NAME: the old numel()>100K split put ft_bias in the decayed tail group and
# would flip l2.weight into the no-decay FT group at L2>=128.
# + AMSGrad (bounds effective LR per-param, prevents any single weight running away).
_ft_params = [model.ft.weight, model.ft_bias]
_tail_w    = [model.l2.weight, model.l3.weight, model.out.weight]
_tail_b    = [model.l2.bias, model.l3.bias, model.out.bias]
_tail_wd = float(os.environ.get("NNUE_TAIL_WD", "1e-2"))
# Per-group LR multipliers (pass-4 levers): the FT is converged after 3 passes,
# the tiny tail can still move (tail mult), and a FRESH DOSL head needs to train
# fast within its warmup window (lin mult). Group order below MUST match _group_lrs.
_TAIL_LR_MULT = float(os.environ.get("NNUE_TAIL_LR_MULT", "1.0"))
opt = torch.optim.AdamW([
    {"params": _ft_params, "weight_decay": 0.0},        # NO decay on FT (rare-bucket protection)
    {"params": _tail_w,    "weight_decay": _tail_wd,    # tail weight decay (fixes L2 SCReLU feedback)
                         "lr": LR * _TAIL_LR_MULT},
    {"params": _tail_b,    "weight_decay": 0.0},        # biases are operating points — never decay
], lr=LR, amsgrad=True)
_group_lrs = [1.0, _TAIL_LR_MULT, 1.0]
if _opt_state:
    try:
        opt.load_state_dict(_opt_state)
        # Reset LR to base: the restored opt state carries the OLD block's annealed
        # LR (potentially near-zero from cosine end), which would kill all learning.
        # Per-group multipliers must survive the reset (group order == _group_lrs).
        for pg, _m in zip(opt.param_groups, _group_lrs):
            pg["lr"] = LR * _m
            pg.pop("initial_lr", None)
        print("  [opt] AdamW moments restored, LR reset to base (per-group mults kept)", flush=True)
    except Exception as _e:
        print(f"  [opt] state restore FAILED ({_e}) — rebuilding (1K-step warm-up)", flush=True)
print(f"  [opt] FT params={sum(p.numel() for p in _ft_params):,} (wd=0, lr=x1) | tail params={sum(p.numel() for p in _tail_w + _tail_b):,} (wd={_tail_wd}, lr=x{_TAIL_LR_MULT}) | amsgrad=True", flush=True)

# Wipe tripwire: fingerprint the loaded weights; any code path that rebuilds
# the model between resume and training (calibration reset was one such bug)
# aborts here instead of silently training from scratch for three hours.
_wipe_sig = None
if _ck is not None:
    with torch.no_grad():
        _wipe_sig = float(model.ft.weight.flatten()[::9973].abs().sum().item())

# ---- AUTO-CONVERGENCE SIZING ------------------------------------------------
# Calibrate throughput, then pick the largest data subset that can be trained
# to convergence (loss plateaus + cosine LR reaches zero) inside the budget.
_OVERHEAD_SEC = 300   # startup, model init, first featurize warm-up
_SPS_FALLBACK = 4.0   # conservative steps/sec if calibration unavailable (resume blocks skip calib)

_sps = _SPS_FALLBACK
if gstep == 0 and _ck is None:  # fresh runs only — warm starts skip calibration
    # (the calibration reset below REBUILDS the model, which would silently wipe
    #  warm-started weights; resumed/warm runs take the fallback rate instead)
    import time as _t
    _dummy_w = torch.randint(0, 24576, (BS, 32), device=device, dtype=torch.long)
    _dummy_b = torch.randint(0, 24576, (BS, 32), device=device, dtype=torch.long)
    _dummy_s = torch.ones(BS, device=device)
    _dummy_t = torch.zeros(BS, device=device)
    def _cal_step():
        opt.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _p = model(_dummy_w, _dummy_b, _dummy_s)
            _l = ((_p - _dummy_t).abs().mean())
        _l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), _gc)
        opt.step()
    for _ in range(10):
        _cal_step()   # warmup: kernel selection, allocator growth
    if device == "cuda": torch.cuda.synchronize()   # WITHOUT the syncs we measure the CPU
    _cal_t0 = _t.time()                             # LAUNCH rate (~60/s), not the GPU — the
    for _ in range(_CAL_STEPS):                     # "61.4 st/s" v10 calibration was exactly
        _cal_step()                                 # this illusion (true pipelined rate 6.6)
    if device == "cuda": torch.cuda.synchronize()
    _sps = _CAL_STEPS / (_t.time() - _cal_t0)
    print(f"  [calibrate] {_sps:.1f} steps/s (BS={BS})", flush=True)
    model = LNNUE(L1=L1).to(device)  # reset — calibration dirtied the weights
    # re-init optimizer (fresh model params) — same name-based groups as above
    _tw2 = [model.l2.weight, model.l3.weight, model.out.weight]
    _tb2 = [model.l2.bias, model.l3.bias, model.out.bias]
    opt = torch.optim.AdamW([
        {"params": [model.ft.weight, model.ft_bias], "weight_decay": 0.0},
        {"params": _tw2, "weight_decay": _tail_wd},
        {"params": _tb2, "weight_decay": 0.0},
    ], lr=LR, amsgrad=True)

# BUDGET<=0 means "no time limit": size the LR schedule for effectively-unlimited
# time — the max(600,...) fallback silently turned that into a 600s schedule
# (LR dead after 10 min while training ran on for hours)
if BUDGET <= 0:
    _avail_sec = 365 * 24 * 3600
else:
    _avail_sec = max(600, BUDGET - _OVERHEAD_SEC)
_total_steps = int(_avail_sec * _sps)
_total_visits = _total_steps * BS
_subset_pos = _total_visits   # single pass: every frame the budget could plausibly cover
# An explicitly pinned horizon (NNUE_T_MAX_STEPS) means the user pinned the PASS —
# the budget-based trim must not silently drop frames from it. The wall-clock
# BUDGET break still guards the run.
if _T_MAX_FIXED > 0:
    _subset_pos = int(8e9)  # effectively unlimited: est is ~1.9x real positions anyway
# select frames until subset is filled
_subset_frames = []; _subset_bytes = 0
for f in FRAMES:
    fb = os.path.getsize(f)
    if _subset_bytes + fb / 1.05 > _subset_pos:
        break
    _subset_frames.append(f); _subset_bytes += int(fb / 1.05)
# FRAME SKIP must be applied BEFORE subset selection, not after
_FRAMES_SKIP = int(os.environ.get("NNUE_FRAME_SKIP", "0"))  # skip first N frames (fresh-data mode)
if _FRAMES_SKIP > 0 and _FRAMES_SKIP < len(FRAMES):
    FRAMES = FRAMES[_FRAMES_SKIP:]
    # re-run subset selection on the remaining (fresh) frames
    _subset_frames = []; _subset_bytes = 0
    for f in FRAMES:
        fb = os.path.getsize(f)
        if _subset_bytes + fb / 1.05 > _subset_pos:
            break
        _subset_frames.append(f); _subset_bytes += int(fb / 1.05)
FRAMES = _subset_frames if _subset_frames else FRAMES[:1]  # at least 1 frame
est_pos = sum(os.path.getsize(f) for f in FRAMES) // 1.05
if _FENEVAL:
    # est_pos above is the .xz BYTE size — meaningless for positions. Estimate
    # from the measured test80 dump compression: 995MB xz = 50M lines ≈ 20
    # compressed bytes per position (3.5GB raw text, ~70B/line, xz ratio 0.28).
    # Drives the featurize-cache disk guard below; ±2x error only flips caching
    # on/off, never correctness.
    est_pos = int(sum(os.path.getsize(f) for f in FRAMES) / 20)
EPOCHS = max(1, int(_total_visits / max(1, est_pos)))
_max_ep = int(os.environ.get("NNUE_MAX_EPOCHS", "0"))  # optional hard cap (0 = budget-limited)
if _max_ep > 0: EPOCHS = min(EPOCHS, _max_ep)
if _FENEVAL and _max_ep <= 0:
    # fen-eval: EPOCHS above divides the visit budget by the .xz BYTE size —
    # meaningless for compressed TSV (it computed ~6 here). With the VRAM cache
    # each extra epoch is pure training (no re-featurize), so raise the loop cap
    # and let budget-hit / NNUE_T_MAX_STEPS govern, as the sizing line promises.
    EPOCHS = max(EPOCHS, 4000)
# Featurize-cache disk guard: caching every selected frame needs est_pos*REC bytes on
# /tmp (~1.5TB for a full 74-frame run). If that doesn't fit, tee hits ENOSPC
# mid-frame, the featurizer dies on SIGPIPE, and a full /tmp can also kill the
# checkpoint saves. Auto-disable instead of dying at frame N.
if _FEAT_CACHE:
    import shutil as _sh
    try:
        _free = _sh.disk_usage("/tmp").free
        _need = int(est_pos * REC)
        if _need > 0.9 * _free:
            _FEAT_CACHE = False
            print(f"  [cache] OFF: would need {_need/1e9:.0f}GB, /tmp has {_free/1e9:.0f}GB free "
                  f"(ENOSPC mid-frame would kill the run)", flush=True)
    except Exception:
        pass
# Warm-restart (pass 2+) knobs:
#   NNUE_T_MAX_STEPS — fix the cosine horizon explicitly (defined earlier with
#   the other env vars; the budget-based refinement under-anneals when data < budget).
T_MAX = _T_MAX_FIXED if _T_MAX_FIXED > 0 else min(_total_steps, EPOCHS * est_pos // BS)
_SEG0 = gstep   # resumed runs: the cosine segment is THIS session's steps (the
                # LambdaLR counter restarts at 0 each process), not global gstep.
# LR schedule: clamped cosine (LambdaLR, never rebounds past T_MAX).
# CosineAnnealingLR rebounds after T_MAX (PyTorch behavior); LambdaLR with
# min(1.0, step/T_MAX) clamps at zero permanently.
import math as _math
_T_MAX_LIVE = T_MAX   # mutable: refined per frame from the MEASURED pipeline rate.
                      # The upfront estimate assumes pure-GPU steps/s; a featurize-bound
                      # run achieves ~1/3 of that — without refinement the cosine never
                      # completes and the net gets saved at >50% base LR.
def _lr_lambda(step):
    if _T_MAX_LIVE <= 0: return 1.0
    p = min(1.0, step / float(_T_MAX_LIVE))
    return 0.5 * (1.0 + _math.cos(_math.pi * p))
sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
print(f"  [sched] clamped cosine T_max={T_MAX} steps, refined per frame (initial lr={LR})", flush=True)

# SWA: average weights from the last 25% of training (free quality boost).
# Two triggers: NNUE_SWA_START (epoch number) or NNUE_SWA_START_STEP (global step) —
# the step form exists because a warm-restart PASS is a single epoch over the
# frames, so "start at epoch N" cannot express "average the last quarter".
# The running average is checkpointed in the .pt ("swa"), so a pass split across
# multiple budget sessions averages over the PASS tail, not just the last session.
_SWA_START = int(os.environ.get("NNUE_SWA_START", "0"))  # 0=disabled; set to epoch number to enable
_SWA_STEP  = int(os.environ.get("NNUE_SWA_START_STEP", "0"))  # 0=disabled; global-step trigger
_swa_model = None; _swa_n = 0
if _SWA_START > 0 or _SWA_STEP > 0:
    _swa_model = copy.deepcopy(model)
    for p in _swa_model.parameters(): p.data.zero_()
    _swa_trig = f"epoch {_SWA_START}" if _SWA_START > 0 else f"step {_SWA_STEP}"
    print(f"  [swa] enabled from {_swa_trig}", flush=True)
    _swa_saved = _ck.get("swa") if _ck is not None else None
    if _swa_saved and os.environ.get("NNUE_SWA_RESUME", "0") == "1":
        _swa_model.load_state_dict(_swa_saved["model"]); _swa_n = int(_swa_saved.get("n", 0))
        print(f"  [swa] resumed running average ({_swa_n} updates)", flush=True)
    elif _swa_saved:
        print("  [swa] fresh average this pass (NNUE_SWA_RESUME=1 to carry over)", flush=True)

def _swa_active():
    if _swa_model is None: return False
    if _SWA_START > 0 and epoch >= _SWA_START: return True
    if _SWA_STEP > 0 and gstep - _swa_sess0 >= _SWA_STEP: return True
    return False

def _swa_update():
    global _swa_n
    if _swa_model is None: return
    _swa_n += 1
    with torch.no_grad():
        for swa_p, model_p in zip(_swa_model.parameters(), model.parameters()):
            swa_p.data += (model_p.data - swa_p.data) / _swa_n
print(f"  [sizing] subset={len(FRAMES)} frames (~{est_pos/1e9:.1f}B pos) | epochs={EPOCHS} (loop CAP only — "
      f"budget-hit + per-frame T_max refinement govern) | T_max={T_MAX} steps (initial) | "
      f"budget {_avail_sec}s @ {_sps:.1f} st/s", flush=True)

def _save_final(reason):
    """Unified exit save. .nnue comes from the BEST weights (SWA average if active);
    .pt keeps the RAW model + its own Adam moments — SWA-averaged weights paired
    with pre-average moments would desync the next resume block."""
    if _swa_model is not None and _swa_n > 0:
        print(f"  [swa] exporting averaged weights ({_swa_n} updates)", flush=True)
        save_nnue(_swa_model, os.path.join(out_dir, OUT))
    else:
        save_nnue(model, os.path.join(out_dir, OUT))
    torch.save({"model": model.state_dict(), "gstep": gstep, "opt": opt.state_dict(),
                "swa": ({"model": _swa_model.state_dict(), "n": _swa_n} if _swa_model is not None else None)},
               os.path.join(out_dir, OUT_BASE + ".pt"))
    upload_output()
    print(f"DONE ({reason}) - {OUT}: {gstep} steps, {total_pos:,} pos in {time.time()-t0:.0f}s", flush=True)
    sys.exit(0)

gstep = globals().get('gstep', 0); t0 = time.time(); total_pos = 0   # keeps resumed gstep
# VRAM-resident featurize cache, shared across epochs: keyed per frame file.
# Filled on epoch 0 when the whole frame's tensors fit in free VRAM; later
# epochs reuse it and skip the featurize subprocess entirely.
if _wipe_sig is not None:
    with torch.no_grad():
        _sig2 = float(model.ft.weight.flatten()[::9973].abs().sum().item())
    assert abs(_sig2 - _wipe_sig) < 1e-3 * max(1.0, _wipe_sig), (
        f"FATAL: model weights changed after resume ({_wipe_sig:.1f} -> {_sig2:.1f}) "
        "— a code path rebuilt the model; refusing to train from scratch silently")
    print(f"[wipe-guard] weights stable after resume (sig {_wipe_sig:.1f})", flush=True)

_VRAM_STORE = {}
# ---- validation holdout (NNUE_FENEVAL_VAL_FILE): per-epoch TRUE generalization.
# The split MUST be FEN-hash based (split_val.py) so visit-multiplicity repeats
# cannot land in both sides — the 105-epoch run's train MAE fell to 91.1 while
# real strength collapsed; a position-level random split would have hidden it.
_val_tensors = None
_val_path = os.environ.get("NNUE_FENEVAL_VAL_FILE")
if _val_path and _FENEVAL:
    _vd = "xz -dc" if _val_path.endswith(".xz") else "cat"
    _vproc = subprocess.Popen(["bash", "-c", f"{_vd} {_val_path} | {FEAT} --fen-eval --threads {NTH}"],
                              stdout=subprocess.PIPE, bufsize=0, stderr=None)
    _recs = bytearray()
    while True:
        _d = _vproc.stdout.read(1 << 20)
        if not _d:
            break
        _recs += _d
    try:
        _vproc.stdout.close(); _vproc.wait()
    except Exception:
        pass
    _va = np.frombuffer(bytes(_recs), dtype=np.uint8).reshape(-1, REC)
    del _recs
    _val_tensors = (
        torch.from_numpy(_va[:, :64].copy().view(np.int16).reshape(-1, 32)).long().to(device),
        torch.from_numpy(_va[:, 64:128].copy().view(np.int16).reshape(-1, 32)).long().to(device),
        torch.from_numpy(_va[:, 128:132].copy().view(np.float32).reshape(-1)).to(device),
        torch.from_numpy(_va[:, 132:136].copy().view(np.float32).reshape(-1)).to(device),
        # 5th column: quiet flag (0 quiet / 1 capture-best / 2 in-check) when the
        # dump carries it — gates the quiet-subset val MAE, the round-3 metric.
        torch.from_numpy(_va[:, 136:140].copy().view(np.float32).reshape(-1)).to(device),
    )
    del _va
    print(f"[VAL] held out {len(_val_tensors[3]):,} positions from {_val_path}", flush=True)

for epoch in range(EPOCHS):
    for fi, frame_path in enumerate(FRAMES):
        _f0_step, _f0_t = gstep, time.time()   # per-frame rate measurement (T_max refinement)
        if BUDGET and time.time() - t0 >= BUDGET:
            save_nnue(model, os.path.join(out_dir, OUT)); break
        _vram_key = os.path.basename(str(frame_path))
        _skip_featurize = False
        if epoch > 0 and _vram_key in _VRAM_STORE:
            _skip_featurize = True
        # FEATURIZE CACHE: first epoch featurizes and caches to /tmp; later
        # epochs read the cache (saves ~70s/frame — 25%+ of multi-epoch budget).
        # v2 key: the featurizer now emits 140-byte records (result channel) —
        # stale 136-byte caches from earlier runs would silently misparse.
        # Validation holdout at the STREAM level: when a val file is set, the train
        # featurizer drops every position whose FEN hash falls below the threshold
        # (identical crc32/threshold math in split_val.py / extract_val) so the
        # train and val sets are exact complements from ONE uploaded file.
        _val_frac = float(os.environ.get("NNUE_VAL_FRAC", "0.005"))
        _skip = f" --skip-hash-frac {_val_frac}" if (_val_tensors is not None and _val_frac > 0) else ""
        # decompress-once raw cache: xz decode is single-threaded (~0.5M pos/s)
        # but the MT featurizer does 3.6M/s on plain text; teeing epoch 0's
        # decompression to /tmp makes later epochs ~7x cheaper. Guarded by disk.
        _raw_cache = f"/tmp/feneval_raw_{os.path.basename(str(frame_path))}"
        try:
            import shutil as _sh2
            _need_raw = int(os.path.getsize(_FENEVAL) * (8 if _FENEVAL.endswith(".xz") else 3 if _FENEVAL.endswith(".zst") else 0))
            _raw_cache_ok = _need_raw > 0 and _sh2.disk_usage("/tmp").free > _need_raw * 1.1
        except Exception:
            _raw_cache_ok = False
        _fc = f"/tmp/featcache2_{os.path.basename(frame_path)}.raw"
        # Featurizer progress goes to stderr; pass it through so featurize
        # phases are visible instead of silently swallowed by the pipeline.
        _pass_err = None if os.environ.get("NNUE_QUIET") else None
        if _FENEVAL and _FEAT_CACHE and epoch == 0:
            dec = ("xz -T0 -dc" if _FENEVAL.endswith(".xz") else "zstd -dc") if _FENEVAL.endswith((".xz", ".zst")) else "cat"
            cmd = f"{dec} {_FENEVAL} | {FEAT} --fen-eval --threads {NTH}{_skip} | tee {_fc}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0, stderr=None)
        elif _FENEVAL and _FEAT_CACHE and os.path.exists(_fc) and os.path.getsize(_fc) > REC * 1000:
            # big fen-eval sets re-featurize per epoch otherwise (~15min for 300M pos)
            print(f"  [cache] reading {os.path.basename(_fc)} ({os.path.getsize(_fc)/1e6:.0f}MB)", flush=True)
            p = subprocess.Popen(["cat", _fc], stdout=subprocess.PIPE, bufsize=0)
        elif _FENEVAL and _FENEVAL.endswith((".xz", ".zst")) and os.path.exists(_raw_cache) and os.path.getsize(_raw_cache) > 1e8:
            # raw text cached by epoch 0's tee: cat + MT featurize runs ~3.6M pos/s
            # vs ~0.5M/s through single-threaded xz decode
            print(f"  [raw-cache] reading {os.path.basename(_raw_cache)} "
                  f"({os.path.getsize(_raw_cache)/1e9:.1f}GB)", flush=True)
            cmd = f"cat {_raw_cache} | {FEAT} --fen-eval --threads {NTH}{_skip}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0, stderr=None)
        elif _FENEVAL and _FENEVAL.endswith((".xz", ".zst")) and _raw_cache_ok and epoch == 0:
            # epoch 0: tee the decompressed text to /tmp while featurizing at the
            # xz-bound rate (the tee adds no wall time; later epochs skip xz entirely)
            dec = "xz -T0 -dc" if _FENEVAL.endswith(".xz") else "zstd -dc"
            cmd = f"{dec} {_FENEVAL} | tee {_raw_cache} | {FEAT} --fen-eval --threads {NTH}{_skip}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0, stderr=None)
        elif _FENEVAL:
            dec = ("xz -T0 -dc" if _FENEVAL.endswith(".xz") else "zstd -dc") if _FENEVAL.endswith((".xz", ".zst")) else "cat"
            cmd = f"{dec} {_FENEVAL} | {FEAT} --fen-eval --threads {NTH}{_skip}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0, stderr=None)
        elif _FEAT_CACHE and epoch == 0:
            dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
            cmd = f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {NTH} | tee {_fc}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0, stderr=None)
        elif _FEAT_CACHE and os.path.exists(_fc) and os.path.getsize(_fc) > REC * 1000:
            print(f"  [cache] reading {os.path.basename(_fc)} ({os.path.getsize(_fc)/1e6:.0f}MB)", flush=True)
            p = subprocess.Popen(["cat", _fc], stdout=subprocess.PIPE, bufsize=0)
        else:
            dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
            cmd = f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {NTH}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0, stderr=None)
        # VRAM-resident cache: on epoch 0, if the whole frame's tensors fit in
        # free VRAM, keep them on-device so later epochs skip re-featurizing.
        # est bytes/pos: idx (64×i16)×2 + (stm,tgt,res)×f32 = 256+12 ≈ 268B
        # plus ~1.5x working room for cat/perm/batch indices.
        _vram_store_here = (epoch == 0 and _skip_featurize is False)
        def _vram_frame_bytes(n): return int(n * 268 * 1.5)
        try:
            _free_vram, _ = torch.cuda.mem_get_info(device)
        except Exception:
            _free_vram = 0
        part = 0
        _health = None   # set by the parts loop; stays None if the frame yields nothing
        if _skip_featurize:
            # VRAM cache hit: restore epoch-0 tensors, skip the featurize subprocess
            w, b, s, t, r = _VRAM_STORE[_vram_key]
            N = w.shape[0]
            print(f"  [vram-cache] epoch {epoch}: reusing {N:,} pos (no re-featurize)", flush=True)
        # One iteration per <=CAP-position VRAM part. A cache-hit epoch enters with
        # w/b/s/t/r/N already restored above and must run the SAME training loop
        # below — the old `while not _skip_featurize:` header skipped the whole
        # body for cached epochs, making them 2-second no-ops that trained nothing.
        while True:
            if _skip_featurize:
                part += 1
                print(f"  [frame {fi+1} part {part}: {N:,} pos -> train (vram-cache)]", flush=True)
            else:
                ws, bs, ss, ts, rsl = [], [], [], [], []
                got = 0
                while got < CAP:
                    data = bytearray()
                    while len(data) < BUF * REC:
                        d = p.stdout.read(min(1 << 20, BUF * REC - len(data)))
                        if not d: break
                        data += d
                    m = len(data) // REC
                    if m == 0: break
                    a = np.frombuffer(bytes(data[:m * REC]), dtype=np.uint8).reshape(m, REC)
                    ws.append(torch.from_numpy(a[:, :64].copy().view(np.int16).reshape(m, 32)).to(device))
                    bs.append(torch.from_numpy(a[:, 64:128].copy().view(np.int16).reshape(m, 32)).to(device))
                    ss.append(torch.from_numpy(a[:, 128:132].copy().view(np.float32).reshape(m)).to(device))
                    ts.append(torch.from_numpy(a[:, 132:136].copy().view(np.float32).reshape(m)).to(device))
                    rsl.append(torch.from_numpy(a[:, 136:140].copy().view(np.float32).reshape(m)).to(device))
                    got += m
                if got == 0: break
                part += 1
                if part == 1: print(f"[featurizing+training frame {fi+1}/{len(FRAMES)} ...]", flush=True)
                # cat each, FREE its list immediately -> avoids the 2x peak that OOM'd before
                w = torch.cat(ws); del ws
                b = torch.cat(bs); del bs
                s = torch.cat(ss); del ss
                t = torch.cat(ts); del ts
                r = torch.cat(rsl); del rsl
                N = got
                print(f"  [frame {fi+1} part {part}: {N:,} pos -> train]", flush=True)
            perm = torch.randperm(N, device=device)
            _POWER = float(os.environ.get("NNUE_LOSS_POWER", "2.6"))  # 0 = old sigmoid-MSE
            _FEN_SKIP = float(os.environ.get("NNUE_FEN_SKIP", "0"))  # 0=disabled; hard-drop was losing decisive positions
            for i in range(0, N, BS):
                idx = perm[i:i + BS]
                wi = w[idx].long(); bi = b[idx].long(); si = s[idx]; ti = t[idx]; ri = r[idx]
                # Safety: sanitize non-finite targets FIRST (NaN survives clamp and one
                # backward pass would poison every weight in the net), then clamp
                # extremes (mate scores, parse glitches) instead of dropping them
                # (dropping loses decisive positions) or letting them saturate sigmoid
                if not torch.isfinite(ti).all():
                    ti = torch.nan_to_num(ti, nan=0.0, posinf=1500.0, neginf=-1500.0)
                ti = ti.clamp(-1500.0, 1500.0)
                if part == 1 and i == 0:
                    # One-time data gate per frame: an endian/padding/viewpoint bug in
                    # the C featurizer would otherwise train SILENTLY on garbage
                    print(f"  [DATA] w {wi.min().item()}..{wi.max().item()} | b {bi.min().item()}..{bi.max().item()} | "
                          f"stm {si.unique().tolist()[:4]} | tgt {ti.min().item():.0f}..{ti.max().item():.0f} "
                          f"std={ti.std().item():.0f} | result {ri.unique().tolist()[:4]}", flush=True)
                    assert 0 <= int(wi.min().item()) and int(wi.max().item()) <= NUM_INPUTS, "bad white feature idx"
                    assert 0 <= int(bi.min().item()) and int(bi.max().item()) <= NUM_INPUTS, "bad black feature idx"
                    assert bool(((si == 0.0) | (si == 1.0)).all()), "bad stm values (not 0/1)"
                    # result channel carries the quiet flag for fen-eval data:
                    # 0 quiet, 1 capture-best, 2 in-check (gamepacks keep -1/0/+1)
                    _fen_quiet = bool(((ri == 0.0) | (ri == 1.0) | (ri == 2.0)).all())
                    _game_res = bool(((ri == -1.0) | (ri == 0.0) | (ri == 1.0)).all())
                    assert _fen_quiet or _game_res, "bad result/quiet values"
                    # Label-convention gate (the 589M incident): with a warm-started
                    # model, predictions must correlate POSITIVELY with targets
                    # within EACH side-to-move half. A stm-vs-white-relative score
                    # mix flips one half negative — detectable before any training.
                    if _ck is not None:
                        with torch.no_grad():
                            _g = min(8192, wi.shape[0])
                            _gp = model(wi[:_g], bi[:_g], si[:_g]).float()
                        for _nm, _mask in (("WTM", si[:_g] > 0.5), ("BTM", si[:_g] < 0.5)):
                            if int(_mask.sum()) < 512:
                                continue
                            _c = float(torch.corrcoef(torch.stack(
                                [_gp[_mask], ti[:_g][_mask]]))[0, 1])
                            assert _c > 0.3, (
                                f"LABEL CONVENTION FAIL: corr(pred,target) on {_nm} = {_c:+.3f} "
                                f"(negative ⇒ the cp column's stm convention is flipped for this half)")
                        print("  [DATA] stm-split correlation gate PASS (labels consistent)", flush=True)
                # Smart FEN skip: drop positions where |target| is extreme (>800cp)
                # — these are usually won/lost positions where eval adds noise
                if _FEN_SKIP > 0:
                    _keep = ti.abs() < 800.0
                    if _keep.sum() < BS // 4: continue  # skip batch if too few survive
                    wi = wi[_keep]; bi = bi[_keep]; si = si[_keep]; ti = ti[_keep]; ri = ri[_keep]
                    if len(ti) < BS // 8: continue
                if _LAM0 < 1.0 or _LAM1 < 1.0:
                    _lam = _LAM0 + (_LAM1 - _LAM0) * min(1.0, gstep / _WDL_ANNEAL)
                    if _lam < 1.0:
                        ti = _lam * ti + (1.0 - _lam) * _RESULT_CP * torch.where(si == 1.0, ri, -ri)
                opt.zero_grad()
                # quiet-robust weights (NNUE_QUIET_LOSS="w_q,w_c,w_i", e.g. "1,0.15,0"):
                # per-sample weights from the flag channel; None when disabled or the
                # data carries game results (-1/0/+1) instead of quiet flags.
                _qw = None
                if _QUIET_MODE and (ri == 2.0).any():
                    _qw = torch.where(ri == 0.0, _QUIET_WQ,
             torch.where(ri == 1.0, _QUIET_WC, _QUIET_WI))
                if os.environ.get("NNUE_AUTOCAST", "1") != "0":
                    ctx_ac = torch.autocast(device_type=device, dtype=torch.bfloat16)
                else:
                    ctx_ac = torch.autocast(device_type=device, enabled=False)   # fp32 (no bf16 drift)
                with ctx_ac:
                    pred = model(wi, bi, si)
                if _POWER > 0:
                    # Power-2.6 loss in SIGMOID space (SF's proven formula):
                    # diff = |σ(pred/SCALE) - σ(target/SCALE)|, loss = diff^2.6
                    # NOT raw cp space — raw cp gradients are ~6500x larger and
                    # get annihilated by gradient clipping (the bug that caused
                    # loss to flatline at ~230K with zero effective learning)
                    _sdiff = (torch.sigmoid(pred / SCALE) - torch.sigmoid(ti / SCALE)).abs()
                    _pterms = _sdiff ** _POWER
                    # print on the FIRST weighted batch and every 500 steps after;
                    # a warm-resumed gstep (e.g. 163076) may not hit a multiple of
                    # 500 inside a short pilot at all
                    if _qw is not None and (_QUIET_FIRST_PRINT or (gstep % 500) == 0):
                        # Pilot mechanics check: the share of the WEIGHTED loss each
                        # position class actually carries (quiet/capture/check).
                        with torch.no_grad():
                            _QUIET_FIRST_PRINT = False
                            _mq = float((_pterms * _qw)[ri == 0.0].sum())
                            _mc = float((_pterms * _qw)[ri == 1.0].sum())
                            _mi = float((_pterms * _qw)[ri == 2.0].sum())
                            _ms = _mq + _mc + _mi
                            print(f"  [QUIET] step {gstep} n(q/c/i)="
                                  f"{int((ri==0.0).sum())}/{int((ri==1.0).sum())}/{int((ri==2.0).sum())} "
                                  f"weighted loss-mass q={_mq/_ms:.3f} c={_mc/_ms:.3f} i={_mi/_ms:.3f}", flush=True)
                    if _POS_W1 > 0:
                        # SF position weighting, weight-normalized exactly like
                        # SF: loss = (terms*w).sum()/w.sum() — not w*mean(), so
                        # the loss scale stays comparable to the unweighted run.
                        _pf = torch.sigmoid(ti / SCALE)
                        _pw = 1.0 + (2.0 ** _POS_W1 - 1.0) * ((_pf - 0.5) ** 2 * _pf * (1.0 - _pf)) ** _POS_W2
                        loss = (_pterms * _pw).sum() / _pw.sum()
                    elif _qw is not None:
                        # Round-3 quiet-robust loss: weight-normalized per-sample
                        # weights from the quiet flag channel (0 quiet / 1 capture /
                        # 2 in-check). The engine consumes eval at quiet nodes only;
                        # this stops capacity flowing into the tactical tail where
                        # labels are noise (the sf1-vs-full law).
                        loss = (_pterms * _qw).sum() / _qw.sum()
                    else:
                        loss = _pterms.mean()
                else:
                    loss = ((torch.sigmoid(pred / SCALE) - torch.sigmoid(ti / SCALE)) ** 2).mean()
                if _ANCHOR_W > 0:
                    # Anti-drift anchor (p5/p6 lesson): the power loss is nearly
                    # blind to constant stm-relative offsets; pin the batch mean
                    # prediction to the batch mean target in sigmoid space.
                    _a = (torch.sigmoid(pred / SCALE).mean()
                          - torch.sigmoid(ti / SCALE).mean()) ** 2
                    loss = loss + _ANCHOR_W * _a
                if not torch.isfinite(loss):
                    print(f"  WARN: non-finite loss at step {gstep} — skipping batch "
                          f"(if this repeats, weights are already poisoned — restart)", flush=True)
                    continue
                loss.backward()
                if _gc > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _gc)   # v6 fix
                opt.step()
                if _wc > 0:   # hard weight clamp: stops bf16-driven L2 drift (|W| can't exceed _wc)
                    with torch.no_grad():
                        for p in model.parameters():
                            p.clamp_(-_wc, _wc)
                sched.step(); gstep += 1
                if _swa_model is not None and _swa_active():
                    _swa_update()
                if gstep % 50 == 0:
                    # loss.item() ONLY here — a per-step .item() syncs the GPU queue
                    # every step and costs ~10-20% throughput
                    try: _l2 = f" L2|max|={float(model.l2.weight.detach().abs().max().item()):.2f}"
                    except Exception: _l2 = ""
                    dt = time.time() - t0
                    print(f"  e{epoch} f{fi+1}p{part} step {gstep} loss={loss.item():.5f}{_l2} | {total_pos/1e6:.0f}M+{i/1e6:.0f}M | {gstep/max(dt,1):.1f} steps/s", flush=True)
                if _T_MAX_FIXED > 0 and (gstep - _SEG0) >= _T_MAX_LIVE:
                    # fixed horizon (NNUE_T_MAX_STEPS): cosine has annealed to LR=0 —
                    # stop instead of idling at zero LR until the budget clock expires
                    print(f">>> T_max reached at step {gstep} - saving & stopping", flush=True)
                    _save_final("tmax")
                if BUDGET and time.time() - t0 >= BUDGET:
                    del w, b, s, t, perm; torch.cuda.empty_cache()
                    try: p.stdout.close(); p.terminate()
                    except Exception: pass
                    print(f">>> BUDGET hit at step {gstep} - saving & stopping", flush=True)
                    _save_final("budget")
            total_pos += N
            # v8 health line: computed BEFORE the tensors are freed below.
            # Catches failure-archive signatures loss alone missed (v7-run4
            # weight-growth saturation, eval-scale compression, silent stall).
            _health = None
            try:
                _m = min(200000, int(w.shape[0]))
                if _m > 0:
                    with torch.no_grad():
                        _pv = model(w[:_m].long(), b[:_m].long(), s[:_m]).float()
                        _tv = t[:_m].clamp(-1500.0, 1500.0).float()   # same view as the loss
                        _mae = (_pv - _tv).abs().mean().item()
                        _lmae = ""
                        if False:
                            _lvh = model.linear(w[:_m].long(), b[:_m].long(), s[:_m]).float()
                            _lmae = f" linMAE={(_lvh - _tv).abs().mean().item():.1f}cp"
                        _ps, _ts = _pv.std().item(), _tv.std().item()
                        _wn = " ".join(f"{_nm}={float(getattr(model, _nm).weight.detach().norm().item()):.0f}"
                                       for _nm in ("ft", "emb", "l1", "l2", "out") if hasattr(model, _nm))
                        _pad = ""
                        try: _pad = f" pad|max|={float(model.ft.weight[NUM_INPUTS].abs().max().item()):.1e}"
                        except Exception: pass   # gather mode has no pad row
                    _health = f"MAE={_mae:.1f}cp{_lmae} predSTD={_ps:.0f} tgtSTD={_ts:.0f}{_pad} | {_wn}"
            except Exception as _e:
                _health = f"unavailable ({_e})"
            # VRAM store decision: keep this frame's tensors for later epochs
            # only when the WHOLE frame fits with working room (single-part frames
            # whose estimated bytes < free VRAM at epoch 0). N < CAP proves this
            # part ended early (= the frame is COMPLETE): caching part 1 of a
            # multi-part frame would make later epochs silently train on a subset.
            if _vram_store_here and part == 1 and N < CAP and _vram_frame_bytes(N) < _free_vram:
                _VRAM_STORE[_vram_key] = (w, b, s, t, r)
                print(f"  [vram-cache] stored {N:,} pos ({_vram_frame_bytes(N)/1e9:.1f}GB < {_free_vram/1e9:.0f}GB free)", flush=True)
            else:
                # cache-hit epochs: drops LOCAL refs only — _VRAM_STORE keeps the tensors
                del w, b, s, t, perm; torch.cuda.empty_cache()
            if _skip_featurize:
                break   # cache hit = exactly one restored part; this epoch's frame is done
        if not _skip_featurize:
            try: p.stdout.close(); p.wait()
            except Exception: pass
        # cached epochs now reach the shared health block inside the loop, so
        # _health is real per-epoch MAE — no placeholder. If it's still None the
        # frame genuinely produced nothing and the FATAL below must fire.
        if _health:
            print(f"  [HEALTH f{fi+1}] {_health}", flush=True)
            if _val_tensors is not None:
                with torch.no_grad():
                    _m = min(200000, len(_val_tensors[3]))
                    _vpred = model(_val_tensors[0][:_m], _val_tensors[1][:_m], _val_tensors[2][:_m]).float()
                _vmae = (_vpred - _val_tensors[3][:_m].clamp(-1500.0, 1500.0)).abs().mean().item()
                try:
                    _gap = _vmae - _mae
                except NameError:
                    _gap = float("nan")
                print(f"  [VAL f{fi+1}] MAE={_vmae:.1f}cp  train-val gap={_gap:+.1f}cp "
                      f"(sustained gap above 1.0cp = memorizing, not learning)", flush=True)
                # Quiet-subset val MAE — the metric the engine actually consumes.
                # Blended MAE worsening is EXPECTED under quiet weighting; this is
                # the number that must improve.
                if len(_val_tensors) > 4:
                    try:
                        with torch.no_grad():
                            _vflg = _val_tensors[4][:_m]
                            _verr = (_vpred - _val_tensors[3][:_m].clamp(-1500.0, 1500.0)).abs()
                            _parts = []
                            for _cname, _cval in (("quiet", 0.0), ("capture", 1.0), ("check", 2.0)):
                                _cm = _vflg == _cval
                                if int(_cm.sum()) > 0:
                                    _parts.append(f"{_cname}={float(_verr[_cm].mean()):.1f}")
                            if _parts:
                                print(f"  [VAL f{fi+1}] by class (cp): {' '.join(_parts)}", flush=True)
                    except Exception as _e:
                        print(f"  [VAL] per-class skip: {_e}", flush=True)
        else:
            # A frame that yields ZERO positions means the featurizer crashed
            # (e.g. binary/glibc mismatch) — an empty epoch must never "succeed".
            raise SystemExit(f"FATAL: frame {fi+1} produced 0 positions — featurizer "
                             f"broken? Check GLIBC/exec-bit on {FEAT}.")
        print(f"  [frame {fi+1} done: cum {total_pos:,} ({total_pos/1e9:.2f}B), {time.time()-t0:.0f}s]", flush=True)
        # Refine the schedule denominator from THIS frame's marginal rate (excludes the
        # slow startup, which would skew the projection low and kill the LR early), so
        # the cosine completes exactly at budget-hit instead of saving at high LR.
        if BUDGET > 0 and not _T_MAX_FIXED:
            _mrate = (gstep - _f0_step) / max(time.time() - _f0_t, 1.0)
            _sess = gstep - _SEG0   # session-local steps (warm-restart segment)
            _proj = int(_sess + _mrate * max(0.0, BUDGET - (time.time() - t0) - 60))
            _proj = max(_sess + 50, min(_proj, _total_steps))
            if abs(_proj - _T_MAX_LIVE) > _T_MAX_LIVE // 20:
                print(f"  [sched] T_max -> {_proj:,} steps (measured {_mrate:.1f} st/s)", flush=True)
            _T_MAX_LIVE = _proj
        save_nnue(model, os.path.join(out_dir, OUT))   # incremental save after each frame
        torch.save({"model": model.state_dict(), "gstep": gstep, "opt": opt.state_dict(),
                    "swa": ({"model": _swa_model.state_dict(), "n": _swa_n} if _swa_model is not None else None)},
                   os.path.join(out_dir, OUT_BASE + ".pt"))
    else:
        continue
    break

# SWA never leaks into the .pt here: _save_final exports .nnue from the SWA average
# (if active) and keeps the raw model + matching moments for cross-block resume
print(f"Quantize locally: python quantize_i8.py {OUT} {OUT_BASE}_i8.nnue", flush=True)
_save_final("finished")
