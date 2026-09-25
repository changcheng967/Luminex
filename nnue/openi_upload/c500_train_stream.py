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
  NNUE_FRAME_SHUFFLE=1 NNUE_SHUFFLE_SEED=43 NNUE_DUAL_HEAD=1 \
  NNUE_LIN_WARMUP=2000 NNUE_LIN_LAMBDA=0.15 NNUE_LIN_LR_MULT=5 NNUE_TAIL_LR_MULT=3 \
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
assert os.path.exists(FEAT), f"featurizer missing: {FEAT}"
from luminex_nnue_train import LNNUE, save_nnue, NUM_INPUTS

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
# DOSL dual-head (Step-0): aux linear head trained alongside the full net so it is
# a standalone-usable qsearch stand-pat. Warmup at lam=1.0 (head alone must be a
# competent eval), then joint lam=0.25 persists as the anchor loss throughout.
# DEFAULT ON since pass-4: the Gen768 pass-3 export shipped a zero-trained LINH
# section (valid format, all-zero weights) that the engine happily used as a 0cp
# qsearch stand-pat — the 0-50 match loss. Export now REFUSES an all-zero head.
_DUAL = os.environ.get("NNUE_DUAL_HEAD", "1") == "1"
_LIN_WARM = int(os.environ.get("NNUE_LIN_WARMUP", "2000"))
_LIN_LAM  = float(os.environ.get("NNUE_LIN_LAMBDA", "0.25"))
if _DUAL:
    print(f"[DOSL] dual-head training ON (warmup {_LIN_WARM} steps at lam=1.0, then lam={_LIN_LAM})", flush=True)
else:
    print("[DOSL] dual-head training OFF — export will SKIP the LINH section (head untrained)", flush=True)
# Pinned cosine horizon (must be defined before the subset-trim below)
_T_MAX_FIXED = int(os.environ["NNUE_T_MAX_STEPS"]) if os.environ.get("NNUE_T_MAX_STEPS") else 0
_CAL_STEPS = 60   # calibration steps to measure throughput
_FEAT_CACHE = os.environ.get("NNUE_FEAT_CACHE", "0") == "1"  # OFF by default: single-pass
# never re-reads a frame, and full-data caching would need ~2.2TB of /tmp. Opt-in
# (still disk-guarded below) only makes sense for small-subset multi-epoch runs.
REC  = 136; SCALE = 400.0
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("NNUE_OUT_NAME", "luminex_v6.nnue"); OUT_BASE = OUT[:-5] if OUT.endswith(".nnue") else OUT
total_bytes = sum(os.path.getsize(f) for f in FRAMES)
print(f"per-frame train (OOM-safe): {len(FRAMES)} frames, {total_bytes/1e9:.2f}GB, L1={L1} bs={BS} cap={CAP:,} feat_threads={NTH} grad-clip={_gc} device={device}", flush=True)
print(f"loss: power={os.environ.get('NNUE_LOSS_POWER', '2.6')} in sigmoid(cp/{SCALE:.0f}) space, pos-weight w1={_POS_W1} w2={_POS_W2}{' (OFF)' if _POS_W1 == 0 else ''}, lambda=1.0 (pure eval — gamepack has no game results; SF end-lambda practice)", flush=True)

model = LNNUE(L1=L1).to(device)
gstep = 0  # must exist before resume check (fresh runs would NameError otherwise)
_ck = None  # resume checkpoint dict (kept for SWA state restore further below)
_resume = os.environ.get("NNUE_RESUME") or os.path.join(os.path.dirname(__file__) or ".", OUT_BASE + ".pt")
_opt_state = None  # deferred: optimizer doesn't exist yet at resume time
if os.path.exists(_resume) and os.environ.get("NNUE_RESUME", "1") != "0":
    try:
        _ck = torch.load(_resume, map_location=device, weights_only=False)
        _ck_L1 = _ck["model"]["ft_bias"].shape[0]
        if _ck_L1 != L1:
            # Net2Net widening (Gen v1.3 Step 0): load old-width ckpt into wider model
            from luminex_nnue_train import LNNUE
            LNNUE.net2net_widen(model, _ck["model"], _ck_L1)
            gstep = _ck.get("gstep", 0)
            _opt_state = None  # old moments are shape-incompatible — fresh optimizer
            print(f"[NET2NET] widened {_ck_L1} -> {L1} from {_resume} at gstep={gstep} (optimizer fresh)", flush=True)
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
if _DUAL:
    _lin_trained = float(model.lin.weight.detach().abs().max().item()) > 1e-9
    _lin_base = (gstep - _LIN_WARM) if _lin_trained else gstep
    print(f"  [DOSL] lin head {'pre-trained (warmup skipped)' if _lin_trained else 'fresh (warmup from session start)'} at gstep={gstep}", flush=True)
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
_LIN_LR_MULT  = float(os.environ.get("NNUE_LIN_LR_MULT", "5.0"))
opt = torch.optim.AdamW([
    {"params": _ft_params, "weight_decay": 0.0},        # NO decay on FT (rare-bucket protection)
    {"params": _tail_w,    "weight_decay": _tail_wd,    # tail weight decay (fixes L2 SCReLU feedback)
                         "lr": LR * _TAIL_LR_MULT},
    {"params": _tail_b,    "weight_decay": 0.0},        # biases are operating points — never decay
    {"params": [model.lin.weight], "weight_decay": 0.0, # DOSL head — WITHOUT this group the dual
                         "lr": LR * _LIN_LR_MULT},      # head gets grads but NEVER updates (pass-3 bug class)
], lr=LR, amsgrad=True)
_group_lrs = [1.0, _TAIL_LR_MULT, 1.0, _LIN_LR_MULT]
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
print(f"  [opt] FT params={sum(p.numel() for p in _ft_params):,} (wd=0, lr=x1) | tail params={sum(p.numel() for p in _tail_w + _tail_b):,} (wd={_tail_wd}, lr=x{_TAIL_LR_MULT}) | lin lr=x{_LIN_LR_MULT} | amsgrad=True", flush=True)

# ---- AUTO-CONVERGENCE SIZING ------------------------------------------------
# Calibrate throughput, then pick the largest data subset that can be trained
# to convergence (loss plateaus + cosine LR reaches zero) inside the budget.
_OVERHEAD_SEC = 300   # startup, model init, first featurize warm-up
_SPS_FALLBACK = 4.0   # conservative steps/sec if calibration unavailable (resume blocks skip calib)

_sps = _SPS_FALLBACK
if gstep == 0:  # fresh run: calibrate with a tiny forward+backward
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
    opt = torch.optim.AdamW([
        {"params": [model.ft.weight, model.ft_bias], "weight_decay": 0.0},
        {"params": [model.l2.weight, model.l3.weight, model.out.weight], "weight_decay": _tail_wd},
        {"params": [model.l2.bias, model.l3.bias, model.out.bias], "weight_decay": 0.0},
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
EPOCHS = max(1, int(_total_visits / max(1, est_pos)))
_max_ep = int(os.environ.get("NNUE_MAX_EPOCHS", "0"))  # optional hard cap (0 = budget-limited)
if _max_ep > 0: EPOCHS = min(EPOCHS, _max_ep)
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
    if _swa_saved:
        _swa_model.load_state_dict(_swa_saved["model"]); _swa_n = int(_swa_saved.get("n", 0))
        print(f"  [swa] resumed running average ({_swa_n} updates)", flush=True)

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
for epoch in range(EPOCHS):
    for fi, frame_path in enumerate(FRAMES):
        _f0_step, _f0_t = gstep, time.time()   # per-frame rate measurement (T_max refinement)
        if BUDGET and time.time() - t0 >= BUDGET:
            save_nnue(model, os.path.join(out_dir, OUT)); break
        # FEATURIZE CACHE: first epoch featurizes and caches to /tmp; later
        # epochs read the cache (saves ~70s/frame — 25%+ of multi-epoch budget).
        _fc = f"/tmp/featcache_{os.path.basename(frame_path)}.raw"
        if _FEAT_CACHE and epoch == 0:
            dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
            cmd = f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {NTH} | tee {_fc}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0)
        elif _FEAT_CACHE and os.path.exists(_fc) and os.path.getsize(_fc) > REC * 1000:
            print(f"  [cache] reading {os.path.basename(_fc)} ({os.path.getsize(_fc)/1e6:.0f}MB)", flush=True)
            p = subprocess.Popen(["cat", _fc], stdout=subprocess.PIPE, bufsize=0)
        else:
            dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
            cmd = f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {NTH}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0)
        part = 0
        _health = None   # set by the parts loop; stays None if the frame yields nothing
        while True:   # process frame in <=CAP-position VRAM loads (big frames -> multiple parts)
            ws, bs, ss, ts = [], [], [], []
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
                got += m
            if got == 0: break
            part += 1
            if part == 1: print(f"[featurizing+training frame {fi+1}/{len(FRAMES)} ...]", flush=True)
            # cat each, FREE its list immediately -> avoids the 2x peak that OOM'd before
            w = torch.cat(ws); del ws
            b = torch.cat(bs); del bs
            s = torch.cat(ss); del ss
            t = torch.cat(ts); del ts
            N = got
            print(f"  [frame {fi+1} part {part}: {N:,} pos -> train]", flush=True)
            perm = torch.randperm(N, device=device)
            _POWER = float(os.environ.get("NNUE_LOSS_POWER", "2.6"))  # 0 = old sigmoid-MSE
            _FEN_SKIP = float(os.environ.get("NNUE_FEN_SKIP", "0"))  # 0=disabled; hard-drop was losing decisive positions
            for i in range(0, N, BS):
                idx = perm[i:i + BS]
                wi = w[idx].long(); bi = b[idx].long(); si = s[idx]; ti = t[idx]
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
                          f"std={ti.std().item():.0f}", flush=True)
                    assert 0 <= int(wi.min().item()) and int(wi.max().item()) <= NUM_INPUTS, "bad white feature idx"
                    assert 0 <= int(bi.min().item()) and int(bi.max().item()) <= NUM_INPUTS, "bad black feature idx"
                    assert bool(((si == 0.0) | (si == 1.0)).all()), "bad stm values (not 0/1)"
                # Smart FEN skip: drop positions where |target| is extreme (>800cp)
                # — these are usually won/lost positions where eval adds noise
                if _FEN_SKIP > 0:
                    _keep = ti.abs() < 800.0
                    if _keep.sum() < BS // 4: continue  # skip batch if too few survive
                    wi = wi[_keep]; bi = bi[_keep]; si = si[_keep]; ti = ti[_keep]
                    if len(ti) < BS // 8: continue
                opt.zero_grad()
                if os.environ.get("NNUE_AUTOCAST", "1") != "0":
                    ctx_ac = torch.autocast(device_type=device, dtype=torch.bfloat16)
                else:
                    ctx_ac = torch.autocast(device_type=device, enabled=False)   # fp32 (no bf16 drift)
                with ctx_ac:
                    pred = model(wi, bi, si)
                    _lin_t = None
                    if _DUAL and _POWER > 0:
                        _lv = model.linear(wi, bi, si)
                        _lin_t = ((torch.sigmoid(_lv / SCALE) - torch.sigmoid(ti / SCALE)).abs()) ** _POWER
                    if _POWER > 0:
                        # Power-2.6 loss in SIGMOID space (SF's proven formula):
                        # diff = |σ(pred/SCALE) - σ(target/SCALE)|, loss = diff^2.6
                        # NOT raw cp space — raw cp gradients are ~6500x larger and
                        # get annihilated by gradient clipping (the bug that caused
                        # loss to flatline at ~230K with zero effective learning)
                        _sdiff = (torch.sigmoid(pred / SCALE) - torch.sigmoid(ti / SCALE)).abs()
                        _pterms = _sdiff ** _POWER
                        if _POS_W1 > 0:
                            # SF position weighting, weight-normalized exactly like
                            # SF: loss = (terms*w).sum()/w.sum() — not w*mean(), so
                            # the loss scale stays comparable to the unweighted run.
                            _pf = torch.sigmoid(ti / SCALE)
                            _pw = 1.0 + (2.0 ** _POS_W1 - 1.0) * ((_pf - 0.5) ** 2 * _pf * (1.0 - _pf)) ** _POS_W2
                            loss = (_pterms * _pw).sum() / _pw.sum()
                        else:
                            loss = _pterms.mean()
                    else:
                        loss = ((torch.sigmoid(pred / SCALE) - torch.sigmoid(ti / SCALE)) ** 2).mean()
                if _lin_t is not None:
                    _lam = 1.0 if gstep - _lin_base < _LIN_WARM else _LIN_LAM
                    loss = _lam * _lin_t.mean() + (1.0 - _lam) * loss
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
                        if _DUAL:
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
            del w, b, s, t, perm; torch.cuda.empty_cache()
        try: p.stdout.close(); p.wait()
        except Exception: pass
        if _health:
            print(f"  [HEALTH f{fi+1}] {_health}", flush=True)
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
