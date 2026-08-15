#!/usr/bin/env python3
"""C500 v7 trainer — ONE EPOCH over ALL ~15B positions with CROSS-FRAME full shuffle.

WHY THIS EXISTS (the diagnosis that ended v6/Phase0):
  c500_train_stream.py (per-frame sequential: featurize frame -> train -> free -> next)
  caused CATASTROPHIC FORGETTING. Every net it produced lost to v2 at depth 10 (0 to -32
  with unlimited time). The ONLY trainer that ever produced a good net (+218 Elo vs HCE at
  depth 10) was luminex_nnue_train.py (v2), because it did `perm = torch.randperm(N)` over
  ALL positions every epoch — every batch saw globally-mixed data.

  But v2 held all of N in VRAM. At 15B positions x 136 B = 2 TB that is impossible. This
  trainer recovers v2's full-shuffle property AT 15B SCALE by streaming:

    ParallelChunkSource runs N=8 featurizer pipes CONCURRENTLY (one per frame), interleaving
    their chunks into ONE queue. We accumulate ~300M of those interleaved positions into
    VRAM, then FULL-SHUFFLE (torch.randperm) and train every batch. Each buffer therefore
    mixes positions from ~8 different frames; every batch sees diverse data -> NO forgetting.
    Clear the buffer, continue. When all 115 frames are drained = exactly ONE epoch over
    ~15B UNIQUE positions. That is the user's spec: "1 epoch, 15B positions, full shuffle."

  OPTIMIZER = v2's EXACT recipe (the thing that produced +218 Elo): AdamW, UNIFORM wd=1e-4
  on ALL params, plain (no amsgrad). The earlier "tail-only wd=1e-2 + amsgrad" was a
  MISDIAGNOSIS that COLLAPSED the tail — measured: v7 L2 max|w|=0.0865 vs v2's 2.16 (25x
  smaller) -> eval lost discrimination -> -560 Elo vs v2 (18W-480L-2D, 500g bullet). v6's L2
  divergence was caused by bf16+no-grad-clip, NOT insufficient wd; v2's wd=1e-4 was always
  correct. The ONLY addition over v2 is grad-clip=1.0 (bf16 gradient-spike guard).

RUN (no args), env knobs:
  NNUE_L1=512  NNUE_EPOCHS=1  NNUE_BS=16384  NNUE_LR=1e-3
  NNUE_PARALLEL_WORKERS=8  NNUE_FEAT_THREADS=64  NNUE_VRAM_POS=300000000
  NNUE_GRAD_CLIP=1.0  NNUE_WD=1e-4  NNUE_BUDGET_SEC=12600
"""
import os, sys, subprocess, time, glob, random
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # anti-fragmentation
import numpy as np, torch
torch.set_num_threads(1)
from c2net.context import prepare, upload_output
ctx = prepare(); out_dir = ctx.output_path

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

# ---- discover frames (frame_*.zst/.xz); extract gamepack.tar if needed ----
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
if any(f.endswith(".zst") for f in FRAMES) and not os.path.exists("/usr/bin/zstd"):
    subprocess.run("apt-get install -y zstd >/dev/null 2>&1 || pip install -q zstandard", shell=True)

train_mod = _find_file("luminex_nnue_train.py")
chunk_mod = _find_file("chunk_source.py")
CODE_DIR = os.environ.get("CODE_DIR") or (os.path.dirname(train_mod) if train_mod else "/tmp/code")
sys.path.insert(0, CODE_DIR)
FEAT = os.environ.get("FEAT") or _find_file("luminex-featurize") or os.path.join(CODE_DIR, "luminex-featurize")
assert os.path.exists(FEAT), f"featurizer missing: {FEAT}"
assert chunk_mod, "chunk_source.py missing (needed for cross-frame interleaving)"

from luminex_nnue_train import LNNUE, save_nnue
from chunk_source import ParallelChunkSource

# ---- hyperparameters ----
L1        = int(os.environ.get("NNUE_L1", "512"))
EPOCHS    = int(os.environ.get("NNUE_EPOCHS", "1"))          # 1 epoch over 15B (user spec)
BS        = int(os.environ.get("NNUE_BS", "16384"))          # v2's batch: 919K updates over 15B (> v2's 255K). BS=131072 = only 114K updates (undertrains).
LR        = float(os.environ.get("NNUE_LR", "1e-3"))
NTH       = int(os.environ.get("NNUE_FEAT_THREADS", "14"))    # total featurizer threads, split across workers
NPAR      = int(os.environ.get("NNUE_PARALLEL_WORKERS", "8")) # concurrent frames -> cross-frame diversity
BUFFER_POS= int(os.environ.get("NNUE_VRAM_POS", "300000000")) # ~41 GB per interleaved buffer (fits 64 GB VRAM)
BUF       = int(os.environ.get("NNUE_BUF", "2000000"))        # chunk size (positions) per pipe read
BUDGET    = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.5*3600))))
_gc       = float(os.environ.get("NNUE_GRAD_CLIP", "1.0"))
_autocast = os.environ.get("NNUE_AUTOCAST", "1") != "0"        # bf16 ON: stress test proved bf16 == fp32 growth (NOT the v7 cause). Guard below catches real growth.
REC = 136; SCALE = 400.0
device = os.environ.get("NNUE_DEVICE", "").strip()
if not device:
    try:
        import torch_npu  # Ascend NPU backend (910/910ProA)
        if torch.npu.is_available(): device = "npu"
        elif torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
    except ImportError:
        device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("NNUE_OUT_NAME", "luminex_v7.nnue"); OUT_BASE = OUT[:-5] if OUT.endswith(".nnue") else OUT

# ---- shuffle frame order so different frames land in the same early buffers each run ----
random.seed(1337); random.shuffle(FRAMES)
# ---- SUBSET cap: v2's stability came from CONVERGENCE (multi-epoch repeats). 1 epoch over
# all 15B never converges -> SCReLU cascade runs away (L3->12-16, tactically-blind nets).
# Cap the frame count so each epoch is small enough to converge; loop with NNUE_EPOCHS>1.
_maxf = int(os.environ.get("NNUE_MAX_FRAMES", "0"))
if _maxf > 0 and len(FRAMES) > _maxf:
    FRAMES = FRAMES[:_maxf]
total_bytes = sum(os.path.getsize(f) for f in FRAMES)
est_pos = int(total_bytes / 1.05)   # ~1.05 B/pos -> ~14.5B from 15.3GB
nth_per_worker = max(1, NTH // NPAR)
print(f"v7 cross-frame shuffle: {len(FRAMES)} frames ({total_bytes/1e9:.2f}GB, ~{est_pos/1e9:.1f}B pos), "
      f"{EPOCHS} epoch(s), {NPAR} concurrent featurizers x {nth_per_worker} threads, buffer={BUFFER_POS/1e6:.0f}M, "
      f"L1={L1} bs={BS} bf16={_autocast} grad-clip={_gc} device={device}", flush=True)

# ============================================================================
# Model + optimizer (L2 root-cause fix: tail wd=1e-2, FT wd=0, AMSGrad)
# ============================================================================
# FT mode: embbag (fused EmbeddingBag, ~28x FASTER on CUDA) on cuda/cpu; gather on npu
# (EmbeddingBag may be unsupported on Ascend). Both == engine numerically (+74.7 startpos both).
_ft_mode = os.environ.get("NNUE_FT_MODE", "gather" if device == "npu" else "embbag")
model = LNNUE(L1=L1, ft_mode=_ft_mode).to(device)
# FT mode check. CORRECT modes: 'gather' (ft_w[idx].sum, == engine) and 'embbag2d'
# (nn.EmbeddingBag 2D mode='sum', verified numerically == gather, ~20x faster). The BUGGY
# mode is 'embbag1d' (the old C500 version, whose accumulator != engine -> -600 Elo garbage).
_ft_mode = getattr(model, "ft_mode", "gather")
if _ft_mode == "embbag1d" or hasattr(model, "probe_ft"):
    print(f"  !!! BUGGY embbag1d/probe_ft trainer detected (ft_mode='{_ft_mode}'). Net will be "
          f"GARBAGE. Replace luminex_nnue_train.py with the current version (embbag2d/gather).", flush=True)
print(f"  [LNNUE] L1={L1} ft_mode='{_ft_mode}' (gather/embbag2d == engine inference).", flush=True)

_ft_params   = [p for p in model.parameters() if p.numel() > 100000]    # FT embedding (~12.6M)
_tail_params = [p for p in model.parameters() if p.numel() <= 100000]   # L2/L3/out (<10K each)
# v2 PROVEN recipe: UNIFORM wd=1e-4 on ALL params. (The tail-only wd=1e-2 was the v7 failure:
# it collapsed L2 25x [max 0.0865 vs v2's 2.16] -> eval lost discrimination -> -560 Elo vs v2.
# v6's L2 divergence was bf16+no-grad-clip, NOT insufficient wd; v2's wd=1e-4 was always correct.)
_WD = float(os.environ.get("NNUE_WD", "1e-4"))            # uniform wd on ALL params (v2 recipe)
_AMSGRAD = os.environ.get("NNUE_AMSGRAD", "0") != "0"     # off by default (v2 used plain AdamW)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=_WD, amsgrad=_AMSGRAD)
print(f"  [opt] AdamW UNIFORM wd={_WD} on ALL params (v2 recipe) | amsgrad={_AMSGRAD} | grad-clip={_gc}", flush=True)
print(f"        FT params={sum(p.numel() for p in _ft_params):,} | tail params={sum(p.numel() for p in _tail_params):,}", flush=True)

total_steps = max(1, EPOCHS * est_pos // BS)
# TIME-BASED cosine annealing over the BUDGET (not the full-epoch step count). The run is
# featurizer-bound and budget-saves at ~460K/919K steps (50% of a step-based cosine -> LR still
# 5e-4 -> net saved mid-training, never settles -> calm/imprecise positional eval -> -600 Elo).
# Annealing over BUDGET_SEC guarantees LR->0 by the save, so the net fully settles regardless
# of throughput. v2 (the +218 Elo net) was fully annealed; this restores that property.
import math
_t0_train = time.time()
def _set_lr():
    frac = min((time.time() - _t0_train) / max(BUDGET, 1), 1.0)
    lr_now = LR * (1.0 + math.cos(math.pi * frac)) / 2.0
    for g in opt.param_groups: g["lr"] = lr_now
    return lr_now
_set_lr()   # initialize lr at peak
sched = None
print(f"  TIME-BASED cosine over {BUDGET}s budget -> LR {LR:.1e} -> ~0 by budget-save (full anneal)", flush=True)
print(f"  (full-epoch step count was {total_steps}; budget-save lands ~{int(BUDGET*0.6e6/BS):,} steps at ~0.6M/s)", flush=True)
print(f"  CROSS-FRAME full-shuffle: each {BUFFER_POS/1e6:.0f}M buffer mixes ~{NPAR} frames, randperm per buffer", flush=True)

# ============================================================================
# Cross-frame interleaved featurizer (the heart of the fix)
# ============================================================================
def cmd_for(frame_path, wid):
    dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
    return f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {nth_per_worker}"

# src is created per-epoch inside the loop below (EPOCHS>1 must re-stream the frames)

# ============================================================================
# Training loop: accumulate interleaved buffer -> full shuffle -> train -> clear
# ============================================================================
gstep = 0; total_pos = 0; buffers_done = 0; t0 = time.time()
_UPLOAD_EVERY = int(os.environ.get("NNUE_UPLOAD_EVERY", "6"))   # upload checkpoint every N buffers

def _save_and_upload(tag, force_upload=False):
    save_nnue(model, os.path.join(out_dir, OUT))
    msg = f"  [saved {OUT} at {tag}]"
    if force_upload or (buffers_done > 0 and buffers_done % _UPLOAD_EVERY == 0):
        try:
            upload_output(); msg += " (+uploaded)"
        except Exception as e:
            msg += f" (upload FAILED: {e})"
    print(msg, flush=True)

def _budget_exit(gstep, total_pos):
    print(f">>> BUDGET hit at step {gstep}, {total_pos/1e9:.2f}B pos seen — saving & uploading", flush=True)
    src.kill(); _save_and_upload(f"budget@{gstep}", force_upload=True)
    print(f"DONE (budget) - {OUT}: {gstep} steps, {total_pos:,} pos ({total_pos/1e9:.2f}B)", flush=True)
    sys.exit(0)

def _train_buffer(w_all, b_all, s_all, t_all, bid):
    """Full-shuffle one interleaved buffer and train every batch (v2's loop, verbatim)."""
    global gstep
    N = w_all.shape[0]
    perm = torch.randperm(N, device=device)        # <<< FULL SHUFFLE within the interleaved buffer
    running, window = 0.0, 0
    for i in range(0, N, BS):
        idx = perm[i:i + BS]
        wi = w_all[idx].long(); bi = b_all[idx].long(); si = s_all[idx]; ti = t_all[idx]
        opt.zero_grad()
        ctx_ac = torch.autocast(device_type=device, dtype=torch.bfloat16) if _autocast \
                 else torch.autocast(device_type=device, enabled=False)
        with ctx_ac:
            pred = model(wi, bi, si)
            loss = ((torch.sigmoid(pred / SCALE) - torch.sigmoid(ti / SCALE)) ** 2).mean()
        loss.backward()
        if _gc > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), _gc)
        opt.step(); lr_now = _set_lr(); gstep += 1
        # NO weight clamp — the clamp pegged every layer at the bound (degenerate, saturated):
        # eval correlated on quiet pos (r=0.993) but was TACTICALLY BLIND (walked into mate,
        # 492/500). Growth is now controlled by stronger wd (NNUE_WD=5e-4) so weights settle
        # naturally (v2-like) with tactical vision. Guard below aborts only on true runaway.
        running += loss.item(); window += 1
        if gstep % 500 == 0:
            try:
                _ftm = float(model.ft.weight.detach().abs().max().item())
                _l2m = float(model.l2.weight.detach().abs().max().item())
                _l3m = float(model.l3.weight.detach().abs().max().item())
                _wm = f" FT|max|={_ftm:.2f} L2|max|={_l2m:.2f} L3|max|={_l3m:.2f}"
            except Exception:
                _wm = ""; _ftm = _l2m = _l3m = 0.0
            dt = time.time() - t0
            print(f"  buf{bid} step {gstep}/{total_steps} loss={running/window:.5f}{_wm} | "
                  f"{total_pos/1e9:.2f}B pos ({total_pos/max(dt,1)/1e6:.1f}M/s) "
                  f"lr={lr_now:.2e} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            running = 0.0; window = 0
            # Weight-growth guard: v2 stable at L2=2.16/L3=4.8; >2.3x saturates SCReLU
            # -> eval loses discrimination (measured r=0.52 vs v2) -> 494/500 garbage.
            # Abort early (save+exit) instead of wasting a full 6h run on a broken net.
            if _l2m > 5.0 or _l3m > 12.0:
                print(f"  *** WEIGHT-GROWTH ABORT @ step {gstep}: L2|max|={_l2m:.2f} L3|max|={_l3m:.2f} "
                      f"exceed guard (5.0/12.0). Saving current net + aborting.", flush=True)
                del w_all, b_all, s_all, t_all, perm; torch.cuda.empty_cache()
                _budget_exit(gstep, total_pos)
        if BUDGET and time.time() - t0 >= BUDGET:
            del w_all, b_all, s_all, t_all, perm; torch.cuda.empty_cache()
            _budget_exit(gstep, total_pos)
    del perm

# main epoch loop
for epoch in range(EPOCHS):
    if epoch > 0:
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS}: re-streaming the {len(FRAMES)}-frame subset (convergence repeats) ===", flush=True)
    src = ParallelChunkSource(FRAMES, cmd_for, BUF, NPAR, rec=REC, qsize=max(4, NPAR), raw=False)
    while True:
        # ---- accumulate BUFFER_POS interleaved positions into VRAM ----
        ws, bs, ss, ts = [], [], [], []
        got = 0; stream_end = False
        while got < BUFFER_POS:
            item = src.q.get()
            if item is None:                 # all frames drained -> end of epoch
                stream_end = True; break
            if isinstance(item, str):        # "frame:K" boundary marker -> skip
                continue
            w, b, s, t, m = item
            ws.append(torch.from_numpy(w).to(device, non_blocking=True))
            bs.append(torch.from_numpy(b).to(device, non_blocking=True))
            ss.append(torch.from_numpy(s).to(device, non_blocking=True))
            ts.append(torch.from_numpy(t).to(device, non_blocking=True))
            got += m; total_pos += m
        if got == 0:
            break                             # nothing left (only the None sentinel arrived)
        buffers_done += 1
        w_all = torch.cat(ws).contiguous(); del ws
        b_all = torch.cat(bs).contiguous(); del bs
        s_all = torch.cat(ss).contiguous(); del ss
        t_all = torch.cat(ts).contiguous(); del ts
        print(f"  [buffer {buffers_done}: {got/1e6:.0f}M interleaved pos from {NPAR} frames -> train]", flush=True)
        _train_buffer(w_all, b_all, s_all, t_all, buffers_done)
        del w_all, b_all, s_all, t_all; torch.cuda.empty_cache()
        _save_and_upload(f"buffer{buffers_done}@{gstep}")   # incremental checkpoint
        if stream_end:
            break
    print(f"  [epoch {epoch} done: {buffers_done} buffers, {total_pos/1e9:.2f}B pos, {time.time()-t0:.0f}s]", flush=True)

src.kill()
_save_and_upload(f"final@{gstep}", force_upload=True)
print(f"DONE - {OUT}: {gstep} steps, {total_pos:,} pos ({total_pos/1e9:.2f}B) in {time.time()-t0:.0f}s. uploading...", flush=True)
upload_output()
print(f"Quantize locally: python quantize_i8.py {OUT} {OUT_BASE}_i8.nnue", flush=True)
