#!/usr/bin/env python3
"""C500 training-throughput diagnosis: why is real training ~6 st/s when calibration
measured 61.4 st/s on the same model/BS?

Bench plan (each ~40 steps, ~1 min total):
  ENV   - GPU name, VRAM total/free
  A0    - calibration replica (dummy tensors, no big frame resident)  -> expect ~60
  A1    - SAME bench with a 152M-position frame replica resident in VRAM (~21GB)
          -> if this collapses to ~6, it's VRAM pressure/spill and the fix is just
             NNUE_VRAM_POS=100000000 (smaller parts, no code change)
  A2    - frame replica half-freed (~10GB resident) -> locate the knee
  B1    - op axis: pad indices (24576) present in embedding input
  B2    - op axis: int16 source + w[idx].long() gather (real data path)

Run:  cd /tmp/code && python diag_speed.py
"""
import os, sys, time, glob
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from luminex_nnue_train import LNNUE, NUM_INPUTS

BS   = 131072
LR   = 1e-3
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
NTH  = 1

print(f"torch {torch.__version__} device={DEV}", flush=True)
if DEV == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  VRAM total={p.total_memory/1e9:.1f}GB", flush=True)
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM free at start: {free/1e9:.1f}GB / {total/1e9:.1f}GB", flush=True)
    except Exception as e:
        print(f"(mem_get_info unavailable: {e})", flush=True)

def fresh_model():
    torch.manual_seed(0)
    m = LNNUE(L1=512).to(DEV)
    opt = torch.optim.AdamW([
        {"params": [m.ft.weight, m.ft_bias], "weight_decay": 0.0},
        {"params": [m.l2.weight, m.l3.weight, m.out.weight], "weight_decay": 1e-2},
        {"params": [m.l2.bias, m.l3.bias, m.out.bias], "weight_decay": 0.0},
    ], lr=LR, amsgrad=True)
    return m, opt

def bench(name, fn, steps=40, warmup=8):
    for _ in range(warmup):
        fn()
    if DEV == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        fn()
    if DEV == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  {name:55s} {steps/dt:7.1f} st/s", flush=True)
    return steps / dt

model, opt = fresh_model()

def make_step(w, b, s, t, use_sigmoid_power=True, do_item=False):
    def step():
        opt.zero_grad()
        with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
            pred = model(w, b, s)
            if use_sigmoid_power:
                sd = (torch.sigmoid(pred / 400.0) - torch.sigmoid(t / 400.0)).abs()
                loss = (sd ** 2.6).mean()
            else:
                loss = (pred - t).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if do_item:
            _ = loss.item()
    return step

# ---- A0: calibration replica -------------------------------------------------
torch.manual_seed(1)
w0 = torch.randint(0, NUM_INPUTS, (BS, 32), device=DEV, dtype=torch.long)
b0 = torch.randint(0, NUM_INPUTS, (BS, 32), device=DEV, dtype=torch.long)
s0 = (torch.rand(BS, device=DEV) > 0.5).float()
t0_ = torch.randn(BS, device=DEV) * 300
print("\n[A0] calibration replica (no frame resident):", flush=True)
bench("A0  dummy L1 loss (calibration exact)", make_step(w0, b0, s0, t0_, use_sigmoid_power=False))
bench("A0b dummy sigmoid-p2.6 (real loss)", make_step(w0, b0, s0, t0_))

# ---- B: op-axis variants (still no frame) -------------------------------------
torch.manual_seed(2)
w1 = w0.clone(); b1 = b0.clone()
w1[:, 30:] = NUM_INPUTS; b1[:, 30:] = NUM_INPUTS        # ~2 pads/row like real endgames
print("\n[B1] pad indices present (padding_idx backward path):", flush=True)
bench("B1  2 pads/row + sigmoid-p2.6", make_step(w1, b1, s0, t0_))

src = torch.randint(0, NUM_INPUTS, (4_000_000, 32), device=DEV, dtype=torch.int16)
src[:, 30:] = NUM_INPUTS
def gather_step():
    idx = torch.randint(0, 4_000_000, (BS,), device=DEV)
    w = src[idx].long()
    opt.zero_grad()
    with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
        pred = model(w, w, s0)
        sd = (torch.sigmoid(pred / 400.0) - torch.sigmoid(t0_ / 400.0)).abs()
        loss = (sd ** 2.6).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    _ = loss.item()
print("\n[B2] int16 source + gather + .long() (real data path):", flush=True)
bench("B2  int16 gather + sigmoid-p2.6 + .item()", gather_step)

# ---- A1: with the 152M frame replica resident ---------------------------------
N_FRAME = 151_988_717
print(f"\n[A1] allocating frame replica: {N_FRAME:,} pos (~20.7GB int16+fp32)...", flush=True)
frame = None
try:
    fw = torch.randint(0, NUM_INPUTS, (N_FRAME, 32), device=DEV, dtype=torch.int16)
    fb = torch.randint(0, NUM_INPUTS, (N_FRAME, 32), device=DEV, dtype=torch.int16)
    fs = (torch.rand(N_FRAME, device=DEV) > 0.5).half()
    ft = (torch.randn(N_FRAME, device=DEV) * 300).half()
    frame = (fw, fb, fs, ft)
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"  frame resident. VRAM free now: {free/1e9:.1f}GB / {total/1e9:.1f}GB", flush=True)
    except Exception:
        pass
    bench("A1  dummy bench WITH 20.7GB frame resident", make_step(w0, b0, s0, t0_))
    # real-path step against the frame (perm-slice gather exactly like the trainer)
    def real_step():
        i = torch.randint(0, N_FRAME, (BS,), device=DEV)
        w = fw[i].long(); b = fb[i].long(); sv = fs[i].float(); tv = ft[i].float().clamp(-1500, 1500)
        opt.zero_grad()
        with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
            pred = model(w, b, sv)
            sd = (torch.sigmoid(pred / 400.0) - torch.sigmoid(tv / 400.0)).abs()
            loss = (sd ** 2.6).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        _ = loss.item()
    bench("A1b REAL path (frame gather) replica", real_step)
    # ---- A2: free half the frame, find the knee --------------------------------
    fw2 = fw[: N_FRAME // 2].clone(); fb2 = fb[: N_FRAME // 2].clone()
    del frame, fw, fb, fs, ft
    torch.cuda.empty_cache()
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"\n[A2] half frame freed (~10GB resident). VRAM free: {free/1e9:.1f}GB", flush=True)
    except Exception:
        print("\n[A2] half frame freed (~10GB resident)", flush=True)
    bench("A2  dummy bench with ~10GB resident", make_step(w0, b0, s0, t0_))
    del fw2, fb2
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError as e:
    print(f"  FRAME REPLICA OOM'd: {e}", flush=True)
    print("  -> VRAM is smaller than 21GB+headroom: the trainer's 20.7GB frame does NOT", flush=True)
    print("     fit cleanly. Restart training with NNUE_VRAM_POS=80000000.", flush=True)
except RuntimeError as e:
    print(f"  frame replica failed: {e}", flush=True)

print("\nDONE - paste this whole output back.", flush=True)
