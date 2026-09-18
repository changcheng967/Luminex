#!/usr/bin/env python3
"""C500 bandwidth ceiling + step decomposition: how far from theoretical max is the
6.6 st/s training step, and which alternative path closes the gap?

Run AFTER the training block (same gap as diag2_speed.py):
  python diag2_speed.py && python diag3_bw.py

Sections:
  B1  memcpy bandwidth        (pure copy: read+write)
  B2  read bandwidth          (sum reduction: read-only)
  B3  index_select gather     (the FT forward primitive, isolated)
  B4  index_add_ scatter      (the FT backward primitive, isolated)
  D1  EmbeddingBag fwd only   (both halves, BS=131072)
  D2  EmbeddingBag fwd+bwd
  D3  full step fwd+bwd+clip+opt   (reconfirm ~6.6)
  D4  manual FT path          (index_select + sum instead of EmbeddingBag)
  D5  bf16 embedding path     (weight cast bf16 for the call — halves traffic)
  D6  opt.step() alone
  CEILING: predicted max st/s from measured B3/B4 rates vs required 37GB/step
"""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from luminex_nnue_train import LNNUE, NUM_INPUTS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} VRAM={p.total_memory/1e9:.1f}GB", flush=True)

BS = 131072
L1 = 512

def tbench(name, fn, warmup=5, steps=20, traffic_gb=None):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps): fn()
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps
    extra = f"   ({traffic_gb/dt:7.0f} GB/s)" if traffic_gb else ""
    print(f"  {name:46s} {dt*1000:8.2f} ms{extra}", flush=True)
    return dt

print("\n[B] raw bandwidth primitives:", flush=True)
a = torch.empty(1 << 30, dtype=torch.float32, device=DEV).fill_(1.0)   # 4GB
b = torch.empty_like(a)
dt = tbench("B1  memcpy 4GB->4GB (8GB traffic)", lambda: b.copy_(a), traffic_gb=8.0)
bw_copy = 8.0 / dt
del b
dt = tbench("B2  sum reduction 4GB (4GB read)", lambda: a.sum(), traffic_gb=4.0)
bw_read = 4.0 / dt
del a
torch.cuda.empty_cache()

W = torch.randn(NUM_INPUTS + 1, L1, device=DEV)
G = torch.zeros_like(W)
idx = torch.randint(0, NUM_INPUTS + 1, (BS * 32,), device=DEV)
rows = W[idx]                                        # (4.2M, 512) fp32, 8.6GB
tbench("B3  index_select 4.2Mx512 fp32 (17GB tr)", lambda: W[idx], traffic_gb=17.2)
tbench("B4  index_add_ 4.2Mx512 fp32 (17GB tr)",  lambda: G.index_add_(0, idx, rows), traffic_gb=17.2)
del rows, G
torch.cuda.empty_cache()

print("\n[D] training-step decomposition:", flush=True)
torch.manual_seed(0)
m = LNNUE(L1=L1).to(DEV)
opt = torch.optim.AdamW([
    {"params": [m.ft.weight, m.ft_bias], "weight_decay": 0.0},
    {"params": [m.l2.weight, m.l3.weight, m.out.weight], "weight_decay": 1e-2},
    {"params": [m.l2.bias, m.l3.bias, m.out.bias], "weight_decay": 0.0},
], lr=1e-3, amsgrad=True)
torch.manual_seed(1)
w = torch.randint(0, NUM_INPUTS, (BS, 32), device=DEV, dtype=torch.long)
bi = torch.randint(0, NUM_INPUTS, (BS, 32), device=DEV, dtype=torch.long)
s = (torch.rand(BS, device=DEV) > 0.5).float()
t = torch.randn(BS, device=DEV) * 300

def fwd_nograd():
    with torch.no_grad(), torch.autocast(device_type=DEV, dtype=torch.bfloat16):
        m(w, bi, s)

def fwd_bwd():
    opt.zero_grad()
    with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
        pred = m(w, bi, s)
        loss = ((torch.sigmoid(pred/400) - torch.sigmoid(t/400)).abs() ** 2.6).mean()
    loss.backward()

def full_step():
    fwd_bwd()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    _ = loss_store[0]

loss_store = [None]
def full_step_item():
    opt.zero_grad()
    with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
        pred = m(w, bi, s)
        loss = ((torch.sigmoid(pred/400) - torch.sigmoid(t/400)).abs() ** 2.6).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    _ = loss.item()

def manual_step():
    opt.zero_grad()
    with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
        wt = m.ft.weight                                    # fp32 leaf
        aw = wt[w].sum(dim=1) + m.ft_bias                   # manual bag, w half
        ab = wt[bi].sum(dim=1) + m.ft_bias
        sm = s.view(-1, 1)
        h = torch.cat([sm*aw + (1-sm)*ab, (1-sm)*aw + sm*ab], dim=1)
        h = torch.clamp(h, 0, 1) ** 2
        h = torch.clamp(m.l2(h), 0, 1) ** 2
        h = torch.clamp(m.l3(h), 0, 1) ** 2
        pred = m.out(h).squeeze(-1) * 300.0
        loss = ((torch.sigmoid(pred/400) - torch.sigmoid(t/400)).abs() ** 2.6).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    _ = loss.item()

def bf16emb_step():
    opt.zero_grad()
    with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
        wq = m.ft.weight.to(torch.bfloat16)                 # halves gather+scatter
        aw = wq[w].sum(dim=1).float() + m.ft_bias
        ab = wq[bi].sum(dim=1).float() + m.ft_bias
        sm = s.view(-1, 1)
        h = torch.cat([sm*aw + (1-sm)*ab, (1-sm)*aw + sm*ab], dim=1)
        h = torch.clamp(h, 0, 1) ** 2
        h = torch.clamp(m.l2(h), 0, 1) ** 2
        h = torch.clamp(m.l3(h), 0, 1) ** 2
        pred = m.out(h).squeeze(-1) * 300.0
        loss = ((torch.sigmoid(pred/400) - torch.sigmoid(t/400)).abs() ** 2.6).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    _ = loss.item()

tbench("D1  fwd only, no_grad", fwd_nograd)
tbench("D2  fwd+bwd (EmbeddingBag)", fwd_bwd)
d3 = tbench("D3  full step (fwd+bwd+clip+opt)", full_step_item)
print(f"       -> D3 = {1.0/d3:.2f} st/s ({BS/d3/1e6:.2f} M pos/s)  [expect ~6.6 st/s]", flush=True)

# grad sanity for the alternative paths (they must produce weight grads)
def grad_ok():
    return m.ft.weight.grad is not None and float(m.ft.weight.grad.abs().sum()) > 0

before = float(m.ft.weight.grad.abs().sum()) if m.ft.weight.grad is not None else -1
d4 = tbench("D4  manual FT (index_select+sum) full step", manual_step)
assert grad_ok(), "manual path produced no FT grad!"
print(f"       (manual path grads OK)", flush=True)
d5 = tbench("D5  bf16-embedding full step", bf16emb_step)
assert grad_ok(), "bf16 path produced no FT grad!"
print(f"       (bf16 path grads OK — ACCURACY still needs a real training test)", flush=True)

# D6: opt alone (grads already populated)
tbench("D6  opt.step() alone", lambda: opt.step())

print("\n[CEILING]", flush=True)
req = 37.0    # GB/step minimum traffic: 17.2 fwd gather + 17.2 bwd scatter + ~2.5 rest
print(f"  memcpy BW {bw_copy:.0f} GB/s | read BW {bw_read:.0f} GB/s", flush=True)
print(f"  required traffic/step ~= {req:.0f} GB (fp32 embedding)", flush=True)
print(f"  BW-bound ceiling      ~= {min(bw_copy, bw_read)/req:5.1f} st/s  ({min(bw_copy,bw_read)/req*BS/1e6:.1f} M pos/s)", flush=True)
print(f"  current D3            = {1/d3:5.1f} st/s  -> running at {100*(1/d3)/(min(bw_copy,bw_read)/req):.0f}% of ceiling", flush=True)
if d5 < d4 * 0.8 and d5 < d3 * 0.8:
    print("  bf16 embedding is the big lever — halves the dominant traffic.", flush=True)
elif d4 < d3 * 0.8:
    print("  manual index path beats EmbeddingBag — swap the forward.", flush=True)
print("DONE - paste this whole output back.", flush=True)
