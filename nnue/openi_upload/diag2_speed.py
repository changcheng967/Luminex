#!/usr/bin/env python3
"""C500 training-throughput matrix: find the knob that beats ~6.6 st/s.

Run AFTER the training block finishes (running it during would steal GPU).
Each variant: 10 warmup + 40 timed steps, synced. ~5 min total.

  C0  L1=512 BS=131072   reference (the v10 config)
  C1  L1=512 BS=262144   per-step overhead amortized over 2x batch
  C2  L1=512 BS=65536    inverse probe (smaller batch)
  C3  L1=256 BS=131072   HALF the FT traffic — the architecture question
  C4  L1=256 BS=262144
  C5  fused single-bag   w|b concatenated -> ONE EmbeddingBag call (half the launches)
  C6  no clip_grad_norm_ (is the norm reduction eating anything?)
  C7  fp32 (autocast off)
  C8  torch.compile probe (skips gracefully if mcPyTorch can't)
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from luminex_nnue_train import LNNUE, NUM_INPUTS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} VRAM={p.total_memory/1e9:.1f}GB", flush=True)

def fresh(L1):
    torch.manual_seed(0)
    m = LNNUE(L1=L1).to(DEV)
    opt = torch.optim.AdamW([
        {"params": [m.ft.weight, m.ft_bias], "weight_decay": 0.0},
        {"params": [m.l2.weight, m.l3.weight, m.out.weight], "weight_decay": 1e-2},
        {"params": [m.l2.bias, m.l3.bias, m.out.bias], "weight_decay": 0.0},
    ], lr=1e-3, amsgrad=True)
    return m, opt

def bench(name, fn, steps=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    r = steps / dt
    print(f"  {name:42s} {r:7.2f} st/s   ({r*fn.bs/1e6:6.2f} M pos/s)", flush=True)
    return r

def make_bench(L1, bs, clip=True, autocast=True, fused=False, use_compile=False):
    m, opt = fresh(L1)
    torch.manual_seed(1)
    w = torch.randint(0, NUM_INPUTS, (bs, 32), device=DEV, dtype=torch.long)
    b = torch.randint(0, NUM_INPUTS, (bs, 32), device=DEV, dtype=torch.long)
    s = (torch.rand(bs, device=DEV) > 0.5).float()
    t = torch.randn(bs, device=DEV) * 300

    def step():
        opt.zero_grad()
        with torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=autocast):
            if fused:
                both = m.ft(torch.cat([w, b], dim=1))            # ONE bag call
                sm = s.view(-1, 1)
                aw = both[:, :L1] + m.ft_bias
                ab = both[:, L1:] + m.ft_bias
                h = torch.cat([sm * aw + (1 - sm) * ab,
                               (1 - sm) * aw + sm * ab], dim=1)
            else:
                pred = m(w, b, s)
                h = None
            if h is not None:
                h = torch.clamp(h, 0.0, 1.0) ** 2
                h = torch.clamp(m.l2(h), 0.0, 1.0) ** 2
                h = torch.clamp(m.l3(h), 0.0, 1.0) ** 2
                pred = m.out(h).squeeze(-1) * 300.0
            sd = (torch.sigmoid(pred / 400.0) - torch.sigmoid(t / 400.0)).abs()
            loss = (sd ** 2.6).mean()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        _ = loss.item()

    fn = torch.compile(step, dynamic=False) if use_compile else step
    fn.bs = bs
    return fn

print("\n[C] throughput matrix:", flush=True)
r = {}
try: r["C0"] = bench("C0  L1=512 BS=131072 (v10 reference)", make_bench(512, 131072))
except Exception as e: print(f"  C0 failed: {e}", flush=True)
try: r["C1"] = bench("C1  L1=512 BS=262144", make_bench(512, 262144))
except Exception as e: print(f"  C1 failed: {e}", flush=True)
try: r["C2"] = bench("C2  L1=512 BS=65536",  make_bench(512, 65536))
except Exception as e: print(f"  C2 failed: {e}", flush=True)
try: r["C3"] = bench("C3  L1=256 BS=131072", make_bench(256, 131072))
except Exception as e: print(f"  C3 failed: {e}", flush=True)
try: r["C4"] = bench("C4  L1=256 BS=262144", make_bench(256, 262144))
except Exception as e: print(f"  C4 failed: {e}", flush=True)
try: r["C5"] = bench("C5  L1=512 BS=131072 fused-bag", make_bench(512, 131072, fused=True))
except Exception as e: print(f"  C5 failed: {e}", flush=True)
try: r["C6"] = bench("C6  L1=512 BS=131072 no-clip", make_bench(512, 131072, clip=False))
except Exception as e: print(f"  C6 failed: {e}", flush=True)
try: r["C7"] = bench("C7  L1=512 BS=131072 fp32", make_bench(512, 131072, autocast=False))
except Exception as e: print(f"  C7 failed: {e}", flush=True)
try:
    print("  C8  torch.compile probe (first compile can take minutes)...", flush=True)
    r["C8"] = bench("C8  L1=512 BS=131072 compiled", make_bench(512, 131072, use_compile=True), warmup=3)
except Exception as e:
    print(f"  C8 compile unsupported/failed: {type(e).__name__}: {e}", flush=True)

print("\n[VERDICT guide]", flush=True)
if "C0" in r and "C3" in r:
    print(f"  L1=256 speedup: {r['C3']/r['C0']:.2f}x  (>=1.8x -> serious block-2 A/B candidate)", flush=True)
if "C0" in r and "C1" in r:
    print(f"  BS=262144 pos/s vs C0: {r['C1']*2/r['C0']:.2f}x  (>1.15x -> free win, just env var)", flush=True)
if "C0" in r and "C5" in r:
    print(f"  fused-bag: {r['C5']/r['C0']:.2f}x", flush=True)
print("DONE - paste this whole output back.", flush=True)
