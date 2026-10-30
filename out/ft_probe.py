#!/usr/bin/env python3
"""FT weight-tail statistics across dose checkpoints (+106 mechanism probe).

Pre-registered read: if the FT tail (max |w|, p99.9, count > 1.0) grows >=2x
from ~12k steps to ~20k steps within a lineage, the dose cliff is FT-tail
overfitting (rare rows memorising position noise), and NNUE_ANCHOR_W (already
implemented, currently 0) is the right regulariser. Flat stats -> the cliff
is something else and the +106 stays unexplained.
"""
import os, sys, struct
import torch

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

BASE = "/work/home/scnmyrr3ge/luminex-dcu"
CANDIDATES = [
    f"{BASE}/luminex_gen768p4.pt",      # session warm start
    f"{BASE}/gen768sf.pt",              # v6.2.0-era candidates
    f"{BASE}/gen768sf_full.pt",
    f"{BASE}/luminex_sfq_full.pt",      # v6.3.0-lineage full model?
    f"{BASE}/session1/r0b.pt",          # quiet 6k, seed 99
    f"{BASE}/session1/r0u.pt",          # unweighted 6k
    f"{BASE}/session1/r1.pt",           # capture 0.1, 6k
]


def ft_stats(w):
    w = w.detach().float().abs().flatten()
    n = w.numel()
    small = w[torch.randperm(n)[:500_000]] if n > 500_000 else w
    q99, q999 = torch.quantile(small, torch.tensor([0.99, 0.999]))
    return dict(max=round(w.max().item(), 4),
                p99=round(q99.item(), 4),
                p999=round(q999.item(), 4),
                mean=round(w.mean().item(), 5),
                n_gt_05=int((w > 0.5).sum()),
                n_gt_10=int((w > 1.0).sum()),
                n=n)


def layer_stats(sd, pat):
    ks = [k for k in sd if pat in k.lower()]
    if not ks:
        return None
    w = sd[ks[0]].detach().float().abs().flatten()
    return dict(max=round(w.max().item(), 4),
                mean=round(w.mean().item(), 6),
                std=round(w.std().item(), 6),
                n=w.numel())



for p in CANDIDATES:
    if not os.path.exists(p):
        print(f"{os.path.basename(p):28s} MISSING")
        continue
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{os.path.basename(p):28s} LOAD_FAIL {e}")
        continue
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    ftk = [k for k in sd if "ft" in k.lower() and "weight" in k.lower()]
    if not ftk:
        print(f"{os.path.basename(p):28s} NO_FT_KEY keys={list(sd)[:4]}")
        continue
    s = ft_stats(sd[ftk[0]])
    gs = ck.get("gstep", "?") if isinstance(ck, dict) else "?"
    print(f"{os.path.basename(p):28s} gstep={gs:>7} key={ftk[0]:<14} "
          f"max={s['max']:<8} p99={s['p99']:<8} p99.9={s['p999']:<8} "
          f"mean={s['mean']:<9} n>0.5={s['n_gt_05']:<9} n>1.0={s['n_gt_10']:<8} n={s['n']}")
    for pat in ("l2.weight", "l3.weight", "out", "l2.bias", "out.bias"):
        ls = layer_stats(sd, pat)
        if ls:
            print(f"    {pat:10s} max={ls['max']:<9} mean={ls['mean']:<10} std={ls['std']:<10} n={ls['n']}")
