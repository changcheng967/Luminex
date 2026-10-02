#!/usr/bin/env python3
"""V13NNUE audition on the DCU bench: probe_ft guard + 8,000-step train.

--mode scratch : from-scratch init, lr 8e-4   (reference: conf A = 235.3/243.6)
--mode warm    : SVD warm-start from a gen768 .pt, lr 1e-4
"""
import sys, time, math, random, argparse
import torch

sys.path.insert(0, "/workspace/luminex-lab/repo/nnue/openi_upload")
sys.path.insert(0, "/workspace/luminex-lab")
from arch_search_v3 import Data
from v13_model import V13NNUE, warm_start_from_gen768

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["scratch", "warm"], default="scratch")
ap.add_argument("--ckpt", default="/workspace/luminex-lab/luminex_gen768p4.pt")
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--seed", type=int, default=71)
args = ap.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed)
DEV = "cuda:0"
SCALE, POWER = 400.0, 2.6
LR = 8e-4 if args.mode == "scratch" else 1e-4

model = V13NNUE().to(DEV)
d = model.probe_ft(DEV)
print(f"[{args.mode}] probe_ft max drift = {d:.2e}", flush=True)
if args.mode == "warm":
    warm_start_from_gen768(model, args.ckpt)
    model = model.to(DEV)
    d2 = model.probe_ft(DEV)
    print(f"[{args.mode}] probe_ft after warm-start = {d2:.2e}", flush=True)

DATA = Data("/workspace/luminex-lab/real_data.npz", DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, amsgrad=True)
warm = 100
best = float("inf"); t0 = time.time()
for step in range(args.steps):
    frac = step / warm if step < warm else 0.5 * (
        1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))
    for g in opt.param_groups:
        g["lr"] = LR * frac
    w, b, s, t, side, men = DATA.train_batch(4096)
    pred = model(w, b, s)
    loss = ((torch.sigmoid(pred / SCALE) - torch.sigmoid(t / SCALE)).abs() ** POWER).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step + 1) % 500 == 0 or step == args.steps - 1:
        model.eval(); tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(DATA.ev[3]), 65536):
                w, b, s, t, _, _ = [x[i:i + 65536] for x in DATA.ev]
                p = model(w, b, s)
                tot += (p - t).abs().sum().item(); cnt += len(p)
        model.train()
        mae = tot / cnt
        best = min(best, mae)
        print(f"  [{args.mode}] step {step+1:5d} loss={loss.item():.5f} "
              f"evalMAE={mae:.1f} best={best:.1f}", flush=True)
print(f"V13 AUDITION [{args.mode}]: BEST MAE={best:.1f}cp ({time.time()-t0:.0f}s)",
      flush=True)
