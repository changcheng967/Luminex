#!/usr/bin/env python3
"""NNUE Architecture Search v2 — Bayesian component optimizer for DCU.

Forks the production LNNUE (same init, activations, loss) and searches
L1 width / FM interaction rank / cross layer / tail dims with Optuna TPE.
Data: real Leela evals packed by sample_stream.py from gamepack frames.
"""
import sys, os, time, argparse, random, math
import numpy as np
import torch
import torch.nn as nn
import optuna

sys.path.insert(0, "/workspace/luminex-lab/repo/nnue/openi_upload")
from luminex_nnue_train import NUM_INPUTS, MAX_PIECES

DEVICE = "cuda:0"
SCALE = 400.0
POWER = 2.6

# ----------------------------------------------------------------------------
# Model: production LNNUE + searchable dims + optional FM interaction block
# ----------------------------------------------------------------------------
class SearchableNNUE(nn.Module):
    def __init__(self, L1=768, fm_rank=16, use_cross=False, tail_l2=16, tail_l3=32):
        super().__init__()
        self.L1, self.fm_rank, self.use_cross = L1, fm_rank, use_cross
        self.ft = nn.EmbeddingBag(NUM_INPUTS + 1, L1, mode="sum", padding_idx=NUM_INPUTS)
        self.ft_bias = nn.Parameter(torch.zeros(L1))
        nn.init.normal_(self.ft.weight, std=0.2)      # fan-in 32 -> acc std ~1 (production)
        self.ft.weight.data[NUM_INPUTS].zero_()
        if fm_rank > 0:
            self.fm_v = nn.Embedding(NUM_INPUTS + 1, fm_rank, padding_idx=NUM_INPUTS)
            nn.init.normal_(self.fm_v.weight, std=0.5)
            self.fm_v.weight.data[NUM_INPUTS].zero_()
        if use_cross and fm_rank > 0:
            self.cross_w = nn.Linear(fm_rank, fm_rank)
        in_dim = 2 * L1 + (2 * fm_rank if fm_rank > 0 else 0)
        self.l2 = nn.Linear(in_dim, tail_l2)
        self.l3 = nn.Linear(tail_l2, tail_l3)
        self.out = nn.Linear(tail_l3, 1)

    def _interact(self, idx):
        v = self.fm_v(idx)                       # (B, 32, r)
        s = v.sum(dim=1)                         # (B, r)
        i_raw = 0.5 * (s ** 2 - (v ** 2).sum(dim=1))
        i_cap = torch.clamp(i_raw, -2.0, 2.0) / 2.0
        if self.use_cross:
            i_cap = i_cap + i_cap * torch.sigmoid(self.cross_w(i_cap))
        return i_cap

    def forward(self, w_idx, b_idx, stm):
        acc_w = self.ft(w_idx) + self.ft_bias
        acc_b = self.ft(b_idx) + self.ft_bias
        m = stm.view(-1, 1).float()
        stm_acc = m * acc_w + (1 - m) * acc_b
        nstm_acc = (1 - m) * acc_w + m * acc_b
        h = torch.cat([stm_acc, nstm_acc], dim=1)
        if self.fm_rank > 0:
            h = torch.cat([h, self._interact(w_idx), self._interact(b_idx)], dim=1)
        h = torch.clamp(h, 0.0, 1.0) ** 2
        h = torch.clamp(self.l2(h), 0.0, 1.0) ** 2
        h = torch.clamp(self.l3(h), 0.0, 1.0) ** 2
        return self.out(h).squeeze(-1) * 300.0   # production output scale


class LinearPST(nn.Module):
    """Piece-square linear floor: same features collapsed to 768 (drop king buckets)."""
    def __init__(self):
        super().__init__()
        self.ft = nn.EmbeddingBag(NUM_INPUTS + 1, 1, mode="sum", padding_idx=NUM_INPUTS)
        nn.init.normal_(self.ft.weight, std=0.2)
        self.ft.weight.data[NUM_INPUTS].zero_()

    def forward(self, w_idx, b_idx, stm):
        w = self.ft(w_idx).squeeze(-1) + self.ft(b_idx).squeeze(-1) * 0  # white-pov sum
        b = self.ft(b_idx).squeeze(-1)
        m = stm.float()
        return (m * w + (1 - m) * (-b)) * 300.0


# ----------------------------------------------------------------------------
# Data: npz packed by sample_stream.py (w/b int32 padded, s/t float32)
# ----------------------------------------------------------------------------
class Data:
    def __init__(self, path, device):
        z = np.load(path)
        n = len(z["t"])
        split = int(n * 0.88)
        ev_start = split + 20000                    # gap: no train/eval adjacency
        def mov(x, dt): return torch.from_numpy(x).to(device=device, dtype=dt)
        self.tr_w = mov(z["w"][:split], torch.long)
        self.tr_b = mov(z["b"][:split], torch.long)
        self.tr_s = mov(z["s"][:split], torch.float32)
        self.tr_t = mov(z["t"][:split], torch.float32)
        self.ev_w = mov(z["w"][ev_start:], torch.long)
        self.ev_b = mov(z["b"][ev_start:], torch.long)
        self.ev_s = mov(z["s"][ev_start:], torch.float32)
        self.ev_t = mov(z["t"][ev_start:], torch.float32)
        print(f"data: {split:,} train / {len(self.ev_t):,} eval | "
              f"|t| mean {self.tr_t.abs().mean():.0f}cp", flush=True)

    def train_batch(self, bs):
        i = random.randint(0, len(self.tr_t) - bs - 1)
        return (self.tr_w[i:i+bs], self.tr_b[i:i+bs],
                self.tr_s[i:i+bs], self.tr_t[i:i+bs])

    def eval_mae(self, model, bs=65536):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(self.ev_t), bs):
                p = model(self.ev_w[i:i+bs], self.ev_b[i:i+bs], self.ev_s[i:i+bs])
                tot += (p - self.ev_t[i:i+bs]).abs().sum().item()
                cnt += len(p)
        model.train()
        return tot / cnt


# ----------------------------------------------------------------------------
# Trial runner: production loss (sigmoid^2.6), warmup+cosine, best-along-run score
# ----------------------------------------------------------------------------
def run_trial(model, data, steps, lr, bs, tag):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    warm = 100
    best = float("inf")
    t0 = time.time()
    for step in range(steps):
        frac = step / warm if step < warm else 0.5 * (
            1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * frac
        w, b, s, t = data.train_batch(bs)
        pred = model(w, b, s)
        loss = ((torch.sigmoid(pred / SCALE) - torch.sigmoid(t / SCALE)).abs() ** POWER).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0 or step == steps - 1:
            mae = data.eval_mae(model)
            best = min(best, mae)
            print(f"  [{tag}] step {step+1:5d} loss={loss.item():.5f} "
                  f"evalMAE={mae:.1f} best={best:.1f}", flush=True)
    return best, time.time() - t0


def objective(trial):
    L1 = trial.suggest_categorical("L1", [256, 384, 512, 768])
    fm_rank = trial.suggest_categorical("fm_rank", [0, 8, 16, 32])
    use_cross = (trial.suggest_categorical("use_cross", [False, True])
                 if fm_rank > 0 else False)
    tail_l2 = trial.suggest_categorical("tail_l2", [8, 16, 32])
    tail_l3 = trial.suggest_categorical("tail_l3", [16, 32, 64])
    model = SearchableNNUE(L1=L1, fm_rank=fm_rank, use_cross=use_cross,
                           tail_l2=tail_l2, tail_l3=tail_l3)
    params = sum(p.numel() for p in model.parameters())
    print(f"trial {trial.number}: L1={L1} fm={fm_rank} cross={use_cross} "
          f"tail=({tail_l2},{tail_l3}) params={params:,}", flush=True)
    best, el = run_trial(model, DATA, STEPS, LR, BATCH, f"t{trial.number}")
    print(f"trial {trial.number}: BEST MAE={best:.1f}cp ({el:.0f}s, {params:,} params)",
          flush=True)
    trial.set_user_attr("params", params)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="real_data.npz")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=None,
                    help="train one architecture: L1,fm_rank,cross,tail_l2,tail_l3")
    args = ap.parse_args()

    STEPS, BATCH, LR = args.steps, args.batch, args.lr
    torch.manual_seed(args.seed); random.seed(args.seed)
    DATA = Data(args.data, DEVICE)

    if args.config:
        L1, fm, cr, tl2, tl3 = [int(v) for v in args.config.split(",")]
        model = SearchableNNUE(L1=L1, fm_rank=fm, use_cross=bool(cr),
                               tail_l2=tl2, tail_l3=tl3)
        best, el = run_trial(model, DATA, STEPS, LR, BATCH, "cfg")
        print(f"CONFIG RESULT: {args.config} BEST MAE={best:.1f}cp ({el:.0f}s)")
        sys.exit(0)

    if args.smoke:
        print("=== smoke: linear PST floor ===", flush=True)
        m1, _ = run_trial(LinearPST(), DATA, STEPS, LR, BATCH, "lin")
        print("=== smoke: default NNUE L1=768 no FM ===", flush=True)
        m2, _ = run_trial(SearchableNNUE(L1=768, fm_rank=0), DATA, STEPS, LR, BATCH, "nnue")
        print(f"SMOKE RESULT: linear={m1:.1f} nnue={m2:.1f} "
              f"(nnue must beat linear)", flush=True)
        sys.exit(0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    print("\n" + "=" * 55)
    print("  BEST ARCHITECTURE")
    print("=" * 55)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  best MAE: {study.best_value:.1f}cp")
    print("\nTop 5:")
    for t in sorted(study.trials, key=lambda x: x.value)[:5]:
        print(f"  #{t.number}: MAE={t.value:.1f}  "
              f"params={t.user_attrs.get('params', 0):,}  {t.params}")
