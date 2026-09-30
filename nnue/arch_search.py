#!/usr/bin/env python3
"""NNUE Architecture Search v3 — V14/V15 component zoo on the DCU bench.

Forks the production LNNUE (same init, activations, loss) and searches NEW
components around the confirmed V13 core (L1=768, FM, cross, tail 16/64):
factorized FT buckets, material-bucketed tails, cross variants, activation
family, derived structural side features, FM rank 4..64.

Scoring is cost-aware: objective = MAE + lambda * (cost/cost_anchor - 1),
so extra eval cost must buy accuracy (accuracy-per-latency discipline).
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

# V13 core = confirmed anchor (252.2cp). Cost anchor derives from it.
ANCHOR = dict(L1=768, fm_rank=8, cross="sigmoid", act="screlu", buckets=1,
              ft_factor=32, side_feats=0, tail_l2=16, tail_l3=64)


def _act_fn(name, h):
    c = torch.clamp(h, 0.0, 1.0)
    if name == "screlu":  return c * c
    if name == "crelu":   return c
    if name == "cube":    return c * c * c
    raise ValueError(name)


class SearchableNNUE(nn.Module):
    def __init__(self, L1=768, fm_rank=8, cross="sigmoid", act="screlu",
                 buckets=1, ft_factor=32, side_feats=False,
                 tail_l2=16, tail_l3=64):
        super().__init__()
        self.L1, self.fm_rank, self.cross, self.act = L1, fm_rank, cross, act
        self.buckets, self.ft_factor = buckets, ft_factor
        self.side_feats = bool(side_feats)
        self.fact = ft_factor < 32
        ft_rows = (ft_factor * 768 + 768 + 1) if self.fact else (NUM_INPUTS + 1)
        self.pad_row = ft_rows - 1
        self.ft = nn.EmbeddingBag(ft_rows, L1, mode="sum", padding_idx=self.pad_row)
        self.ft_bias = nn.Parameter(torch.zeros(L1))
        nn.init.normal_(self.ft.weight, std=0.2)
        self.ft.weight.data[self.pad_row].zero_()
        if fm_rank > 0:
            self.fm_v = nn.Embedding(NUM_INPUTS + 1, fm_rank, padding_idx=NUM_INPUTS)
            nn.init.normal_(self.fm_v.weight, std=0.5)
            self.fm_v.weight.data[NUM_INPUTS].zero_()
            if cross in ("sigmoid", "double", "linear"):
                self.cross_w = nn.Linear(fm_rank, fm_rank)
            if cross == "double":
                self.cross_w2 = nn.Linear(fm_rank, fm_rank)
        if self.side_feats:
            self.side_proj = nn.Linear(33, 32)
            nn.init.normal_(self.side_proj.weight, std=0.05)
            nn.init.zeros_(self.side_proj.bias)
        W = 2 * L1 + (2 * fm_rank if fm_rank > 0 else 0) + (32 if self.side_feats else 0)
        self.in_dim = W
        K = buckets
        # tail init = plain nn.Linear defaults (production), copied to all buckets
        def pL(i, o):
            lin = nn.Linear(i, o)
            w = lin.weight.data.unsqueeze(0).repeat(K, 1, 1).clone()
            b = lin.bias.data.unsqueeze(0).repeat(K, 1).clone()
            return nn.Parameter(w), nn.Parameter(b)
        self.l2_w, self.l2_b = pL(W, tail_l2)
        self.l3_w, self.l3_b = pL(tail_l2, tail_l3)
        self.out_w, self.out_b = pL(tail_l3, 1)

    def _acc(self, idx):
        if not self.fact:
            return self.ft(idx) + self.ft_bias
        valid = idx < NUM_INPUTS
        plane = idx % 768
        bkt = (idx // 768).clamp(max=self.ft_factor - 1)
        pad = self.pad_row
        base = torch.where(valid, bkt * 768 + plane, torch.full_like(idx, pad))
        shared = torch.where(valid, torch.full_like(idx, self.ft_factor * 768) + plane,
                             torch.full_like(idx, pad))
        return self.ft(base) + self.ft(shared) + self.ft_bias

    def _interact(self, idx):
        v = self.fm_v(idx)
        s = v.sum(dim=1)
        i_raw = 0.5 * (s ** 2 - (v ** 2).sum(dim=1))
        h = torch.clamp(i_raw, -2.0, 2.0) / 2.0
        if self.cross in ("sigmoid", "double"):
            h = h + h * torch.sigmoid(self.cross_w(h))
            if self.cross == "double":
                h = h + h * torch.sigmoid(self.cross_w2(h))
        elif self.cross == "linear":
            h = h + self.cross_w(h)
        return h

    def _tail(self, x, bidx):
        outs = torch.empty(x.shape[0], device=x.device)
        for k in range(self.buckets):
            m = bidx == k
            if not bool(m.any()):
                continue
            h = torch.nn.functional.linear(x[m], self.l2_w[k], self.l2_b[k])
            h = _act_fn(self.act, h)
            h = torch.nn.functional.linear(h, self.l3_w[k], self.l3_b[k])
            h = _act_fn(self.act, h)
            outs[m] = torch.nn.functional.linear(h, self.out_w[k], self.out_b[k]).squeeze(-1)
        return outs

    def forward(self, w_idx, b_idx, stm, side, men):
        acc_w = self._acc(w_idx)
        acc_b = self._acc(b_idx)
        m = stm.view(-1, 1).float()
        h = torch.cat([m * acc_w + (1 - m) * acc_b,
                       (1 - m) * acc_w + m * acc_b], dim=1)
        if self.fm_rank > 0:
            h = torch.cat([h, self._interact(w_idx), self._interact(b_idx)], dim=1)
        if self.side_feats:
            h = torch.cat([h, self.side_proj(side)], dim=1)
        h = _act_fn(self.act, h)
        bidx = torch.clamp((men - 1) // 4, 0, self.buckets - 1)
        return self._tail(h, bidx) * 300.0

    def eval_cost_bytes(self):
        ft = 64 * self.L1 * 2 * (2 if self.fact else 1)
        fm = 64 * self.fm_rank * 2 if self.fm_rank > 0 else 0
        tl2 = self.l2_w.shape[1]
        tl3 = self.l3_w.shape[1]
        tail = self.buckets * (self.in_dim * tl2 + tl2 * tl3 + tl3 + 2 * tl2)
        return ft + fm + tail


def anchor_cost_bytes():
    return SearchableNNUE(**ANCHOR).eval_cost_bytes()


# ----------------------------------------------------------------------------
class Data:
    """npz packed by sample_stream.py + derived structural side features."""
    def __init__(self, path, device):
        z = np.load(path)
        n = len(z["t"])
        split = int(n * 0.88)
        ev0 = split + 20000
        def mov(x, dt): return torch.from_numpy(x).to(device=device, dtype=dt)
        w_all = mov(z["w"], torch.long); b_all = mov(z["b"], torch.long)
        s_all = mov(z["s"], torch.float32); t_all = mov(z["t"], torch.float32)
        side_all, men_all = self._derive(w_all)
        self.tr = [w_all[:split], b_all[:split], s_all[:split], t_all[:split],
                   side_all[:split], men_all[:split]]
        self.ev = [w_all[ev0:], b_all[ev0:], s_all[ev0:], t_all[ev0:],
                   side_all[ev0:], men_all[ev0:]]
        print(f"data: {split:,} train / {len(self.ev[3]):,} eval | "
              f"|t| mean {self.tr[3].abs().mean():.0f}cp", flush=True)

    @staticmethod
    def _derive(w):
        # w rows are white-pov features; plane = (idx%768)//64:
        # 0=wp 1=bp ... 10=wk 11=bk. Squares are orientation-frame (consistent).
        n = w.shape[0]
        valid = (w != NUM_INPUTS).float()
        planes = ((w % 768) // 64).clamp(0, 11)
        counts = torch.zeros(n, 12, device=w.device).scatter_add_(
            1, planes, valid)
        sq = w & 63
        f = (sq & 7).clamp(0, 7)
        wp = (planes == 0).float() * valid
        bp = (planes == 1).float() * valid
        wfiles = torch.zeros(n, 8, device=w.device).scatter_add_(1, f, wp)
        bfiles = torch.zeros(n, 8, device=w.device).scatter_add_(1, f, bp)
        wk = (sq * (planes == 10).long()).sum(1)   # exactly one king per row
        bk = (sq * (planes == 11).long()).sum(1)
        men = (w != NUM_INPUTS).sum(1)
        side = torch.cat([counts / 8.0, wfiles, bfiles,
                          (wk & 7).unsqueeze(1).float() / 7.0,
                          (wk >> 3).unsqueeze(1).float() / 7.0,
                          (bk & 7).unsqueeze(1).float() / 7.0,
                          (bk >> 3).unsqueeze(1).float() / 7.0,
                          men.unsqueeze(1).float() / 32.0], dim=1)
        assert side.shape[1] == 33
        return side, men

    def train_batch(self, bs):
        i = random.randint(0, len(self.tr[3]) - bs - 1)
        return [x[i:i + bs] for x in self.tr]

    def eval_mae(self, model, bs=65536):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(self.ev[3]), bs):
                ch = [x[i:i + bs] for x in self.ev]
                p = model(ch[0], ch[1], ch[2], ch[4], ch[5])
                tot += (p - ch[3]).abs().sum().item()
                cnt += len(p)
        model.train()
        return tot / cnt


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
        w, b, s, t, side, men = data.train_batch(bs)
        pred = model(w, b, s, side, men)
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


def suggest(trial):
    p = {}
    p["L1"] = trial.suggest_categorical("L1", [512, 768])
    p["fm_rank"] = trial.suggest_categorical("fm_rank", [0, 4, 8, 16, 32, 64])
    p["cross"] = (trial.suggest_categorical("cross", ["none", "sigmoid", "linear", "double"])
                  if p["fm_rank"] > 0 else "none")
    p["act"] = trial.suggest_categorical("act", ["screlu", "crelu", "cube"])
    p["buckets"] = trial.suggest_categorical("buckets", [1, 4, 8])
    p["ft_factor"] = trial.suggest_categorical("ft_factor", [4, 8, 16, 32])
    p["side_feats"] = trial.suggest_categorical("side_feats", [0, 1])
    p["tail_l2"], p["tail_l3"] = 16, 64   # confirmed anchor, frozen
    return p


def objective(trial):
    p = suggest(trial)
    model = SearchableNNUE(**p)
    params = sum(q.numel() for q in model.parameters())
    cost = model.eval_cost_bytes()
    print(f"trial {trial.number}: {p} params={params:,} cost={cost/1024:.0f}KB",
          flush=True)
    best, el = run_trial(model, DATA, STEPS, LR, BATCH, f"t{trial.number}")
    penalty = COST_LAMBDA * (cost / ANCHOR_COST - 1.0)
    score = best + penalty
    print(f"trial {trial.number}: MAE={best:.1f} penalty={penalty:+.1f} "
          f"score={score:.1f} ({el:.0f}s)", flush=True)
    trial.set_user_attr("params", params)
    trial.set_user_attr("cost_kb", round(cost / 1024))
    trial.set_user_attr("mae", round(best, 1))
    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="real_data.npz")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-lambda", type=float, default=20.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=None,
                    help="JSON dict overriding ANCHOR for a single run")
    args = ap.parse_args()

    STEPS, BATCH, LR, COST_LAMBDA = args.steps, args.batch, args.lr, args.cost_lambda
    ANCHOR_COST = anchor_cost_bytes()
    print(f"anchor cost: {ANCHOR_COST/1024:.0f}KB, lambda={COST_LAMBDA}cp per 100%",
          flush=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    DATA = Data(args.data, DEVICE)

    if args.config:
        import json
        p = dict(ANCHOR); p.update(json.loads(args.config))
        m, el = run_trial(SearchableNNUE(**p), DATA, STEPS, LR, BATCH, "cfg")
        print(f"CONFIG RESULT: {p} BEST MAE={m:.1f}cp ({el:.0f}s)", flush=True)
        sys.exit(0)

    if args.smoke:
        print("=== smoke: V13 anchor (must land ~252-258cp) ===", flush=True)
        m, el = run_trial(SearchableNNUE(**ANCHOR), DATA, STEPS, LR, BATCH, "anchor")
        print(f"SMOKE RESULT: anchor MAE={m:.1f}cp ({el:.0f}s)", flush=True)
        sys.exit(0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    print("\nTop 8 by penalized score (raw MAE in attrs):")
    for t in sorted(study.trials, key=lambda x: x.value)[:8]:
        print(f"  #{t.number}: score={t.value:.1f} mae={t.user_attrs.get('mae')} "
              f"cost={t.user_attrs.get('cost_kb')}KB {t.params}")
