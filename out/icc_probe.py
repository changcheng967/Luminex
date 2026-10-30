#!/usr/bin/env python3
"""Sibling-error-correlation probe (agent-1 idea 1, adapted to available data).

Hypothesis: playing strength tracks the COVARIANCE of eval error across
positions search compares (siblings), not per-position error. Class-driven
bias (correlated error) survives any depth; idiosyncratic error washes out.

The corpus has no labelled sibling sets, so two proxies are measured on a
game-ordered 300K slice (signed error = pred - target, stm-relative):
  A. class-ICC  - one-way random-effects ICC of signed error within material
     classes (piece-count vectors). Class-driven bias => high ICC.
  B. traj-lag1  - Pearson autocorrelation of signed error between consecutive
     positions of the same game. Class-driven bias => high autocorr.

Nets (all .pt float checkpoints on the DCU NFS):
  T0    luminex_sfq_full.pt   quiet 6k   (the +106 net)
  r0u   session1/r0u.pt       UNWEIGHTED 6k  (weight control: dose matched)
  sf20k gen768sf_full.pt      unweighted 20k (dose point: the -106 net)
  sf42k gen768sf.pt           unweighted 42k (deeper past the cliff)

Pre-registered reads:
  H1 weight->decorrelation: corr(T0) < corr(r0u) on A or B (or both)
  H2 dose->correlation:     corr(T0) < corr(sf20k) on A or B
  Clear margin: >= 0.02 absolute ICC difference or consistent ordering on
  both statistics. Equal/reversed => the covariance story is dead.
"""
import os, subprocess, sys
import numpy as np
import torch

sys.path.insert(0, "/work/home/scnmyrr3ge/luminex-dcu/gates/Luminex/nnue/openi_upload")
from luminex_nnue_train import LNNUE  # noqa: E402

N = 300_000
BASE = "/work/home/scnmyrr3ge/luminex-dcu"
CORPUS = f"{BASE}/sf_train_full_q.tsv.xz"
FEAT = f"{BASE}/luminex-featurize-fresh"
REC = 140
NETS = [
    ("T0",    f"{BASE}/luminex_sfq_full.pt"),
    ("r0u",   f"{BASE}/session1/r0u.pt"),
    ("sf20k", f"{BASE}/gen768sf_full.pt"),
    ("sf42k", f"{BASE}/gen768sf.pt"),
]

os.environ.setdefault("OMP_NUM_THREADS", "2")

# ---- featurized records for net inputs + targets ----
p = subprocess.Popen(["bash", "-c", f"xz -dc {CORPUS} | head -n {N} | {FEAT} --fen-eval --threads 4"],
                     stdout=subprocess.PIPE)
recs = bytearray()
while len(recs) < N * REC:
    d = p.stdout.read(1 << 20)
    if not d:
        break
    recs += d
p.stdout.close(); p.wait()
n = len(recs) // REC
va = np.frombuffer(bytes(recs[:n * REC]), dtype=np.uint8).reshape(n, REC)
wi = torch.from_numpy(va[:, :64].copy().view(np.int16).reshape(-1, 32)).long()
bi = torch.from_numpy(va[:, 64:128].copy().view(np.int16).reshape(-1, 32)).long()
si = torch.from_numpy(va[:, 128:132].copy().view(np.float32).reshape(-1))
ti = torch.from_numpy(va[:, 132:136].copy().view(np.float32).reshape(-1)).numpy().clip(-1500, 1500)
print(f"[data] {n:,} positions", flush=True)

# ---- raw FENs (same head -n stream) for game boundaries + material classes ----
pf = subprocess.Popen(["bash", "-c", f"xz -dc {CORPUS} | head -n {N} | cut -f1"],
                      stdout=subprocess.PIPE)
fens = []
for _ in range(n):
    line = pf.stdout.readline()
    if not line:
        break
    fens.append(line.decode("ascii", "replace").strip())
pf.stdout.close(); pf.wait()
assert len(fens) == n, f"FEN/rec misalignment: {len(fens)} vs {n}"

game_id = np.empty(n, dtype=np.int64)
mat_key = np.empty(n, dtype=np.int64)
gid = 0
prev_fullmove = 10 ** 9
for i, fen in enumerate(fens):
    parts = fen.split()
    if len(parts) < 6:
        parts = fen.split() or ["x"] * 6  # defensive; count below will fail loudly
    board = parts[0]
    try:
        fm = int(parts[5])
    except ValueError:
        fm = prev_fullmove
    if fm < prev_fullmove:
        gid += 1
    prev_fullmove = fm
    game_id[i] = gid
    counts = tuple(board.count(c) for c in "PNBRQpnbrq")
    mat_key[i] = hash(counts)

same_game = game_id[1:] == game_id[:-1]
print(f"[games] {gid:,} games detected; same-game adjacent pairs {int(same_game.sum()):,}", flush=True)


def pred_of(path):
    m = LNNUE(L1=768)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, n, 100_000):
            e = min(s + 100_000, n)
            out.append(m(wi[s:e], bi[s:e], si[s:e]).float().numpy())
    return np.concatenate(out)


def class_icc(err):
    """One-way random-effects ICC(1) of signed error across material classes."""
    ok = np.zeros(len(err), dtype=bool)
    order = np.argsort(mat_key, kind="stable")
    sk = mat_key[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    sizes = np.diff(np.r_[starts, len(sk)])
    grand = err.mean()
    sw_num, sw_den, sb_num = 0.0, 0, 0.0
    G = int((sizes >= 50).sum())
    kbar_num = 0
    for st, sz in zip(starts, sizes):
        if sz < 50:
            continue
        e = err[order[st:st + sz]]
        sw_num += ((e - e.mean()) ** 2).sum()
        sw_den += sz - 1
        sb_num += sz * (e.mean() - grand) ** 2
        kbar_num += sz
    if G < 10 or sw_den == 0:
        return float("nan"), G
    msw = sw_num / sw_den
    msb = sb_num / (G - 1)
    kbar = (kbar_num - (kbar_num / G)) / max(G - 1, 1)  # avg class size (approx)
    icc = (msb - msw) / (msb + max(kbar - 1, 1) * msw)
    return icc, G


def lag1(err):
    a, b = err[:-1][same_game], err[1:][same_game]
    return float(np.corrcoef(a, b)[0, 1])


print(f"{'net':6s} {'MAE':>7s} {'classICC':>9s} {'(G)':>6s} {'lag1':>7s}")
results = {}
for name, path in NETS:
    err = pred_of(path) - ti
    icc, G = class_icc(err)
    l1 = lag1(err)
    results[name] = (icc, l1)
    print(f"{name:6s} {np.abs(err).mean():7.1f} {icc:9.4f} {G:6d} {l1:7.4f}", flush=True)

t0i, t0l = results["T0"]
for other in ("r0u", "sf20k", "sf42k"):
    oi, ol = results[other]
    print(f"T0 vs {other:5s}: dICC={t0i - oi:+.4f}  dlag1={t0l - ol:+.4f}")
