#!/usr/bin/env python3
"""Verify the +0.43 reinterpretation: convert signed stm-relative error to
white-relative by flipping the sign on black-to-move positions, then re-measure
lag-1 autocorrelation within games. Prediction (agent 1): ~ +0.43 on all nets,
invariant to recipe/dose. Also re-check class-ICC in white-relative space."""
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
stm = si.numpy()  # 1.0 = white to move, 0.0 = black (per trainer gate)

pf = subprocess.Popen(["bash", "-c", f"xz -dc {CORPUS} | head -n {N} | cut -f1"], stdout=subprocess.PIPE)
fens = [pf.stdout.readline().decode("ascii", "replace").strip() for _ in range(n)]
pf.stdout.close(); pf.wait()

game_id = np.empty(n, dtype=np.int64)
gid = 0
prev = 10 ** 9
for i, fen in enumerate(fens):
    parts = fen.split()
    try:
        fm = int(parts[5])
    except (ValueError, IndexError):
        fm = prev
    if fm < prev:
        gid += 1
    prev = fm
    game_id[i] = gid
same_game = game_id[1:] == game_id[:-1]


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


print(f"{'net':6s} {'lag1_stm':>9s} {'lag1_white':>11s}")
for name, path in NETS:
    err = pred_of(path) - ti
    l1_stm = float(np.corrcoef(err[:-1][same_game], err[1:][same_game])[0, 1])
    # white-relative: negate error where black is to move
    err_w = err * np.where(stm > 0.5, 1.0, -1.0)
    l1_w = float(np.corrcoef(err_w[:-1][same_game], err_w[1:][same_game])[0, 1])
    print(f"{name:6s} {l1_stm:9.4f} {l1_w:11.4f}", flush=True)
