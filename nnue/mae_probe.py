#!/usr/bin/env python3
"""MAE probe: load any LNN1 .nnue, evaluate the labeled FEN sample, report
clamp(+-1500) MAE exactly in the trainer HEALTH view (stm-relative cp).
Usage: python mae_probe.py net1.nnue net2.nnue ..."""
import os, sys, struct, glob
import numpy as np
import torch
import chess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "openi_upload"))
from luminex_nnue_train import active_features, MAX_PIECES, NUM_INPUTS

DEV = "cpu"

def load_lnn1(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"LNN1", f"{path}: not LNN1"
        L1, L2, L3, NI = struct.unpack("iiii", f.read(16))
        def rt():
            (n,) = struct.unpack("i", f.read(4))
            return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
        ft_w = torch.from_numpy(rt().reshape(L1, NI))          # [L1, NI]
        ft_b = torch.from_numpy(rt())                           # [L1]
        l2w = torch.from_numpy(rt().reshape(L2, 2 * L1)); l2b = torch.from_numpy(rt())
        l3w = torch.from_numpy(rt().reshape(L3, L2));     l3b = torch.from_numpy(rt())
        ow = torch.from_numpy(rt().reshape(1, L3));       ob = torch.from_numpy(rt())
    return dict(L1=L1, L2=L2, L3=L3, NI=NI, ft_w=ft_w, ft_b=ft_b,
                l2w=l2w, l2b=l2b, l3w=l3w, l3b=l3b, ow=ow, ob=ob)

def forward(net, w_idx, b_idx, stm):
    # old nets export without the pad row; append a zero row so pad indices are no-ops
    ftw = torch.cat([net["ft_w"].t(), torch.zeros(1, net["L1"])])  # [NI+1, L1]
    acc_w = ftw[w_idx].sum(dim=1) + net["ft_b"]
    acc_b = ftw[b_idx].sum(dim=1) + net["ft_b"]
    sm = stm.view(-1, 1)
    h = torch.cat([sm * acc_w + (1 - sm) * acc_b,
                   (1 - sm) * acc_w + sm * acc_b], dim=1)
    h = torch.clamp(h, 0, 1) ** 2
    h = torch.clamp(h @ net["l2w"].t() + net["l2b"], 0, 1) ** 2
    h = torch.clamp(h @ net["l3w"].t() + net["l3b"], 0, 1) ** 2
    return (h @ net["ow"].t() + net["ob"]).squeeze(-1) * 300.0

def featurize(fens):
    B = len(fens)
    w_idx = torch.full((B, MAX_PIECES), NUM_INPUTS, dtype=torch.long)
    b_idx = torch.full((B, MAX_PIECES), NUM_INPUTS, dtype=torch.long)
    stm = torch.zeros(B)
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        w, b = active_features(board)
        w_idx[i, :len(w)] = torch.tensor(w, dtype=torch.long)
        b_idx[i, :len(b)] = torch.tensor(b, dtype=torch.long)
        stm[i] = 1.0 if board.turn == chess.WHITE else 0.0
    return w_idx, b_idx, stm

def main():
    lines = [l.rstrip("\n").split("\t") for l in open("nnue/sample_fens.txt") if l.strip()]
    fens = [l[0] for l in lines]
    stm_white = torch.tensor([float(l[1]) for l in lines])
    target = torch.tensor([float(l[2]) for l in lines]).clamp(-1500, 1500)
    print(f"sample: {len(fens)} positions, tgt std={target.std():.0f}", flush=True)
    w_idx, b_idx, stm = featurize(fens)
    nets = []
    for pat in sys.argv[1:]:
        for p in sorted(glob.glob(pat)):
            nets.append(p)
    print(f"{'net':34s} {'MAE':>7s} {'p50':>6s} {'p90':>7s} {'predSTD':>8s} {'bias':>7s}")
    for p in nets:
        try:
            net = load_lnn1(p)
            pred = torch.empty(len(fens))
            with torch.no_grad():
                for i in range(0, len(fens), 4096):
                    pred[i:i+4096] = forward(net, w_idx[i:i+4096], b_idx[i:i+4096], stm[i:i+4096])
            pred = pred.clamp(-1500, 1500)
            err = (pred - target).abs()
            print(f"{os.path.basename(p):34s} {err.mean():7.1f} {err.median():6.0f} "
                  f"{err.quantile(0.9):7.0f} {pred.std():8.0f} {(pred-target).mean():7.1f}  L1={net['L1']}")
        except Exception as e:
            print(f"{os.path.basename(p):34s} FAILED: {e}")

if __name__ == "__main__":
    main()
