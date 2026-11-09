#!/usr/bin/env python3
"""How much of the network's capacity do real positions use?

usage: probe_net.py TRAINER_DIR FLOAT_NET REF.jsonl N_POS

On N_POS positions from refereed games (stride-sampled), runs the float net's
forward pass by hand and reports, per layer and per piece-count bucket:
  FT  (768 per perspective): share of lanes below 0 (SCReLU zero, no gradient),
      above 1 (saturated, no gradient), live in between; neurons that are dead
      or saturated on >= 99% of positions; lanes beyond the int16 range at the
      engine's x8192 scale (|a| >= 4.0);
  L2 (16) / L3 (32): the same live / dead / saturated split per neuron;
  the effective dimension of the 1536-wide activation (components for 90/99%
  of variance) and of the 16-wide L2 activation.
"""
import json
import os
import sys

import numpy as np

tdir, net, ref, n_pos = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, tdir)
os.environ["NNUE_DUAL_HEAD"] = "0"
os.environ["NNUE_BUCKETS"] = "1"
import torch  # noqa: E402

torch.set_num_threads(1)
import luminex_nnue_train as T  # noqa: E402

PT = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}


def feats(fen):
    board, stm = fen.split()[0], fen.split()[1]
    pcs, rank, file = [], 7, 0
    for ch in board:
        if ch == "/":
            rank, file = rank - 1, 0
        elif ch.isdigit():
            file += int(ch)
        else:
            pcs.append((rank * 8 + file, 1 if ch.isupper() else 0, PT[ch.lower()]))
            file += 1
    wk = next(s for s, c, t in pcs if t == 6 and c == 1)
    bk = next(s for s, c, t in pcs if t == 6 and c == 0)
    return ([T._halfka_idx(True, wk, s, c, t) for s, c, t in pcs],
            [T._halfka_idx(False, bk, s, c, t) for s, c, t in pcs], stm == "w", len(pcs))


def load_any(path):
    """LNN1 / LNB1 through the trainer; LNI8 dequantized (w8 / scale) into the same keys."""
    import struct
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"LNI8":
        return T.LNNUE.load_lnn1_state(path)
    with open(path, "rb") as f:
        f.read(4)
        L1, L2, L3, NI = struct.unpack("iiii", f.read(16))

        def rf():
            (n,) = struct.unpack("i", f.read(4))
            return torch.frombuffer(bytearray(f.read(n * 4)), dtype=torch.float32).clone()

        def ri8():
            (n,) = struct.unpack("i", f.read(4))
            return torch.frombuffer(bytearray(f.read(n)), dtype=torch.int8).clone().float()

        ft_w = rf().reshape(L1, NI)
        ft_b = rf()
        layers = []
        for rows, cols in ((L2, 2 * L1), (L3, L2), (1, L3)):
            w8 = ri8().reshape(rows, cols)
            b = rf()
            (s,) = struct.unpack("f", f.read(4))
            layers.append((w8 / s, b))
    sd = {"ft.weight": torch.cat([ft_w.t(), torch.zeros(1, L1)], 0), "ft_bias": ft_b}
    for name, (w, b) in zip(("l2", "l3", "out"), layers):
        sd[name + ".weight"], sd[name + ".bias"] = w, b.view(-1)
    return sd


def main():
    fens = []
    with open(ref) as f:
        lines = f.readlines()
    step = max(1, len(lines) // n_pos)
    for line in lines[::step][:n_pos]:
        fens.append(json.loads(line)["fen"])
    sd = load_any(net)
    L1 = sd["ft_bias"].numel()
    m = T.LNNUE(L1=L1)
    m.load_state_dict(sd, strict=False)
    m.eval()
    W = m.ft.weight.detach()
    if W.shape[0] in (T.NUM_INPUTS, T.NUM_INPUTS + 1):  # EmbeddingBag layout [inputs(+pad), L1]
        Wf = W[:T.NUM_INPUTS]
    else:                                               # Linear layout [L1, inputs]
        Wf = W.t()
    bias = m.ft_bias.detach()
    acc_all, h_all, l2_all, l3_all, bk_all, out_all = [], [], [], [], [], []
    with torch.no_grad():
        for fen in fens:
            w, b, white, npc = feats(fen)
            aw = Wf[w].sum(0) + bias
            ab = Wf[b].sum(0) + bias
            stm, nstm = (aw, ab) if white else (ab, aw)
            acc = torch.cat([stm, nstm])
            h = torch.clamp(acc, 0, 1) ** 2
            p2 = m.l2(h)
            h2 = torch.clamp(p2, 0, 1) ** 2
            p3 = m.l3(h2)
            h3 = torch.clamp(p3, 0, 1) ** 2
            out = m.out(h3).item() * 300
            acc_all.append(acc.numpy()); h_all.append(h.numpy()); l2_all.append(p2.numpy())
            l3_all.append(p3.numpy()); bk_all.append((npc - 1) * 8 // 32); out_all.append(out)
    A = np.array(acc_all)          # (N, 1536) pre-activation, stm half then nstm half
    H = np.array(h_all)
    P2 = np.array(l2_all)
    P3 = np.array(l3_all)
    BK = np.array(bk_all)
    N = len(A)
    print(f"{N} positions, L1={L1}")

    def split(x):
        return (x <= 0).mean(), (x >= 1).mean(), ((x > 0) & (x < 1)).mean()

    Aft = np.concatenate([A[:, :L1], A[:, L1:]])   # both perspectives pooled per FT neuron
    d, s, live = split(Aft)
    print(f"FT lanes: zero {d:.1%}  saturated {s:.1%}  live {live:.1%};  |a|>=4 (int16 x8192 overflow) "
          f"{(np.abs(Aft) >= 4.0).mean():.4%}, max |a| {np.abs(Aft).max():.2f}")
    live_n = ((Aft > 0) & (Aft < 1)).mean(0)
    dead_n = (Aft <= 0).mean(0)
    sat_n = (Aft >= 1).mean(0)
    print(f"FT neurons: dead on >=99% of lanes {int((dead_n >= 0.99).sum())}, saturated on >=99% "
          f"{int((sat_n >= 0.99).sum())}, live on <5% {int((live_n < 0.05).sum())}, "
          f"median live share {np.median(live_n):.1%}")
    for b in range(8):
        sel = BK == b
        if sel.sum() < 30:
            continue
        a = np.concatenate([A[sel, :L1], A[sel, L1:]])
        d, s, live = split(a)
        print(f"   bucket {b} ({sel.sum():5d} pos): FT zero {d:.1%} sat {s:.1%} live {live:.1%};"
              f"  L2 zero {(P2[sel] <= 0).mean():.1%} sat {(P2[sel] >= 1).mean():.1%};"
              f"  L3 zero {(P3[sel] <= 0).mean():.1%} sat {(P3[sel] >= 1).mean():.1%}")
    print("L2 neurons (live share / zero / saturated):")
    for j in range(P2.shape[1]):
        x = P2[:, j]
        print(f"   n{j:02d}: live {((x > 0) & (x < 1)).mean():.1%} zero {(x <= 0).mean():.1%} sat {(x >= 1).mean():.1%}")

    def eff_dim(X):
        X = X - X.mean(0)
        sv = np.linalg.svd(X, compute_uv=False) ** 2
        c = np.cumsum(sv) / sv.sum()
        return int(np.searchsorted(c, 0.90) + 1), int(np.searchsorted(c, 0.99) + 1)

    dead3 = int(((P3 <= 0).mean(0) >= 0.99).sum()); sat3 = int(((P3 >= 1).mean(0) >= 0.99).sum())
    dead2 = int(((P2 <= 0).mean(0) >= 0.99).sum()); sat2 = int(((P2 >= 1).mean(0) >= 0.99).sum())
    print(f"SUMMARY L2 dead {dead2}/{P2.shape[1]} stuck-saturated {sat2}; L3 dead {dead3}/{P3.shape[1]} stuck-saturated {sat3}")
    print("effective dimension (components for 90% / 99% variance):")
    print(f"   FT activation (1536): {eff_dim(H)}   L2 activation (16): {eff_dim(np.clip(P2, 0, 1) ** 2)}")
    for b in range(8):
        sel = BK == b
        if sel.sum() >= 200:
            print(f"   bucket {b}: FT {eff_dim(H[sel])}  L2 {eff_dim(np.clip(P2[sel], 0, 1) ** 2)}")


if __name__ == "__main__":
    main()
