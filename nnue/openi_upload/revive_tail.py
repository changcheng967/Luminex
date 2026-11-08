#!/usr/bin/env python3
"""Revive the dense tail's dead and stuck neurons without changing the network.

usage: python3 revive_tail.py IN_NET OUT_NET POSITIONS [N] [SEED]

IN_NET / OUT_NET are float nets (LNN1, or LNB1 with output buckets); quantize
OUT_NET with quantize_i8.py as usual. POSITIONS: a .jsonl with a "fen" field
per line, or a text file with the FEN as the first tab-separated column (the
training TSV works). N positions are stride-sampled (default 40000).

A unit of L2 or L3 whose pre-activation is <= 0 on every probe position (dead:
SCReLU outputs 0 and passes no gradient) or >= 1 on every one (stuck: it
outputs the constant 1, again with no gradient) can never learn again. Each
such unit is redrawn so it is live, and the network's output stays the same:
  - its outgoing weights are set to zero; a stuck unit's constant output is
    first folded into the next layer's bias;
  - its incoming weights are drawn N(0, 1) and rescaled so that its
    pre-activation has standard deviation 0.25 over the probe positions,
    clipped to the layer's existing max |w| (the int8 scale stays the same);
    its bias puts the median pre-activation at 0.5, inside SCReLU's live
    window (0, 1).
Training then grows the zeroed outgoing weights; the redrawn incoming weights
start learning as soon as those are non-zero. Each bucket's tail is probed on
the positions routed to it. REVIVE_STUCK=0 leaves stuck units alone: the
int8 net is then bit-identical to the input's (folding a stuck unit's constant
into a float bias removes that column's int8 rounding, a few cp).
Prints the counts and the largest output change on the probe positions.
"""
import json
import os
import sys

src, dst, pos_path = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["NNUE_DUAL_HEAD"] = "0"

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
torch.manual_seed(SEED)

with open(src, "rb") as f:
    _magic = f.read(4)
NB = 1
if _magic == b"LNB1":
    import struct
    with open(src, "rb") as f:
        f.read(20)
        NB = struct.unpack("i", f.read(4))[0]
os.environ["NNUE_BUCKETS"] = str(NB)
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
            [T._halfka_idx(False, bk, s, c, t) for s, c, t in pcs], 1.0 if stm == "w" else 0.0)


def read_fens(path, n):
    lines = open(path).read().splitlines()
    step = max(1, len(lines) // n)
    out = []
    for line in lines[::step][:n]:
        if not line.strip():
            continue
        out.append(json.loads(line)["fen"] if line.lstrip().startswith("{") else line.split("\t")[0])
    return out


def ft_activation(model, W, B, S):
    """SCReLU(stm ++ nstm accumulator), exactly as LNNUE.forward builds it."""
    if model.ft_mode == "gather":
        wt = model.ft.weight.t()
        acc_w = (wt[W.clamp(max=T.NUM_INPUTS - 1)] * (W < T.NUM_INPUTS).unsqueeze(-1).to(wt.dtype)).sum(1) + model.ft_bias
        acc_b = (wt[B.clamp(max=T.NUM_INPUTS - 1)] * (B < T.NUM_INPUTS).unsqueeze(-1).to(wt.dtype)).sum(1) + model.ft_bias
    else:
        acc_w = model.ft(W) + model.ft_bias
        acc_b = model.ft(B) + model.ft_bias
    m = S.view(-1, 1)
    h = torch.cat([m * acc_w + (1 - m) * acc_b, (1 - m) * acc_w + m * acc_b], dim=1)
    return torch.clamp(h, 0.0, 1.0) ** 2


def revive_layer(lin, nxt, x, nxt_bias_fold=True):
    """Revive dead/stuck units of `lin` (input activations x), keeping the
    function: zero their columns in `nxt` (fold a stuck unit's 1.0 into nxt's
    bias). Returns (dead, stuck) counts."""
    with torch.no_grad():
        p = lin(x)
        dead = (p <= 0).all(0)
        stuck = (p >= 1).all(0)
        if os.environ.get("REVIVE_STUCK", "1") == "0":
            stuck = torch.zeros_like(stuck)
        todo = (dead | stuck).nonzero().flatten().tolist()
        wmax = float(lin.weight.abs().max())
        for j in todo:
            if stuck[j]:
                nxt.bias += nxt.weight[:, j] * 1.0
            nxt.weight[:, j] = 0.0
            g = torch.randn(lin.weight.shape[1])
            z = x @ g
            g = g * (0.25 / max(float(z.std()), 1e-6))
            g = g.clamp(-wmax, wmax)
            z = x @ g
            lin.weight[j] = g
            lin.bias[j] = 0.5 - float(z.median())
        return int(dead.sum()), int(stuck.sum())


def main():
    sd = T.LNNUE.load_lnn1_state(src)
    L1 = sd["ft_bias"].numel()
    model = T.LNNUE(L1=L1)
    model.load_state_dict(sd, strict=False)
    model.eval()
    fens = read_fens(pos_path, N)
    n = len(fens)
    W = torch.full((n, 32), T.NUM_INPUTS, dtype=torch.long)
    B = torch.full((n, 32), T.NUM_INPUTS, dtype=torch.long)
    S = torch.zeros(n)
    npc = torch.zeros(n, dtype=torch.long)
    for i, fen in enumerate(fens):
        w, b, s = feats(fen)
        W[i, :len(w)] = torch.tensor(w)
        B[i, :len(b)] = torch.tensor(b)
        S[i] = s
        npc[i] = len(w)
    with torch.no_grad():
        before = model(W, B, S)
        H = ft_activation(model, W, B, S)
    if NB > 1:
        tails = [(model.bl2[b], model.bl3[b], model.bout[b], ((npc - 1) * NB // 32).clamp(0, NB - 1) == b)
                 for b in range(NB)]
    else:
        tails = [(model.l2, model.l3, model.out, torch.ones(n, dtype=torch.bool))]
    print(f"{src}: {'LNB1 %d buckets' % NB if NB > 1 else 'LNN1'}, L1={L1}, {n} probe positions")
    for k, (l2, l3, out, sel) in enumerate(tails):
        x = H[sel]
        if x.shape[0] < 50:
            print(f"  tail {k}: only {x.shape[0]} probe positions, skipped")
            continue
        d2, s2 = revive_layer(l2, l3, x)
        with torch.no_grad():
            h2 = torch.clamp(l2(x), 0.0, 1.0) ** 2
        d3, s3 = revive_layer(l3, out, h2)
        with torch.no_grad():
            p2 = l2(x)
            h2 = torch.clamp(p2, 0.0, 1.0) ** 2
            p3 = l3(h2)
            live2 = ((p2 > 0) & (p2 < 1)).float().mean(0)
            live3 = ((p3 > 0) & (p3 < 1)).float().mean(0)
        print(f"  tail {k} ({x.shape[0]} pos): revived L2 {d2} dead + {s2} stuck of {p2.shape[1]}, "
              f"L3 {d3} dead + {s3} stuck of {p3.shape[1]}; now live on >= 5% of positions: "
              f"L2 {int((live2 >= 0.05).sum())}/{p2.shape[1]}, L3 {int((live3 >= 0.05).sum())}/{p3.shape[1]}")
    with torch.no_grad():
        after = model(W, B, S)
    d = (after - before).abs()
    print(f"output change on the probe positions: max {float(d.max()):.6f} cp, mean {float(d.mean()):.6f} cp")
    T.save_nnue(model, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
