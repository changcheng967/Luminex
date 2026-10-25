#!/usr/bin/env python3
"""Engine-side eval-scale probe: run two nets over the same FENs at low nodes
and report the |eval| quantile ratio (candidate vs incumbent). A ratio far
from 1.0 means pruning margins tuned for the incumbent misserve the candidate;
per the scale-match rule, gate only after accounting for it.
    python3 scale_probe.py <engine> <netA> <netB> [n_fens]
"""
import subprocess, sys, lzma, statistics

ENG, NETA, NETB = sys.argv[1], sys.argv[2], sys.argv[3]
NFEN = int(sys.argv[4]) if len(sys.argv) > 4 else 400
SRC = "/mnt/workspace/r1/sf_train_14m.tsv.xz"

fens = []
with lzma.open(SRC, "rt") as f:
    for line in f:
        tab = line.find("\t")
        if tab > 0:
            fens.append(line[:tab])
        if len(fens) >= NFEN:
            break

def evals(net):
    p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    out = []
    p.stdin.write(f"setoption name NNUEFile value {net}\n")
    for fen in fens:
        p.stdin.write(f"position fen {fen}\ngo nodes 400\n")
        p.stdin.flush()
        cp = None
        while True:
            ln = p.stdout.readline()
            if not ln or ln.startswith("bestmove"):
                break
            if "score cp" in ln and "upperbound" not in ln and "lowerbound" not in ln:
                cp = int(ln.split("score cp")[1].split()[0])
        out.append(cp)
    p.stdin.write("quit\n"); p.stdin.flush(); p.wait()
    return out

def q(v, p):  # quantile
    s = sorted(abs(x) for x in v if x is not None)
    return s[min(len(s) - 1, int(p * len(s)))]

ea, eb = evals(NETA), evals(NETB)
print(f"n={len(fens)}")
for name, pct in (("q50", 0.50), ("q75", 0.75), ("q90", 0.90)):
    qa, qb = q(ea, pct), q(eb, pct)
    print(f"{name}: A={qa} B={qb} ratio(B/A)={qb / max(qa, 1):.3f}")
import math
na = [x for x in ea if x is not None]; nb = [x for x in eb if x is not None]
d = [b - a for a, b in zip(na, nb) if a is not None and b is not None]
print(f"mean(evalB-evalA)={statistics.mean(d):+.1f}cp  stdev={statistics.stdev(d):.1f}cp")
