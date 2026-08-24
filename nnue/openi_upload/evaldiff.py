#!/usr/bin/env python3
"""Differential: our Stash-eval port vs the real Stash binary on shared FENs."""
import subprocess
import sys

STASH = "/hyperai/home/stash-36.0-linux-x86_64-bmi2"
OURS = "/hyperai/home/luminex_stasheval"

fens = [ln.split("\t")[0] for ln in open(sys.argv[1]) if ln.strip()][:int(sys.argv[2])]

def evals(eng, stash=False):
    p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while "uciok" not in p.stdout.readline():
        pass
    if not stash:
        p.stdin.write("setoption name UseStashEval value true\n"); p.stdin.flush()
    out = []
    for fen in fens:
        stm = " w " in fen
        p.stdin.write(f"position fen {fen}\ngo depth 1\n"); p.stdin.flush()
        cp = None
        guard = 0
        while guard < 400:
            ln = p.stdout.readline().strip()
            guard += 1
            if ln.startswith("info depth 1 ") and " score cp " in ln and " pv " in ln:
                try:
                    cp = int(ln.split(" score cp ")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif ln.startswith("bestmove"):
                break
        if cp is not None:
            out.append(cp if stm else -cp)
        else:
            out.append(None)
    p.stdin.write("quit\n"); p.stdin.flush()
    return out

a = evals(STASH, True)
b = evals(OURS)
pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
n = len(pairs)
a2 = [x for x, _ in pairs]
b2 = [y for _, y in pairs]
d = [x - y for x, y in pairs]
if not n:
    print("no data — eval output not parsed (check real-stash eval format)")
    sys.exit(0)
print(f"n={n}/{len(fens)} mean_diff={sum(d)/n:.1f} mae={sum(abs(x) for x in d)/n:.1f}")
q = sorted(abs(x) for x in d)
print(f"p50={q[n//2]} p90={q[int(n*0.9)]} p99={q[int(n*0.99)] if n>100 else q[-1]}")
big = sorted(range(n), key=lambda i: -abs(d[i]))[:6]
for i in big:
    print(f"  diff={d[i]:+d} stash={a2[i]} ours={b2[i]} fen={fens[i][:58]}")
