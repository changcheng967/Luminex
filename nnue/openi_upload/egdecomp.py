#!/usr/bin/env python3
"""Fine-grained EG decomposition: which endgame TYPES carry the SF-error gap?

Buckets EG positions (npm<10) by material configuration and reports our
depth-1 SF-error vs V36's per type. The worst types name the term family.
"""
import subprocess
import sys

STASH = "/hyperai/home/stash-36.0-linux-x86_64-bmi2"
OURS = "/hyperai/home/sprt_base_clean"

def mat_conf(fen):
    board = fen.split()[0]
    c = {"P":0,"N":0,"B":0,"R":0,"Q":0,"p":0,"n":0,"b":0,"r":0,"q":0}
    for ch in board:
        if ch in c:
            c[ch] += 1
    w = (c["P"], c["N"], c["B"], c["R"], c["Q"])
    b = (c["p"], c["n"], c["b"], c["r"], c["q"])
    npm = 3*(c["N"]+c["n"]+c["B"]+c["b"]) + 5*(c["R"]+c["r"]) + 9*(c["Q"]+c["q"])
    pawns = c["P"] + c["p"]
    # symmetric type key: sorted piece multisets both sides
    key = "/".join(sorted([
        "P"*c["P"] + "N"*c["N"] + "B"*c["B"] + "R"*c["R"] + "Q"*c["Q"],
        "P"*c["p"] + "N"*c["n"] + "B"*c["b"] + "R"*c["r"] + "Q"*c["q"],
    ]))
    return npm, pawns, key

def evals(eng, fens, static):
    p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while "uciok" not in p.stdout.readline():
        pass
    out = []
    for fen in fens:
        stm = " w " in fen
        cmd = "eval\n" if static else "go depth 1\n"
        p.stdin.write(f"position fen {fen}\n{cmd}"); p.stdin.flush()
        cp = None
        guard = 0
        while guard < 400:
            ln = p.stdout.readline().strip()
            guard += 1
            if static:
                if ln.startswith("eval cp"):
                    try: cp = int(ln.split()[2])
                    except (IndexError, ValueError): pass
                    break
            else:
                if ln.startswith("info depth 1 ") and " score cp " in ln and " pv " in ln:
                    try: cp = int(ln.split(" score cp ")[1].split()[0])
                    except (IndexError, ValueError): pass
                elif ln.startswith("bestmove"):
                    break
        out.append(cp if stm else (-cp if cp is not None else None))
    p.stdin.write("quit\n"); p.stdin.flush()
    return out

def main():
    rows = []
    for ln in open(sys.argv[1]):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3:
            rows.append((f[0], int(float(f[2]))))
        if len(rows) >= int(sys.argv[2]):
            break
    # keep only EG positions
    eg = [(f, s) for f, s in rows if mat_conf(f)[0] < 10]
    fens = [r[0] for r in eg]
    sf = [r[1] for r in eg]
    print(f"EG positions: {len(eg)} of {len(rows)}", flush=True)
    ours = evals(OURS, fens, static=False)
    print("ours done", flush=True)
    stash = evals(STASH, fens, static=False)
    print("stash done", flush=True)

    import collections
    bytype = collections.defaultdict(list)
    for fen, s, o, t in zip(fens, sf, ours, stash):
        if o is None or t is None:
            continue
        npm, pawns, key = mat_conf(fen)
        bytype[key].append((abs(s - o), abs(s - t)))
    print(f"{'material type':<22} {'n':>4} {'ourErr':>8} {'v36Err':>8} {'gap':>7}")
    stats = []
    for k, v in bytype.items():
        if len(v) < 5:
            continue
        n = len(v)
        g_us = sum(x[0] for x in v) / n
        g_st = sum(x[1] for x in v) / n
        stats.append((g_us - g_st, k, n, g_us, g_st))
    stats.sort(reverse=True)
    for gap, k, n, g_us, g_st in stats[:18]:
        print(f"{k:<22} {n:>4} {g_us:>8.1f} {g_st:>8.1f} {gap:>+7.1f}")

if __name__ == "__main__":
    main()
