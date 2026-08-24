#!/usr/bin/env python3
"""Weakness map: where does fit6's eval disagree with Stash V36's?

Takes gamepack positions (fen<TAB>stm<TAB>sf_eval), gets fit6 static evals
(UCI `eval`) and Stash depth-1 evals (UCI `go depth 1`), then segments
fit6-vs-Stash disagreement by position class so the top regimes name the
weak term families. SF eval from the file is the third reference: a regime
where Stash tracks SF better than we do = accuracy gap = our weakness.
"""
import subprocess
import sys

STASH = "/hyperai/home/stash-36.0-linux-x86_64-bmi2"
OURS = "/hyperai/home/sprt_base_clean"

VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}

def board_features(fen):
    board = fen.split()[0]
    counts = {"P":0,"N":0,"B":0,"R":0,"Q":0,"p":0,"n":0,"b":0,"r":0,"q":0}
    for ch in board:
        if ch in counts:
            counts[ch] += 1
    npm = 3*(counts["N"]+counts["n"]) + 3*(counts["B"]+counts["b"]) \
        + 5*(counts["R"]+counts["r"]) + 9*(counts["Q"]+counts["q"])
    pawns = counts["P"] + counts["p"]
    w_mat = sum(VAL[c.lower()] * counts[c] for c in "PNBRQ")
    b_mat = sum(VAL[c] * counts[c.upper()] for c in "pnbrq")
    queens = counts["Q"] + counts["q"]
    rooks = counts["R"] + counts["r"]
    minors = counts["N"]+counts["n"]+counts["B"]+counts["b"]
    return npm, pawns, w_mat - b_mat, queens, rooks, minors

def evals(eng, fens, stash=False, static=False):
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

def seg(npm):
    if npm >= 20: return "MG"
    if npm >= 10: return "MG-EG"
    return "EG"

def main():
    rows = []
    for ln in open(sys.argv[1]):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3:
            rows.append((f[0], int(float(f[2]))))
        if len(rows) >= int(sys.argv[2]):
            break
    fens = [r[0] for r in rows]
    sf = [r[1] for r in rows]
    print(f"loaded {len(rows)} positions", flush=True)
    ours = evals(OURS, fens, static=True)
    print("fit6 evals done", flush=True)
    stash = evals(STASH, fens)
    print("stash evals done", flush=True)

    import collections
    buckets = collections.defaultdict(list)
    for fen, s, o, t in zip(fens, sf, ours, stash):
        if o is None or t is None:
            continue
        npm, pawns, imbal, queens, rooks, minors = board_features(fen)
        ph = seg(npm)
        d = (o - t, abs(s - o), abs(s - t))   # (bias, our SF-gap, their SF-gap)
        buckets[("phase", ph)].append(d)
        buckets[("sharp", "sharp" if abs(s) > 150 else "quiet")].append(d)
        buckets[("pawns", "closed" if pawns >= 14 else "open" if pawns <= 8 else "mid")].append(d)
        buckets[("imbalance", "imbal" if abs(imbal) >= 3 else "balanced")].append(d)
        buckets[("queens", "Q" if queens > 0 else "noQ")].append(d)
        buckets[("rooks", "R2+" if rooks >= 2 else "R<2")].append(d)
    print(f"{'segment':<24} {'n':>5} {'bias(f6-V36)':>13} {'MAE_f6vV36':>11} {'sfMAE_us':>9} {'sfMAE_V36':>10}")
    for k in sorted(buckets, key=str):
        v = buckets[k]
        n = len(v)
        bias = sum(x[0] for x in v) / n
        mae = sum(abs(x[0]) for x in v) / n
        g_us = sum(x[1] for x in v) / n
        g_st = sum(x[2] for x in v) / n
        # sfGap_us - sfGap_stash > 0 means we're FARTHER from SF = weaker there
        print(f"{str(k):<24} {n:>5} {bias:>13.1f} {mae:>11.1f} {g_us:>9.1f} {g_st:>10.1f}")

if __name__ == "__main__":
    main()
