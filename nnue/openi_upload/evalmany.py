#!/usr/bin/env python3
"""Eval many FENs with one persistent engine process at fixed depth.
Input: samples.txt (fen\tgame\tcolor\twdl)  Output: same lines + \t<cp>"""
import subprocess
import sys
import time

DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 7
ENGINE = "/hyperai/home/Luminex/build_release/luminex"

p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\n"); p.stdin.flush()
while "uciok" not in p.stdout.readline():
    pass

def eval_fen(fen):
    p.stdin.write(f"position fen {fen}\n"); p.stdin.flush()
    p.stdin.write(f"go depth {DEPTH}\n"); p.stdin.flush()
    cp = None
    mate = None
    deadline = time.time() + 30
    while time.time() < deadline:
        ln = p.stdout.readline().strip()
        if not ln:
            break
        if ln.startswith("info") and " score " in ln and " pv " in ln:
            if " score cp " in ln:
                cp = int(ln.split(" score cp ")[1].split()[0]); mate = None
            elif " score mate " in ln:
                mate = int(ln.split(" score mate ")[1].split()[0]); cp = None
        if ln.startswith("bestmove"):
            break
    if cp is None and mate is not None:
        return 10000 if mate > 0 else -10000
    return cp

def main():
    src, dst = sys.argv[1], sys.argv[2]
    lines = [ln.rstrip("\n") for ln in open(src) if ln.strip()]
    with open(dst, "w") as fo:
        for i, ln in enumerate(lines):
            fen = ln.split("\t")[0]
            side = int(ln.split("\t")[2])
            cp = eval_fen(fen)
            if cp is not None:
                # normalize to OUR perspective
                cp = cp if side == 0 else -cp
                fo.write(f"{ln}\t{cp}\n")
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(lines)}", flush=True)
    print("DONE_EVALMANY")

if __name__ == "__main__":
    main()
