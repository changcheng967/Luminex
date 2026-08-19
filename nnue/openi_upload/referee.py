#!/usr/bin/env python3
"""SF-referee: analyze swing FENs with latest Stockfish (ground truth)."""
import subprocess
import sys
import time

SF = r"C:\Users\chang\Downloads\sf18\stockfish\stockfish-windows-x86-64-avx2.exe"
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 24

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("setoption name Threads value 4\nuci\n"); p.stdin.flush()
while True:
    if "uciok" in p.stdout.readline():
        break

def analyse(fen):
    p.stdin.write("ucinewgame\n"); p.stdin.flush()
    p.stdin.write(f"position fen {fen}\n"); p.stdin.flush()
    p.stdin.write(f"go depth {DEPTH}\n"); p.stdin.flush()
    cp = mate = None
    best = None
    deadline = time.time() + 90
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
            best = ln.split()[1]
            break
    return best, cp, mate

def main():
    src, dst = sys.argv[1], sys.argv[2]
    lines = [ln.strip() for ln in open(src) if ln.strip()]
    with open(dst, "w") as fo:
        fo.write("fen\tplayed\tsw\tphase\tply\tsf_bm\tsf_cp\tsf_mate\n")
        for i, ln in enumerate(lines):
            fen, played, sw, phase, ply = (ln.split("\t") + [""] * 5)[:5]
            bm, cp, mt = analyse(fen)
            fo.write(f"{fen}\t{played}\t{sw}\t{phase}\t{ply}\t{bm}\t{cp}\t{mt}\n")
            fo.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(lines)}", flush=True)
    print("DONE_REF")

if __name__ == "__main__":
    main()
