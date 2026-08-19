#!/usr/bin/env python3
"""Fixed-depth showdown on pre-swing loss positions.

For each FEN: both engines search at equal depth; record best move and cp.
Question answered: does Luminex (with time/depth) AVOID the game-losing
move? Does the strong engine? Classifies each loss as time-pressure blunder
(engine avoids it now) vs true search/eval gap (engine repeats it).
"""
import subprocess
import sys
import threading

DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 12

class Engine:
    def __init__(self, path, name):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.name = name
        self.send("uci")
        self.waitfor("uciok")

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def waitfor(self, token, timeout=120):
        # crude line poll
        import time
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            ln = self.p.stdout.readline()
            if not ln:
                break
            lines.append(ln.strip())
            if token in ln:
                return lines
        return lines

    def analyse(self, fen):
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send(f"go depth {DEPTH}")
        cp = None
        best = None
        mate = None
        import time
        deadline = time.time() + 60
        while time.time() < deadline:
            ln = self.p.stdout.readline()
            if not ln:
                break
            ln = ln.strip()
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
    fens_path, out_path = sys.argv[1], sys.argv[2]
    lu = Engine("/hyperai/home/Luminex/build_release/luminex", "luminex")
    st = Engine("/hyperai/home/stash-20.0-linux-x86_64-bmi2", "stashv20")
    rows = []
    with open(fens_path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    for idx, ln in enumerate(lines):
        fen, played, sw, phase, ply = (ln.split("\t") + [""] * 5)[:5]
        bm_lu, cp_lu, mt_lu = lu.analyse(fen)
        bm_st, cp_st, mt_st = st.analyse(fen)
        rows.append((fen, played, sw, phase, ply, bm_lu, cp_lu, mt_lu, bm_st, cp_st, mt_st))
        if (idx + 1) % 25 == 0:
            print(f"{idx + 1}/{len(lines)} done", flush=True)
    with open(out_path, "w") as fo:
        fo.write("fen\tplayed\tsw\tphase\tply\tlu_bm\tlu_cp\tlu_mate\tst_bm\tst_cp\tst_mate\n")
        for r in rows:
            fo.write("\t".join(str(x) for x in r) + "\n")
    n = len(rows)
    lu_repeat = sum(1 for r in rows if r[5] == r[1])
    st_repeat = sum(1 for r in rows if r[8] == r[1])
    agree = sum(1 for r in rows if r[5] == r[8])
    print(f"n={n}  luminex repeats game blunder: {lu_repeat} ({100*lu_repeat/max(n,1):.0f}%)  "
          f"stash repeats: {st_repeat} ({100*st_repeat/max(n,1):.0f}%)  engines agree: {agree}")

if __name__ == "__main__":
    main()
