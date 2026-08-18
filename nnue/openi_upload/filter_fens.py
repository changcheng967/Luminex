#!/usr/bin/env python3
"""Filter fens_all.txt to structurally-valid FEN lines.

The 8-thread featurizer dump interleaved occasional partial lines (two FENs
overlapped mid-write), producing boards like ".../1R61r6/..." (13 ranks).
Those crash pos.set() (OOB walk over a >8-rank board) -> SIGSEGV mid-solve.

A line is kept iff field 1 (fen) has exactly 8 '/'-separated ranks, each
summing to exactly 8 squares (digits 1-8 = runs, [pnbrqkPNBRQK] = singles).
Prints reject stats to stderr.
"""
import sys

VALID = set("pnbrqkPNBRQK")

def board_ok(b):
    ranks = b.split("/")
    if len(ranks) != 8:
        return False
    for rk in ranks:
        n = 0
        for ch in rk:
            if ch in VALID:
                n += 1
            elif "1" <= ch <= "8":
                n += ord(ch) - 48
            else:
                return False
        if n != 8:
            return False
    return True

def main():
    src, dst = sys.argv[1], sys.argv[2]
    kept = rej = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for raw in fi:
            line = raw.rstrip(b"\r\n")
            fen = line.split(b"\t", 1)[0]
            try:
                parts = fen.split(b" ")
                ok = len(parts) >= 4 and board_ok(parts[0].decode("ascii"))
            except UnicodeDecodeError:
                ok = False
            if ok:
                fo.write(line + b"\n")
                kept += 1
            else:
                rej += 1
    print(f"kept {kept} rejected {rej}", file=sys.stderr)

if __name__ == "__main__":
    main()
