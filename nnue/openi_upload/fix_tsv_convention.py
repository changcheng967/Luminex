#!/usr/bin/env python3
"""One-time repair for a tsv whose cp column is side-to-move-relative instead
of the white-relative contract: negate cp on every black-to-move line. Reads
tsv or .xz, writes the same. The FEN's side-to-move is field 2 of 6.

    python3 fix_tsv_convention.py in.tsv.xz out.tsv.xz
"""
import sys
import lzma


def main():
    src, dst = sys.argv[1], sys.argv[2]
    fin = lzma.open(src, "rt") if src.endswith(".xz") else open(src, "r")
    fout = lzma.open(dst, "wt", preset=6) if dst.endswith(".xz") else open(dst, "w")
    n = nv = 0
    for line in fin:
        tab = line.find("\t")
        if tab <= 0:
            continue
        sp1 = line.find(" ")
        # FEN field 2 sits between the first and second space: "w" or "b"
        if sp1 > 0 and line[sp1 + 1] == "b":
            cp = int(line[tab + 1:])
            fout.write(f"{line[:tab]}\t{-cp}\n")
            nv += 1
        else:
            fout.write(line)
        n += 1
        if n % 50_000_000 == 0:
            print(f"{n:,} lines ({nv:,} negated)", flush=True)
    fin.close(); fout.close()
    print(f"DONE {n:,} lines, {nv:,} negated ({nv / max(1, n):.1%})")


if __name__ == "__main__":
    main()
