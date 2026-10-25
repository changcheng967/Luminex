#!/usr/bin/env python3
"""Split a fen<tab>cp tsv (or .xz) into train and validation files by FEN hash.

The SF-relabel data repeats positions with visit multiplicity, so a random
LINE split leaks repeats into both sides and flatters wider nets (the 105
epoch run's train-MAE fell while real play collapsed). Splitting on a hash
of the FEN puts every repeat of a position on the same side.

Deterministic: the same FEN lands in the same bucket regardless of dump
order, so a future re-dump produces a compatible split.

    python3 split_val.py sf_train_full.tsv.xz out_train.tsv.xz out_val.tsv.xz [val_frac=0.02]
"""
import sys
import zlib


def frac_of(fen: str) -> float:
    # crc32 is fast and stable; the field before the first tab is the FEN
    return (zlib.crc32(fen.encode()) & 0xFFFFFFFF) / 4294967296.0


def main():
    src, out_train, out_val = sys.argv[1], sys.argv[2], sys.argv[3]
    val_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    import lzma
    fin = lzma.open(src, "rt") if src.endswith(".xz") else open(src, "r")
    ft = lzma.open(out_train, "wt", preset=6) if out_train.endswith(".xz") else open(out_train, "w")
    fv = lzma.open(out_val, "wt", preset=6) if out_val.endswith(".xz") else open(out_val, "w")
    n_tr = n_val = 0
    for line in fin:
        tab = line.find("\t")
        if tab <= 0:
            continue
        if frac_of(line[:tab]) < val_frac:
            fv.write(line); n_val += 1
        else:
            ft.write(line); n_tr += 1
    fin.close(); ft.close(); fv.close()
    print(f"train {n_tr:,}  val {n_val:,}  ({n_val / max(1, n_tr + n_val):.4f})")


if __name__ == "__main__":
    main()
