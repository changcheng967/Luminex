#!/usr/bin/env python3
"""Remap the current src/eval_fitted.h (610-wide, fit6 values) into a
1242-space c0_v5.txt: MG block 0..609 unchanged, 11 zeros appended; EG
block likewise. The per-file shield features (610..620 / 1231..1241)
start at 0, so the engine stays exactly fit6 until a fit is adopted.

Usage: python remap_c0_v5.py [src/eval_fitted.h] [c0_v5.txt]
"""
import re
import sys

OLD = 610
NEW = 621

src = sys.argv[1] if len(sys.argv) > 1 else "src/eval_fitted.h"
dst = sys.argv[2] if len(sys.argv) > 2 else "c0_v5.txt"

text = open(src).read()
blocks = re.findall(r"inline int FE_(?:MG|EG)\[\d+\] = \{(.*?)\};", text, re.S)
assert len(blocks) == 2, f"expected FE_MG + FE_EG blocks, got {len(blocks)}"


def parse(block):
    vals = [int(v) for v in re.findall(r"(-?\d+),", block)]
    assert len(vals) == OLD, f"block has {len(vals)} values, want {OLD}"
    return vals


mg, eg = parse(blocks[0]), parse(blocks[1])
with open(dst, "w") as fh:
    for j in range(NEW):
        v = mg[j] if j < OLD else 0
        fh.write(f"{j} {v}\n")
    for j in range(NEW):
        v = eg[j] if j < OLD else 0
        fh.write(f"{NEW + j} {v}\n")
print(f"wrote {dst}: {2*NEW} rows (fit6 values, shield slots zero)")
