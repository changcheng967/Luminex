#!/usr/bin/env python3
"""Generate gauge anchors for solve_ridge.py from the solve .bin.

One rank1 anchor per piece per phase pinning the EFFECTIVE value
(MAT_pt + mean PST_pt) to the CURRENT engine's effective value, with
weight = A_jj of the MAT feature (equal-strength prior: cuts gauge
movement roughly in half even when the data screams).
"""
import sys
import numpy as np

NPHASE, NS, PST, MAT = 621, 1242, 0, 384
PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king"]

prefix = sys.argv[1]
c0_path = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "gauge.anchors"

with open(prefix + ".bin", "rb") as fh:
    n = int(np.fromfile(fh, dtype=np.int64, count=1)[0])
    A_diag = np.fromfile(fh, dtype=np.float64, count=NS * NS).reshape(NS, NS).diagonal().copy()
c0 = np.array([float(ln.split()[1]) for ln in open(c0_path) if ln.strip()])

lines = []
for pt in range(6):
    for ph_off, ph in [(0, "mg"), (NPHASE, "eg")]:
        mat_j = MAT + pt + ph_off
        cells = [PST + pt * 64 + s + ph_off for s in range(64)]
        target = c0[mat_j] + c0[cells].mean()
        w = float(A_diag[mat_j])
        tok = [f"{j} {1.0/64:.8f}" for j in cells]
        lines.append(
            f"# {PIECES[pt]} {ph}: pin effective to {target:.1f}, w={w:.3g}\n"
            f"rank1 {w:.6e} {target:.6f} {mat_j} 1.0 " + " ".join(tok)
        )
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"wrote {out}: 12 gauge anchors (6 pieces x mg/eg)")
