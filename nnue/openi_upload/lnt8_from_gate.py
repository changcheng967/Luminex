"""Write a gen768t (LNT8) network from a V5 gate arm's kept files, for the engine before any QAT net exists.

ARM_i8.nnue  the arm's LNR8 net (quantize_i8.py --rows): the float feature layer and the int8 tails
ARM.rel      the arm's relation rows (the gate's g4_export.py): "LXR1", six uint32 (rows, T, set, pawn-pair mode,
             scale 8192, 0), then int16[rows][T], the trained rows on the x8192 grid
OUT.nnue     the LNT8 file (the layout: src/nnue.cpp's LNT8 reader)

The piece-square table and its bias are rounded onto the engine's x2048 grid exactly as the engine rounds an
LNR8 net at load (x 2048, then half away from zero), so the LNT8 net's piece-square lanes are the ones the engine
computes for the LNR8 net. The relation rows are rounded to int8 steps of 1/128 (RS = 4 on x2048), ties to even
as the gate's row-rounding check measured them (under 0.1% of loss on every trained table), and clipped to +-127.
The tails, and any section after them, are LNR8's bytes unchanged.
--zero-rel writes the relation table as zeros: that LNT8 net must evaluate and search exactly as the LNR8 one.

usage: lnt8_from_gate.py ARM_i8.nnue ARM.rel OUT.nnue [--zero-rel]"""
import struct
import sys

import numpy as np

FS, RS, LAYOUT, ROWS = 11, 4, 1, 55960

args = [a for a in sys.argv[1:] if not a.startswith("--")]
zero_rel = "--zero-rel" in sys.argv
if len(args) != 3:
    raise SystemExit(__doc__)
src, rel_path, out = args

raw = open(src, "rb").read()
if raw[:4] != b"LNR8":
    raise SystemExit(f"FAIL: {src} is not an LNR8 net")
L1, L2, L3, NI, NB = struct.unpack("<5i", raw[4:24])
off = 24
(n,) = struct.unpack("<i", raw[off:off + 4])
if n != L1 * NI:
    raise SystemExit(f"FAIL: {src}: feature table of {n} values, expected {L1 * NI}")
ft = np.frombuffer(raw, dtype="<f4", count=n, offset=off + 4).reshape(L1, NI)   # lane-major, as LNR8 stores it
off += 4 + 4 * n
(n,) = struct.unpack("<i", raw[off:off + 4])
if n != L1:
    raise SystemExit(f"FAIL: {src}: bias of {n} values, expected {L1}")
bias = np.frombuffer(raw, dtype="<f4", count=n, offset=off + 4)
off += 4 + 4 * n
tails = raw[off:]


def half_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


scale = np.float32(1 << FS)
q = half_away((ft.astype(np.float32) * scale).astype(np.float64))
clipped = int((np.abs(q) > 32767).sum())
q = np.clip(q, -32767, 32767).astype("<i2").T.copy()                             # feature-major [NI][L1]
qb = half_away((bias.astype(np.float32) * scale).astype(np.float64)).astype("<i4")
beyond = int((np.abs(ft) > 3.99).sum())

rel = open(rel_path, "rb").read()
if rel[:4] != b"LXR1":
    raise SystemExit(f"FAIL: {rel_path} is not a row file")
rows, T, rset, ppm, rscale, _ = struct.unpack("<6I", rel[4:28])
if rows != ROWS or rset != 0:
    raise SystemExit(f"FAIL: {rel_path}: {rows} rows, set {rset}; LNT8 layout 1 is the full set of {ROWS} rows")
if T % 64 or T > L1:
    raise SystemExit(f"FAIL: {rel_path}: T = {T}, a multiple of 64 up to L1 = {L1}")
w = np.frombuffer(rel, dtype="<i2", offset=28).reshape(rows, T).astype(np.float64)
step = rscale / float(1 << (FS - RS))   # trained units per int8 step: 8192 / 128 = 64
r8 = np.rint(w / step)
rel_clipped = int((np.abs(r8) > 127).sum())
r8 = np.clip(r8, -127, 127).astype(np.int8)
if zero_rel:
    r8[:] = 0

with open(out, "wb") as f:
    f.write(b"LNT8")
    f.write(struct.pack("<5i", L1, L2, L3, NI, NB))
    f.write(struct.pack("<6i", T, rows, FS, RS, ppm, LAYOUT))
    f.write(struct.pack("<i", NI * L1))
    f.write(q.tobytes())
    f.write(struct.pack("<i", L1))
    f.write(qb.tobytes())
    f.write(struct.pack("<i", rows * T))
    f.write(r8.tobytes())
    f.write(tails)
print(f"{out}: L1 {L1}, L2 {L2}, L3 {L3}, {NB} buckets; T {T}, {rows} rows, pawn pairs {'all' if ppm else 'in contact'}, "
      f"grid x{1 << FS}, step 1/{1 << (FS - RS)}{' (relation table zeroed)' if zero_rel else ''}; "
      f"piece-square weights beyond 3.99: {beyond}, clipped to int16: {clipped}; relation values clipped to int8: {rel_clipped}")
