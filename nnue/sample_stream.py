#!/usr/bin/env python3
"""Sample featurizer --stream records from stdin into an npz.

stdin: 136-byte records = int16 w[32] | int16 b[32] | float32 stm | float32 target
usage: sample_stream.py OUT.npz STRIDE [CAP]
"""
import sys
import numpy as np

out_path, stride = sys.argv[1], int(sys.argv[2])
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 1_300_000

CHUNK = 100_000  # records per read
REC = 136
dt = np.dtype([("w", "<i2", 32), ("b", "<i2", 32), ("s", "<f4"), ("t", "<f4")])
assert dt.itemsize == REC

ws, bs, ss, ts = [], [], [], []
n_in = n_kept = 0
while True:
    buf = sys.stdin.buffer.read(CHUNK * REC)
    if not buf:
        break
    recs = np.frombuffer(buf[: (len(buf) // REC) * REC], dtype=dt)
    n_in += len(recs)
    pick = recs[::stride][: max(0, cap - n_kept)]
    if len(pick):
        ws.append(pick["w"].astype(np.int32))
        bs.append(pick["b"].astype(np.int32))
        ss.append(pick["s"])
        ts.append(pick["t"])
        n_kept += len(pick)
    if n_kept >= cap:
        break

w = np.concatenate(ws); b = np.concatenate(bs)
s = np.concatenate(ss); t = np.concatenate(ts)
np.savez_compressed(out_path, w=w, b=b, s=s, t=t)
print(f"read {n_in:,} records, kept {len(t):,} (stride {stride})")
print(f"target cp: mean|t|={np.abs(t).mean():.1f}  p10={np.percentile(t,10):.0f} "
      f"p50={np.percentile(t,50):.0f} p90={np.percentile(t,90):.0f}")
print(f"stm white frac: {s.mean():.3f}")
