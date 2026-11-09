#!/usr/bin/env python3
"""Convert a float net (output by the trainers) to the int8 net the engine
loads. Pure numpy, runs on CPU:

    python quantize_i8.py luminex_v2.nnue luminex_v6_i8.nnue
    python quantize_i8.py net.nnue net_i8.nnue --calib 'held_out.*.lcr'
    python quantize_i8.py net.nnue net_i8.nnue --per-neuron --calib 'held_out.*.lcr'

A float LNN1 net (one dense tail) becomes LNI8; an LNB1 net (one tail per
output bucket, NNUE_BUCKETS > 1) becomes LNB8, each bucket's tail quantized
with its own scales. Then drop the result onto the engine and bench it.

Each tail weight matrix gets one scale, set by its largest weight, so a typical
weight gets only a few int8 steps, and the rounding shifts each unit's
pre-activation on average: the engine's eval sat 12-20 cp from the trained
net's. --calib corrects the tail biases for that shift, layer by layer along the
engine's own int8 path (src/nnue.cpp: int16 feature transformer at x8192, 8-bit
activations lrintf(clamp(x, 0, 1)^2 * 127), int8 weights, float bias and scale).
Each unit's bias moves by the mean difference between the float net's
pre-activation and the int8 path's, over the calibration positions of its
bucket where the float unit is in its active window (0, 1), with the layers
before it already corrected; the output bias then takes each bucket's mean
eval offset. Give positions the net was not trained on (LCR1 records, e.g. a
held-out month). Without --calib the output is the same bytes as before.

--per-neuron gives every output neuron of each tail matrix its own scale, set by
that neuron's largest weight (LNIR / LNBR: the scale is a tensor of one float per
row); the engine applies it at no extra cost. On the shipped net and on
gen768r_03 this more than halves the rounding error.
"""
import argparse
import glob
import struct

import numpy as np

FT_WSCALE = 8192.0   # src/nnue.cpp: int16 feature transformer lanes
OUT_SCALE = 300.0    # src/nnue.cpp: cp = llround(out * 300)
KB = np.maximum(np.array(   # king bucket of the oriented king square (resident_train.py KB)
    [-1, -1, -1, -1, 31, 30, 29, 28, -1, -1, -1, -1, 27, 26, 25, 24,
     -1, -1, -1, -1, 23, 22, 21, 20, -1, -1, -1, -1, 19, 18, 17, 16,
     -1, -1, -1, -1, 15, 14, 13, 12, -1, -1, -1, -1, 11, 10, 9, 8,
     -1, -1, -1, -1, 7, 6, 5, 4, -1, -1, -1, -1, 3, 2, 1, 0]), 0)


def rt(f):
    (n,) = struct.unpack('i', f.read(4))
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def q(w, per_row=False):
    """int8 weights and their scale: one per matrix, or one per output row."""
    if per_row:
        s = 127.0 / np.maximum(np.abs(w).max(1).astype(np.float64), 1e-8)
        return np.round(w * s[:, None]).clip(-127, 127).astype(np.int8), s.astype(np.float32)
    s = 127.0 / max(float(np.abs(w).max()), 1e-8)
    return np.round(w * s).clip(-127, 127).astype(np.int8), np.float32(s)


# ---------------------------------------------------------------------------
# Calibration: the engine's int8 evaluation, emulated
# ---------------------------------------------------------------------------
def lcr1_features(rec, ni):
    """LCR1 records (uint8 [n, 32]) -> (white idx, black idx, white to move,
    pieces). resident_train.expand in numpy: square XOR 7 when that side's
    king is on files a-d, XOR 56 more for black's view; plane = piece type * 2,
    +1 for the other side's pieces; king bucket from the oriented king square.
    Empty slots get index ni (a zero row)."""
    n = len(rec)
    occ = rec[:, 0:8].copy().view(np.uint64).reshape(n)
    nib = rec[:, 8:24].astype(np.int64)
    codes = np.stack((nib & 15, nib >> 4), axis=2).reshape(n, 32)
    bits = ((occ[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(bool)
    cnt = bits.sum(1)
    sqs = np.sort(np.where(bits, np.arange(64), 64), axis=1)[:, :32]
    valid = np.arange(32)[None, :] < cnt[:, None]
    rows = np.arange(n)
    wk = sqs[rows, np.argmax((codes == 5) & valid, axis=1)]
    bk = sqs[rows, np.argmax((codes == 11) & valid, axis=1)]
    xw = np.where((wk & 7) < 4, 7, 0)
    xb = np.where((bk & 7) < 4, 7, 0) ^ 56
    pt, blk = codes % 6, codes // 6
    sq = np.minimum(sqs, 63)
    wf = (xw[:, None] ^ sq) + (pt * 2 + blk) * 64 + (KB[xw ^ wk] * 768)[:, None]
    bf = (xb[:, None] ^ sq) + (pt * 2 + 1 - blk) * 64 + (KB[xb ^ bk] * 768)[:, None]
    return np.where(valid, wf, ni), np.where(valid, bf, ni), (rec[:, 26] & 1).astype(bool), cnt


def first_layer(ft_t, ft_b, wf, bf, stm, int16):
    """Both perspectives' feature-transformer sums, side to move first, then
    clamp(x, 0, 1)^2: float32 for the float net; for the engine (int16) the
    weights and bias rounded to 1/8192 as src/nnue.cpp loads them, the lanes
    clamped to int16 and the activations rounded to bytes (0..127)."""
    out = []
    for i in range(0, len(wf), 2048):
        a = ft_b + ft_t[wf[i:i + 2048]].sum(1, dtype=ft_t.dtype)
        b = ft_b + ft_t[bf[i:i + 2048]].sum(1, dtype=ft_t.dtype)
        s = stm[i:i + 2048, None]
        x = np.concatenate([np.where(s, a, b), np.where(s, b, a)], 1)
        if int16:
            x = np.clip(x, -32768, 32767).astype(np.float32) * np.float32(1.0 / FT_WSCALE)
            c = np.clip(x, 0, 1)
            out.append(np.rint((c * c) * np.float32(127)).astype(np.uint8))
        else:
            c = np.clip(x.astype(np.float32), 0, 1)
            out.append(c * c)
    return np.concatenate(out) if out else np.zeros((0, 2 * ft_t.shape[1]), np.float32)


def dense_q(x8, w8, b, s):
    """The engine's int8 layer: exact integer dot of bytes and int8 weights,
    then bias + dot * (1 / (scale * 127)) in float32 (fused multiply-add, as
    GCC contracts it at -O3 on FMA hardware); s is one scale or one per output."""
    d = x8.astype(np.float64) @ w8.astype(np.float64).T          # exact: |d| < 2^53
    ics = np.float32(1.0) / (np.asarray(s, dtype=np.float32) * np.float32(127.0))
    return (d * ics.astype(np.float64) + np.asarray(b, np.float64)).astype(np.float32)


def act_bytes(pre):
    c = np.clip(pre, 0, 1)
    return np.rint((c * c) * np.float32(127)).astype(np.uint8)


def screlu(pre):
    c = np.clip(pre, 0, 1)
    return c * c


def correct_tail(h_f, x8, tail, qt):
    """Float path and int8 path through one bucket's tail on its calibration
    positions; returns the corrected biases and the mean eval offset (float -
    int8, cp) before and after."""
    l2_w, l2_b, l3_w, l3_b, out_w, out_b = tail
    (w2, s2), (w3, s3), (wo, so) = qt
    p2f = h_f @ l2_w.T + l2_b
    p3f = screlu(p2f) @ l3_w.T + l3_b
    ovf = (screlu(p3f) @ out_w.T).ravel() + out_b

    def shift(pf, pq):   # mean (float - int8) where the float unit is in its window, else over all
        act = (pf > 0) & (pf < 1)
        n_act = act.sum(0)
        part = np.where(act, pf - pq, 0).sum(0) / np.maximum(n_act, 1)
        return np.where(n_act >= 50, part, (pf - pq).mean(0)).astype(np.float32)

    def run(b2, b3, bo):
        p2q = dense_q(x8, w2, b2, s2)
        p3q = dense_q(act_bytes(p2q), w3, b3, s3)
        return p2q, p3q, dense_q(act_bytes(p3q), wo, [bo], so).ravel()

    ov0 = run(l2_b, l3_b, out_b)[2]
    b2 = l2_b + shift(p2f, dense_q(x8, w2, l2_b, s2))
    p2q = dense_q(x8, w2, b2, s2)
    b3 = l3_b + shift(p3f, dense_q(act_bytes(p2q), w3, l3_b, s3))
    ovq = run(b2, b3, out_b)[2]
    bo = np.float32(out_b + (ovf - ovq).mean())
    ov1 = run(b2, b3, bo)[2]
    return b2, b3, bo, float((ovf - ov0).mean() * OUT_SCALE), float((ovf - ov1).mean() * OUT_SCALE)


def calibrate(ft_w, ft_b, ni, tails, pattern, cmax, per_row=False):
    files = sorted(f for p in pattern.split(",") for f in glob.glob(p.strip()))
    if not files:
        raise SystemExit(f"--calib {pattern}: no files")
    sizes = [np.memmap(f, dtype=np.uint8, mode="r").size // 32 for f in files]
    stride = max(1, sum(sizes) // cmax)
    rec = np.concatenate([np.memmap(f, dtype=np.uint8, mode="r").reshape(-1, 32)[::stride] for f in files])[:cmax]
    wf, bf, stm, cnt = lcr1_features(np.ascontiguousarray(rec), ni)
    nb = len(tails)
    bucket = np.clip((cnt - 1) * nb // 32, 0, nb - 1)   # src/nnue.cpp bucket_of
    ft_t = np.vstack([ft_w.T, np.zeros((1, ft_w.shape[0]), np.float32)])
    away = lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5)   # std::llround / std::lround
    ftq = np.vstack([np.clip(away(ft_w.T.astype(np.float64) * FT_WSCALE), -32767, 32767),
                     np.zeros((1, ft_w.shape[0]))]).astype(np.int32)
    bq = away(ft_b.astype(np.float64) * FT_WSCALE).astype(np.int32)
    print(f"calibration: {len(rec):,} positions (every {stride}th) from {len(files)} file(s)")
    out = []
    for b, tail in enumerate(tails):
        sel = bucket == b
        n = int(sel.sum())
        if n < 200:
            print(f"  bucket {b}: {n} positions, too few: biases left as they are")
            out.append(tail)
            continue
        h_f = first_layer(ft_t, ft_b, wf[sel], bf[sel], stm[sel], False)
        x8 = first_layer(ftq, bq, wf[sel], bf[sel], stm[sel], True)
        l2_w, l2_b, l3_w, l3_b, out_w, out_b = tail
        b2, b3, bo, off0, off1 = correct_tail(h_f, x8, tail, (q(l2_w, per_row), q(l3_w, per_row), q(out_w, per_row)))
        print(f"  bucket {b}: {n:,} positions, eval offset (float - int8) {off0:+.1f} cp -> {off1:+.1f} cp; "
              f"mean bias shift L2 {np.abs(b2 - l2_b).mean():.4f}, L3 {np.abs(b3 - l3_b).mean():.4f}, "
              f"out {abs(float(bo) - float(out_b)) * OUT_SCALE:.1f} cp")
        out.append((l2_w, b2, l3_w, b3, out_w, bo))
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert a float LNN1/LNB1 net to the engine's int8 LNI8/LNB8 net.")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--calib", help="LCR1 files (globs, comma-separated) of held-out positions: correct the "
                                    "tail biases for rounding along the engine's int8 path")
    ap.add_argument("--calib-max", type=int, default=200000, help="positions to calibrate on [200000]")
    ap.add_argument("--per-neuron", action="store_true", help="one scale per output neuron (LNIR / LNBR)")
    a = ap.parse_args()
    with open(a.src, 'rb') as f:
        magic = f.read(4)
        assert magic in (b'LNN1', b'LNB1'), f"{a.src} is not a float LNN1/LNB1 net"
        L1q, L2, L3, NI = struct.unpack('iiii', f.read(16))
        nb = struct.unpack('i', f.read(4))[0] if magic == b'LNB1' else 1
        ft_w = rt(f).reshape(L1q, NI); ft_b = rt(f)
        tails = []
        for _ in range(nb):
            l2_w = rt(f).reshape(L2, 2 * L1q); l2_b = rt(f)
            l3_w = rt(f).reshape(L3, L2);       l3_b = rt(f)
            out_w = rt(f).reshape(1, L3);       out_b = rt(f)[0]
            tails.append((l2_w, l2_b, l3_w, l3_b, out_w, out_b))
        # DOSL linear head: pass through verbatim (independent of int8 conversion)
        linh = None
        tag = f.read(4)
        if tag == b'LINH':
            (n,) = struct.unpack('i', f.read(4))
            linh = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
            assert n == NI
    if a.calib:
        tails = calibrate(ft_w, ft_b, NI, tails, a.calib, a.calib_max, a.per_neuron)
    with open(a.dst, 'wb') as f:
        magic = (b'LNBR' if nb > 1 else b'LNIR') if a.per_neuron else (b'LNB8' if nb > 1 else b'LNI8')
        f.write(magic); f.write(struct.pack('iiii', L1q, L2, L3, NI))
        if nb > 1:
            f.write(struct.pack('i', nb))
        for arr in [ft_w, ft_b]:
            f.write(struct.pack('i', arr.size)); f.write(arr.astype(np.float32).tobytes())
        for l2_w, l2_b, l3_w, l3_b, out_w, out_b in tails:
            pr = a.per_neuron
            l2w, s2 = q(l2_w, pr); l3w, s3 = q(l3_w, pr); outw, so = q(out_w, pr)
            for w8, b, s in [(l2w, l2_b, s2), (l3w, l3_b, s3), (outw, np.array([out_b], np.float32), so)]:
                f.write(struct.pack('i', w8.size)); f.write(w8.tobytes())
                f.write(struct.pack('i', b.size)); f.write(b.astype(np.float32).tobytes())
                if pr:
                    f.write(struct.pack('i', s.size)); f.write(s.astype(np.float32).tobytes())
                else:
                    f.write(struct.pack('f', float(s)))
        if linh is not None:
            f.write(b'LINH'); f.write(struct.pack('i', linh.size)); f.write(linh.astype(np.float32).tobytes())
    kind = f"{nb} output buckets" if nb > 1 else "one tail"
    print(f"wrote {a.dst}: L1={L1q} L2={L2} L3={L3}, {kind}  (FT float, L2/L3/out int8)"
          f"{' +lin' if linh is not None else ''}{', per-neuron scales' if a.per_neuron else ''}"
          f"{', biases corrected' if a.calib else ''}")


if __name__ == "__main__":
    main()
