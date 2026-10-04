#!/usr/bin/env python3
"""Convert a float LXV3 net (V13 exporter output) to the quantized LX3Q net the
engine loads — quantize_i8.py's V13 counterpart:

    python quantize_v13q.py luminex_v13p2.nnue luminex_v13p2_q.nnue

Quantization (V13.md §1/§5): FT weights/bias int16 at FT_WSCALE=8192, l2/l3
weights int8 at the FIXED scale 8192/127 (keeps the dense chain in the x8192
requant domain), out weights int8 at s_out=127/max|w| (the engine folds it into
the final rescale), side projection float (its inputs are normalized scalars).
Every requant path is asserted.
"""
import sys, struct
import numpy as np

PT = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
KING_BUCKETS = [
    -1, -1, -1, -1, 31, 30, 29, 28, -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20, -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12, -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4, -1, -1, -1, -1, 3, 2, 1, 0,
]


def _orient(white_pov, sq, ksq):
    return ((7 if (ksq & 7) < 4 else 0) ^ (0 if white_pov else 56) ^ sq)


def parse_fen(fen):
    """Board-only parse -> (pieces [(sq, pt, is_white)], white_king, black_king)."""
    board = fen.split()[0]
    pieces, sq = [], 56
    for ch in board:
        if ch == '/':
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            pieces.append((sq, PT[ch.upper()], ch.isupper()))
            sq += 1
    wk = next(s for s, pt, w in pieces if pt == 5 and w)
    bk = next(s for s, pt, w in pieces if pt == 5 and not w)
    return pieces, wk, bk


def v13_state(pieces, wk, bk, pov, ftqi, ftbi, fac, spw, spb, L1):
    """(float h, quant qi) input vectors for one perspective — mirrors the engine."""
    white_pov = (pov == 0)
    ksq = wk if white_pov else bk
    acc = ftbi.copy()
    for sq, pt, isw in pieces:
        pidx = pt * 2 + (1 if isw != white_pov else 0)
        idx = _orient(white_pov, sq, ksq) + pidx * 64 + KING_BUCKETS[_orient(white_pov, ksq, ksq)] * 768
        plane = idx % 768
        b = min(idx // 768, fac - 1)
        acc = acc + ftqi[b * 768 + plane] + ftqi[fac * 768 + plane]
    counts = np.zeros(12)
    for _, pt, w in pieces:
        counts[pt * 2 + (0 if w else 1)] += 1
    side = list(counts / 8.0)
    for white in (True, False):
        files = [0] * 8
        for sq, pt, w in pieces:
            if pt == 0 and w == white:
                files[sq & 7] += 1
        side += files
    side += [ (wk & 7) / 7, (wk >> 3) / 7, (bk & 7) / 7, (bk >> 3) / 7, len(pieces) / 32 ]
    return acc, np.array(side)


def raw(f, n):
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def main(src, dst, fen_file=None):
    fens = []
    if fen_file:
        with open(fen_file) as fh:
            fens = [ln.strip() for ln in fh if ln.strip()]
    with open(src, 'rb') as f:
        assert f.read(4) == b'LXV3', f"{src} is not a float LXV3 net"
        ver, L1, L2, L3, fac = struct.unpack('5i', f.read(20))
        assert ver == 1 and L1 == 512 and L2 == 16 and L3 == 64, "unsupported dims"
        rows = fac * 768 + 768
        ft = raw(f, (rows + 1) * L1).reshape(rows + 1, L1)   # trailing pad row dropped
        ftb = raw(f, L1)
        spw = raw(f, 32 * 33).reshape(32, 33); spb = raw(f, 32)
        l2w = raw(f, L2 * (2 * L1 + 32)).reshape(L2, 2 * L1 + 32)
        l2b = raw(f, L2)
        l3w = raw(f, L3 * L2).reshape(L3, L2); l3b = raw(f, L3)
        ow = raw(f, L3); ob = raw(f, 1)
        linh = raw(f, 24576) if f.read(4) == b'LINH' else None

    FT, TAIL = 8192.0, 8192.0 / 128.0
    ftq = np.round(ft[:rows] * FT).astype(np.int32)
    assert np.abs(ftq).max() <= 32767, f"FT weight overflows int16: {np.abs(ftq).max()}"
    fbq = np.round(ftb * FT).astype(np.int32)
    assert np.abs(fbq).max() <= 32767, f"FT bias overflows int16: {np.abs(fbq).max()}"

    def qfix(w, name):
        q = np.round(w * TAIL)
        assert np.abs(q).max() <= 127, f"{name} exceeds the {TAIL:.0f} requant scale: {np.abs(q).max()}"
        return q.astype(np.int8)

    l2q = qfix(l2w, 'l2_w'); l3q = qfix(l3w, 'l3_w')
    s_out = 127.0 / max(float(np.abs(ow).max()), 1e-8)
    oq = np.round(ow * s_out).clip(-127, 127).astype(np.int8)
    l2bq = np.round(l2b * FT).astype(np.int64)
    l3bq = np.round(l3b * FT).astype(np.int64)
    obq = np.int64(round(float(ob[0]) * s_out * 128.0))

    # The >>6 requant floors activations and caps them at 127; which lanes sit
    # where is structurally correlated with the tail weights (real positions
    # saturate specific lanes), so the bias must be measured on real-position
    # accumulator ensembles — random synthetic lanes mislead by tens of cp.
    # Each layer is calibrated fed the already-calibrated quantized layer.
    def crelu_q8(v):
        return np.minimum(np.clip(v, 0, 8192).astype(np.int64) >> 6, 127)

    ftqi = np.round(ft[:rows] * FT)
    ftbi = np.round(ftb * FT)
    if fens:
        acc_f, acc_q = [], []
        for fen in fens:
            pieces, wk, bk = parse_fen(fen)
            stm_white = fen.split()[1] != 'b'
            accs, sides = [], []
            for pov in (0, 1):
                acc, side = v13_state(pieces, wk, bk, pov, ftqi, ftbi, fac, spw, spb, L1)
                accs.append(acc); sides.append(side)
            a_stm = accs[0] if stm_white else accs[1]
            a_nstm = accs[1] if stm_white else accs[0]
            sp_out = np.clip(spw @ sides[0] + spb, 0, 1)   # side is pov-invariant
            acc_f.append(np.concatenate([np.clip(a_stm / FT, 0, 1), np.clip(a_nstm / FT, 0, 1), sp_out]))
            acc_q.append(np.concatenate([crelu_q8(a_stm), crelu_q8(a_nstm),
                                         np.minimum(np.round(sp_out * 128), 127)]))
    else:
        rng = np.random.default_rng(20261003)
        acc_f, acc_q = [], []
        for t in range(256):
            lanes = (rng.uniform(0, 1, (2, L1)) if t % 2 else
                     np.clip(rng.normal(0.35, 0.5, (2, L1)), -1.6, 1.6))
            side = rng.uniform(0, 1, 33)
            sp_out = np.clip(spw @ side + spb, 0, 1)
            acc_f.append(np.concatenate([np.clip(lanes[0], 0, 1), np.clip(lanes[1], 0, 1), sp_out]))
            acc_q.append(np.concatenate([crelu_q8(np.round(lanes[0] * FT)), crelu_q8(np.round(lanes[1] * FT)),
                                         np.minimum(np.round(sp_out * 128), 127)]))
    pre2_gap = np.mean([ (l2q @ qi + bq)/FT - l2w @ h - l2b
                         for h, qi, bq in zip(acc_f, acc_q, [l2bq]*len(acc_f)) ], axis=0)
    l2bq = np.round(l2bq - pre2_gap * FT)
    q2_all = [crelu_q8(l2q @ qi + l2bq) for qi in acc_q]
    h2_all = [np.clip(l2w @ h + l2b, 0, 1) for h in acc_f]
    pre3_gap = np.mean([ (l3q @ q2 + bq)/FT - l3w @ h2 - l3b
                         for h2, q2, bq in zip(h2_all, q2_all, [l3bq]*len(h2_all)) ], axis=0)
    l3bq = np.round(l3bq - pre3_gap * FT)
    q3_all = [crelu_q8(l3q @ q2 + l3bq) for q2 in q2_all]
    h3_all = [np.clip(l3w @ h2 + l3b, 0, 1) for h2 in h2_all]
    out_gap = np.mean([ (oq @ q3 + obq)/(s_out*128) - ow @ h3 - ob[0]
                        for h3, q3 in zip(h3_all, q3_all) ])
    obq = np.int64(round(obq - out_gap * s_out * 128))

    with open(dst, 'wb') as f:
        f.write(b'LX3Q'); f.write(struct.pack('5i', 1, L1, L2, L3, fac))
        f.write(ftq.astype('<i2').tobytes())
        f.write(fbq.astype('<i2').tobytes())
        f.write(spw.astype('<f4').tobytes()); f.write(spb.astype('<f4').tobytes())
        f.write(l2q.tobytes()); f.write(l2bq.astype('<i4').tobytes())
        f.write(l3q.tobytes()); f.write(l3bq.astype('<i4').tobytes())
        f.write(oq.tobytes()); f.write(struct.pack('f', s_out)); f.write(struct.pack('i', int(obq)))
        if linh is not None:
            f.write(b'LINH'); f.write(linh.astype('<f4').tobytes())

    print(f"wrote {dst}: FT int16@8192 (max {np.abs(ftq).max()}), "
          f"l2/l3 int8@{TAIL:.0f} (max {int(max(np.abs(l2q).max(), np.abs(l3q).max()))}), "
          f"out int8@s_out={s_out:.2f}, calibrated on {len(acc_f)} "
          f"{'FEN' if fens else 'synthetic'} states")


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        print("usage: python quantize_v13q.py <float_lxv3.nnue> <out_lx3q.nnue> [calib.fens]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
