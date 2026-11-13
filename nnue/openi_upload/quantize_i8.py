#!/usr/bin/env python3
"""Convert a float net (output by the C500 trainer) to the int8 net the engine
loads. Pure numpy, runs on CPU in seconds. Run LOCALLY after downloading the
trainer's output from the C500 job:

    python quantize_i8.py luminex_v2.nnue luminex_v6_i8.nnue

A float LNN1 net (one dense tail) becomes LNI8; an LNB1 net (one tail per
output bucket, NNUE_BUCKETS > 1) becomes LNB8, each bucket's tail quantized
with its own scales. Then drop the result onto the engine and bench it.

    python quantize_i8.py IN.nnue OUT.nnue --ft8 7       int8 feature weights too (LNF8)
    python quantize_i8.py IN.nnue OUT.nnue --ft-grid 7   float FT rounded to that grid (LNB8)

--ft8 SHIFT stores each feature weight w as q = round(w * 2^(13 - SHIFT)) clipped to +-127,
standing for q << SHIFT on the engine's x8192 domain (SHIFT 7: steps of 1/64, |w| <= 1.98),
feature-major as the engine indexes it. --ft-grid SHIFT writes the same rounded weights as a
float LNB8 net, which the engine turns into the same int16 lanes: the two nets evaluate bit for
bit alike, so they measure what the int8 table alone changes.
"""
import sys, struct
import numpy as np


def rt(f):
    (n,) = struct.unpack('i', f.read(4))
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def q(w):
    s = 127.0 / max(float(np.abs(w).max()), 1e-8)
    return np.round(w * s).clip(-127, 127).astype(np.int8), np.float32(s)


def main(src, dst, ft8=None, grid=None):
    with open(src, 'rb') as f:
        magic = f.read(4)
        assert magic in (b'LNN1', b'LNB1'), f"{src} is not a float LNN1/LNB1 net"
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
    shift = ft8 if ft8 is not None else grid
    note = ""
    if shift is not None:
        k = float(2 ** (13 - shift))   # int8 units per 1.0 of weight
        qft = np.round(ft_w * k)
        clipped = int((np.abs(qft) > 127).sum())
        qft = qft.clip(-127, 127)
        err = qft / k - ft_w
        note = (f", FT on the 1/{int(k)} grid ({clipped} of {ft_w.size:,} clipped, rms error "
                f"{float(np.sqrt((err ** 2).mean())):.5f} of rms weight {float(np.sqrt((ft_w ** 2).mean())):.5f})")
        if grid is not None:
            ft_w = (qft / k).astype(np.float32)
    with open(dst, 'wb') as f:
        if ft8 is not None:
            f.write(b'LNF8'); f.write(struct.pack('iiii', L1q, L2, L3, NI)); f.write(struct.pack('i', nb))
            f.write(struct.pack('i', ft8))
            t8 = np.ascontiguousarray(qft.astype(np.int8).T)   # [NI][L1]: one row per feature
            f.write(struct.pack('i', t8.size)); f.write(t8.tobytes())
            f.write(struct.pack('i', ft_b.size)); f.write(ft_b.astype(np.float32).tobytes())
        else:
            f.write(b'LNB8' if nb > 1 else b'LNI8'); f.write(struct.pack('iiii', L1q, L2, L3, NI))
            if nb > 1:
                f.write(struct.pack('i', nb))
            for arr in [ft_w, ft_b]:
                f.write(struct.pack('i', arr.size)); f.write(arr.astype(np.float32).tobytes())
        for l2_w, l2_b, l3_w, l3_b, out_w, out_b in tails:
            l2w, s2 = q(l2_w); l3w, s3 = q(l3_w); outw, so = q(out_w)
            for w8, b, s in [(l2w, l2_b, s2), (l3w, l3_b, s3), (outw, np.array([out_b], np.float32), so)]:
                f.write(struct.pack('i', w8.size)); f.write(w8.tobytes())
                f.write(struct.pack('i', b.size)); f.write(b.astype(np.float32).tobytes())
                f.write(struct.pack('f', float(s)))
        if linh is not None:
            f.write(b'LINH'); f.write(struct.pack('i', linh.size)); f.write(linh.astype(np.float32).tobytes())
    kind = f"{nb} output buckets" if nb > 1 else "one tail"
    ft_kind = "FT int8" if ft8 is not None else "FT float"
    print(f"wrote {dst}: L1={L1q} L2={L2} L3={L3}, {kind}  ({ft_kind}, L2/L3/out int8){note}"
          f"{' +lin' if linh is not None else ''}")


if __name__ == "__main__":
    args = sys.argv[1:]
    ft8 = grid = None
    if len(args) == 4 and args[2] in ("--ft8", "--ft-grid"):
        if args[2] == "--ft8":
            ft8 = int(args[3])
        else:
            grid = int(args[3])
        args = args[:2]
    if len(args) != 2 or (ft8 is not None and not 0 <= ft8 <= 8) or (grid is not None and not 0 <= grid <= 8):
        print("usage: python quantize_i8.py <float LNN1/LNB1 net> <out net> [--ft8 SHIFT | --ft-grid SHIFT]")
        sys.exit(1)
    main(args[0], args[1], ft8, grid)
