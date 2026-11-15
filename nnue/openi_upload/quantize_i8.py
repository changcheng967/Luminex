#!/usr/bin/env python3
"""Convert a float net (output by the C500 trainer) to the int8 net the engine
loads. Pure numpy, runs on CPU in seconds. Run LOCALLY after downloading the
trainer's output from the C500 job:

    python quantize_i8.py luminex_v2.nnue luminex_v6_i8.nnue [--rows]

A float LNN1 net (one dense tail) becomes LNI8; an LNB1 net (one tail per
output bucket, NNUE_BUCKETS > 1) becomes LNB8, each bucket's tail quantized
with its own scales. Then drop the result onto the engine and bench it.

--rows writes LNR8 instead: LNB8 with one int8 scale per row (output neuron)
of L2 and L3, the scheme the trainer's NNUE_QAT_ROWS=1 trains for. Its scales
are computed in float32 as the trainer's fake quantization computes them, so
the engine's weights are the ones QAT saw.
"""
import sys, struct
import numpy as np


def rt(f):
    (n,) = struct.unpack('i', f.read(4))
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def q(w):
    s = 127.0 / max(float(np.abs(w).max()), 1e-8)
    return np.round(w * s).clip(-127, 127).astype(np.int8), np.float32(s)


def q_rows(w):
    """One scale per row: s = 127 / max|row| and round(w x s), in float32 like the trainer's fq_w."""
    w = np.asarray(w, dtype=np.float32)
    s = (np.float32(127.0) / np.maximum(np.abs(w).max(axis=1, keepdims=True), np.float32(1e-8))).astype(np.float32)
    return np.round(w * s).clip(-127, 127).astype(np.int8), s.reshape(-1)


def main(src, dst, rows=False):
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
    with open(dst, 'wb') as f:
        f.write(b'LNR8' if rows else b'LNB8' if nb > 1 else b'LNI8'); f.write(struct.pack('iiii', L1q, L2, L3, NI))
        if nb > 1 or rows:
            f.write(struct.pack('i', nb))
        for arr in [ft_w, ft_b]:
            f.write(struct.pack('i', arr.size)); f.write(arr.astype(np.float32).tobytes())
        for l2_w, l2_b, l3_w, l3_b, out_w, out_b in tails:
            if rows:
                l2w, s2 = q_rows(l2_w); l3w, s3 = q_rows(l3_w)
            else:
                l2w, s2 = q(l2_w); l3w, s3 = q(l3_w)
            outw, so = q(out_w)   # one row: the same scale either way
            for w8, b, s in [(l2w, l2_b, s2), (l3w, l3_b, s3), (outw, np.array([out_b], np.float32), so)]:
                f.write(struct.pack('i', w8.size)); f.write(w8.tobytes())
                f.write(struct.pack('i', b.size)); f.write(b.astype(np.float32).tobytes())
                if np.ndim(s):   # LNR8's per-row scales: a float tensor
                    f.write(struct.pack('i', s.size)); f.write(s.astype(np.float32).tobytes())
                else:
                    f.write(struct.pack('f', float(s)))
        if linh is not None:
            f.write(b'LINH'); f.write(struct.pack('i', linh.size)); f.write(linh.astype(np.float32).tobytes())
    kind = f"{nb} output buckets" if nb > 1 else "one tail"
    scales = "a scale per row of L2 and L3" if rows else "a scale per layer"
    print(f"wrote {dst}: L1={L1q} L2={L2} L3={L3}, {kind}, {scales}  (FT float, L2/L3/out int8){' +lin' if linh is not None else ''}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--rows"]
    if len(args) != 2:
        print("usage: python quantize_i8.py <float LNN1/LNB1 net> <out LNI8/LNB8 net> [--rows: LNR8, per-row scales]")
        sys.exit(1)
    main(args[0], args[1], rows="--rows" in sys.argv[1:])
