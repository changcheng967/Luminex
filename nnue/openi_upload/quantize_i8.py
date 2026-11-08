#!/usr/bin/env python3
"""Convert a float net (output by the C500 trainer) to the int8 net the engine
loads. Pure numpy, runs on CPU in seconds. Run LOCALLY after downloading the
trainer's output from the C500 job:

    python quantize_i8.py luminex_v2.nnue luminex_v6_i8.nnue

A float LNN1 net (one dense tail) becomes LNI8; an LNB1 net (one tail per
output bucket, NNUE_BUCKETS > 1) becomes LNB8, each bucket's tail quantized
with its own scales. Then drop the result onto the engine and bench it.
"""
import sys, struct
import numpy as np


def rt(f):
    (n,) = struct.unpack('i', f.read(4))
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def q(w):
    s = 127.0 / max(float(np.abs(w).max()), 1e-8)
    return np.round(w * s).clip(-127, 127).astype(np.int8), np.float32(s)


def main(src, dst):
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
    print(f"wrote {dst}: L1={L1q} L2={L2} L3={L3}, {kind}  (FT float, L2/L3/out int8){' +lin' if linh is not None else ''}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python quantize_i8.py <float LNN1/LNB1 net> <out LNI8/LNB8 net>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
