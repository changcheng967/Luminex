#!/usr/bin/env python3
"""Convert a float LNN1 net (output by the C500 trainer) to the int8 LNI8 net the
engine loads. Pure numpy, runs on CPU in seconds. Run LOCALLY after downloading
luminex_v2.nnue from the C500 job output:

    python quantize_i8.py luminex_v2.nnue luminex_v6_i8.nnue

Then drop luminex_v6_i8.nnue onto the engine and bench vs HCE / Stash.
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
        assert f.read(4) == b'LNN1', f"{src} is not a float LNN1 net"
        L1q, L2, L3, NI = struct.unpack('iiii', f.read(16))
        ft_w = rt(f).reshape(L1q, NI); ft_b = rt(f)
        l2_w = rt(f).reshape(L2, 2 * L1q); l2_b = rt(f)
        l3_w = rt(f).reshape(L3, L2);       l3_b = rt(f)
        out_w = rt(f).reshape(1, L3);       out_b = rt(f)[0]
    l2w, s2 = q(l2_w); l3w, s3 = q(l3_w); outw, so = q(out_w)
    with open(dst, 'wb') as f:
        f.write(b'LNI8'); f.write(struct.pack('iiii', L1q, L2, L3, NI))
        for arr in [ft_w, ft_b]:
            f.write(struct.pack('i', arr.size)); f.write(arr.astype(np.float32).tobytes())
        for w8, b in [(l2w, l2_b), (l3w, l3_b), (outw, np.array([out_b], np.float32))]:
            f.write(struct.pack('i', w8.size)); f.write(w8.tobytes())
            f.write(struct.pack('i', b.size)); f.write(b.astype(np.float32).tobytes())
            f.write(struct.pack('f', float(s2 if w8 is l2w else s3 if w8 is l3w else so)))
    print(f"wrote {dst}: L1={L1q} L2={L2} L3={L3}  (FT float, L2/L3/out int8)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python quantize_i8.py <float_lnn1.nnue> <out_lni8.nnue>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
