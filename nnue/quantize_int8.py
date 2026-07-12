#!/usr/bin/env python3
"""Quantize a float LNN1 .nnue -> int8 LNI8 .nnue.

Why: the float L2 weight matrix is 64KB (16 outputs x 1024 inputs x 4 bytes) and spills
the 32KB L1 cache -> evaluate is memory-bound at ~2.5us. int8 L2 weights are 16KB (fit L1)
and the L2/L3/out matmuls use VPMADDUBSW (32 int8 MACs/instruction) -> ~4x faster evaluate.

Format LNI8:
  magic 'LNI8' | L1 L2 L3 NUM_INPUTS (4x int32)
  FT (kept float): ft.weight (L1*NI float), ft.bias (L1 float)   [int16 accumulator built from these at load]
  per layer (L2, L3, out): int8 weights (n) | float bias (n) | float scale (1)
  The int8 dot product result (int32) is dequantized as: bias + int32 / (scale * 127).
"""
import sys, struct, numpy as np

def read_tensor(f):
    (n,) = struct.unpack('i', f.read(4)); return np.frombuffer(f.read(n*4), dtype=np.float32).copy()

def quant(w):
    s = 127.0 / max(float(np.abs(w).max()), 1e-8)
    q = np.round(w * s).clip(-127, 127).astype(np.int8)
    return q, np.float32(s)

def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, 'rb') as f:
        magic = f.read(4)
        assert magic == b'LNN1', f"not a float LNN1 net: {magic}"
        L1, L2, L3, NI = struct.unpack('iiii', f.read(16))
        ft_w = read_tensor(f).reshape(L1, NI)
        ft_b = read_tensor(f)
        l2_w = read_tensor(f).reshape(L2, 2*L1); l2_b = read_tensor(f)
        l3_w = read_tensor(f).reshape(L3, L2);    l3_b = read_tensor(f)
        out_w = read_tensor(f).reshape(1, L3);    out_b = read_tensor(f)[0]
    l2w, s2 = quant(l2_w); l3w, s3 = quant(l3_w); outw, so = quant(out_w)
    with open(dst, 'wb') as f:
        f.write(b'LNI8'); f.write(struct.pack('iiii', L1, L2, L3, NI))
        # FT float
        f.write(struct.pack('i', ft_w.size)); f.write(ft_w.astype(np.float32).tobytes())
        f.write(struct.pack('i', ft_b.size)); f.write(ft_b.astype(np.float32).tobytes())
        for (w8, b, s) in [(l2w, l2_b, s2), (l3w, l3_b, s3), (outw, np.array([out_b],np.float32), so)]:
            f.write(struct.pack('i', w8.size)); f.write(w8.tobytes())
            f.write(struct.pack('i', b.size));  f.write(b.astype(np.float32).tobytes())
            f.write(struct.pack('f', float(s)))
    print(f"wrote {dst}: L1={L1} L2={L2} L3={L3} | scales L2={s2:.1f} L3={s3:.1f} out={so:.1f}", flush=True)
    # quick accuracy check: max weight quantization error
    for name, w, q, s in [("L2",l2_w,l2w,s2),("L3",l3_w,l3w,s3),("out",out_w,outw,so)]:
        err = np.abs(w - q.astype(np.float32)/s).max()
        print(f"  {name} max quant err = {err:.5f} (rel {err/(np.abs(w).max()+1e-8):.3f})", flush=True)

if __name__ == "__main__":
    main()
