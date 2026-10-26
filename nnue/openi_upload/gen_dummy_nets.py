#!/usr/bin/env python3
"""Dummy-net generator for the FT-width thread-scaling test.

Emits valid LNN1 (float) nets with random weights at a chosen L1 so the engine's
runtime-dimension loader can be benchmarked at different FT widths without
training anything. Weights don't affect throughput; only shapes do.
Each tensor is stored as int32 count + count floats (read_tensor contract).
  python3 gen_dummy_nets.py <l1> [<l1> ...]  ->  dummy_l1_<n>.nnue in cwd
"""
import random
import struct
import sys

NUM_INPUTS = 24576  # HalfKAv2_hm


def tensor(f, vals, scale):
    f.write(struct.pack("<i", len(vals)))
    f.write(struct.pack(f"<{len(vals)}f", *[random.uniform(-scale, scale) for _ in vals]))


def write_net(l1: int, path: str, l2: int = 16, l3: int = 32, seed: int = 42):
    random.seed(seed)
    with open(path, "wb") as f:
        f.write(b"LNN1")
        f.write(struct.pack("<4i", l1, l2, l3, NUM_INPUTS))
        # FT weights: row-major [l1][NUM_INPUTS] per the loader's transpose read
        tensor(f, [0.02] * (l1 * NUM_INPUTS), 0.02)
        tensor(f, [0.0] * l1, 0.1)                    # ft bias
        tensor(f, [0.0] * (l2 * 2 * l1), 0.05)        # L2 weights
        tensor(f, [0.0] * l2, 0.1)                    # L2 bias
        tensor(f, [0.0] * (l3 * l2), 0.05)            # L3 weights
        tensor(f, [0.0] * l3, 0.1)                    # L3 bias
        tensor(f, [0.0] * l3, 0.05)                   # out weights
        f.write(struct.pack("<i", 1))                 # out bias tensor
        f.write(struct.pack("<f", 0.0))
    print(f"wrote {path}: L1={l1} L2={l2} L3={l3}")


if __name__ == "__main__":
    for l1 in [int(x) for x in sys.argv[1:]] or [768, 512, 384]:
        write_net(l1, f"dummy_l1_{l1}.nnue")
