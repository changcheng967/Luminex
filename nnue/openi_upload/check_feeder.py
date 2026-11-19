"""Checks PoolFeeder's feeder.cpp path (NNUE_FEED_THREADS > 0) against its numpy path
(NNUE_FEED_THREADS=0): the same batches, bit for bit, over several passes. Exit 0 = pass.

usage: python check_feeder.py      (NNUE_DEVICE=cpu runs it without a GPU)
Two pool layouts, each for three passes: 300K VRAM + 500K host-RAM positions (20 Feistel
bits, 76% of them valid) and 500K + 1M (22 bits, 36% valid: the cycle walking of a
6.6B-position run). Every row holds its position number, so each batch shows exactly
which positions it drew, and each pass must draw every position at most once."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resident_train as RT  # noqa: E402


def pools(ng, nh):
    rows = np.zeros((ng + nh, RT.REC), dtype=np.uint8)
    ids = np.arange(ng + nh, dtype=np.uint64)
    rows.view(np.uint64)[:, 0] = ids
    rows.view(np.uint64)[:, 1] = ids * np.uint64(0x9E3779B97F4A7C15)   # the rest of the row varies too
    return torch.from_numpy(rows[:ng]).to(RT.dev), torch.from_numpy(rows[ng:].copy())


def batches(gpu, host, threads, bs, count):
    RT.FEED_THREADS = threads
    feeder = RT.PoolFeeder(gpu, host, 7, bs)
    if (feeder.lib is not None) != (threads > 0):
        raise SystemExit(f"FAIL: NNUE_FEED_THREADS={threads} but the feeder took the "
                         f"{'C++' if feeder.lib is not None else 'numpy'} path")
    out, t = [], time.perf_counter()
    for _ in range(count):
        b, epoch = feeder.next()
        out.append((b.cpu().numpy(), epoch))
    return out, (time.perf_counter() - t) / count * 1e3


ok = True
for ng, nh, bs in ((300_000, 500_000, 65_536), (500_000, 1_000_000, 65_536)):
    gpu, host = pools(ng, nh)
    per_pass = (ng + nh) // bs
    ref, t_np = batches(gpu, host, 0, bs, 3 * per_pass)
    new, t_cpp = batches(gpu, host, 8, bs, 3 * per_pass)
    same = all(a.shape == b.shape and np.array_equal(a, b) and ea == eb for (a, ea), (b, eb) in zip(ref, new))
    once = True
    for p in range(3):
        ids = np.concatenate([b.view(np.uint64)[:, 0] for b, _ in new[p * per_pass:(p + 1) * per_pass]])
        once &= len(np.unique(ids)) == len(ids) == per_pass * bs and int(ids.max()) < ng + nh
        once &= all(e == p for _, e in new[p * per_pass:(p + 1) * per_pass])
    ks = RT.KeyedShuffle(ng + nh, 7)
    print(f"  {ng:,} VRAM + {nh:,} host positions, {2 * int(ks.h)} Feistel bits, {3 * per_pass} batches of {bs:,}: "
          f"identical to the numpy path: {same}; every pass draws each position at most once: {once} "
          f"(numpy {t_np:.1f} ms, feeder.cpp {t_cpp:.1f} ms a batch here)", flush=True)
    ok &= same and once
print("FEEDER_PASS" if ok else "FEEDER_FAIL", flush=True)
sys.exit(0 if ok else 1)
