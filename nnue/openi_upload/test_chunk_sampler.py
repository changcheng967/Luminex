"""Sampler test for NNUE_CHUNK: over one pass, every batch is NNUE_BS / c runs of c consecutive
rows, each run starts on a multiple of c, and every run of the pass appears exactly once; with
c = 1 the batches are the plain permutation. Runs on the CPU, no data or GPU needed.
usage: python test_chunk_sampler.py"""
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(chunk, bs):
    os.environ.update({"NNUE_DEVICE": "cpu", "NNUE_CHUNK": str(chunk), "NNUE_BS": str(bs)})
    import resident_train
    return importlib.reload(resident_train)


n = 100_003   # not a multiple of the chunk or the batch: the pass must stay in range
for chunk, bs in ((16, 1024), (8, 512), (1, 1024)):
    RT = load(chunk, bs)
    shuffle = RT.HostShuffle(n // chunk, 7)
    perm = shuffle.next_pass()
    steps = n // bs
    rows = torch.cat([RT.batch_rows(perm, i) for i in range(0, steps * bs, bs)])
    assert rows.numel() == steps * bs and int(rows.max()) < n and int(rows.min()) >= 0
    if chunk > 1:
        runs = rows.view(-1, chunk)
        assert bool((runs[:, 0] % chunk == 0).all()), "a run starts off its grid"
        assert bool((runs.diff(dim=1) == 1).all()), "a run is not consecutive"
        assert runs[:, 0].unique().numel() == runs.shape[0], "a run appears twice in one pass"
    else:
        assert rows.unique().numel() == rows.numel(), "a position appears twice in one pass"
    print(f"  chunk {chunk:2d}, batch {bs}: {steps} batches, {rows.numel():,} rows, "
          f"{rows.numel() // chunk:,} distinct runs, all in range")
print("CHUNK_SAMPLER_PASS")
