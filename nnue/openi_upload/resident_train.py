#!/usr/bin/env python3
"""GPU-resident NNUE trainer for one large-memory CUDA GPU.

The whole dataset sits in VRAM as 32-byte LCR1 records, written by
`luminex-featurize --fen-eval --compact`. Each batch is expanded to
HalfKAv2_hm feature indices on the GPU, so there is no featurizing and no pipe
traffic while the network trains, and no host synchronization inside a step.
Network, loss, optimizer groups, learning-rate schedule and output files are
those of c500_train_stream.py: a .pt from either trainer resumes in the other.

Environment (default in brackets):
  NNUE_DATA          LCR1 file(s): a path, or a comma-separated list of paths and
                     globs (binpack2lcr's OUT.*.lcr parts); a single missing path is
                     built from NNUE_FENEVAL_FILE first
  NNUE_MAX_POS [all] load at most this many positions (to fit the card)
  NNUE_VRAM_POS [0 = all] keep at most this many training positions in VRAM and the
                     rest in host RAM: each batch draws from both pools in proportion
                     (a background thread gathers the host-RAM rows a few batches
                     ahead), and each pass's order comes from a keyed permutation that
                     stores nothing, so the data can exceed the card and 2^32 positions
  NNUE_FEED_THREADS [8]  threads that make each host-RAM batch in feeder.cpp (built
                     with g++ on first use, next to this file): the permutation, the
                     split between the pools and the gather, in one call without the GIL.
                     The batches are the numpy path's, bit for bit. 0 = the numpy path
  NNUE_FENEVAL_FILE  fen<TAB>cp[<TAB>flag] lines (.xz, .zst or plain text)
  NNUE_DATA_VIA_FILE [0]  1: decompress to a temporary text file, then featurize
                     file to file (no pipe at all)
  FEAT               luminex-featurize built with --compact (default: next to this file)
  NNUE_FEAT_THREADS  [all cores]
  NNUE_RESUME        warm start: float .nnue (LNN1/LNB1) or trainer .pt
  NNUE_L1 [768]  NNUE_BUCKETS [8]  NNUE_BS [131072]  NNUE_LR [1e-4]
  NNUE_TAIL_LR_MULT [1]  NNUE_TAIL_WD [1e-2]  NNUE_GRAD_CLIP [1.0]  NNUE_SEED [1]
  NNUE_T_MAX_STEPS   cosine horizon in steps [one pass over the training data]
  NNUE_BUDGET_SEC    [0 = none] wall-clock budget; the horizon shrinks to fit it
  NNUE_LOSS_POWER [2.6]  NNUE_QUIET_LOSS ["" = unweighted, e.g. 1,0.3,0]
  NNUE_VAL_FRAC [0.002]  share of positions held out by FEN hash (never trained on)
  NNUE_VAL_DATA      LCR1 file(s) or globs from games the run never trains on (another
                     month): a [DISJOINT] line beside each [HEALTH] line. The hash split
                     shares games with the training data, so only these positions show
                     a net memorising games or fitting a biased corpus
  NNUE_VAL_MAX [5e6] positions of NNUE_VAL_DATA to load
  NNUE_EVAL_EVERY [1000] steps between [HEALTH] validation lines
  NNUE_SAVE_EVERY [3000] steps between checkpoint saves
  NNUE_KEEP_EVERY [0 = off] steps between numbered copies of the float net
                     (<name>_s<step>.nnue in NNUE_OUT_DIR), so checkpoints along the
                     run can be gated
  NNUE_COMPILE [0]   1: torch.compile the step from the expanded batch to the loss
  NNUE_DEVICE [cuda] cpu runs every code path slowly, for a dry run without a GPU
  NNUE_CONTINUE [0] 1: a resumed .pt keeps its schedule. NNUE_T_MAX_STEPS then counts
                     from the run's first step, so the cosine and the stop carry on
                     where the checkpoint left off (a run cut short resumes as planned)
  NNUE_SCHED_START [0]  with NNUE_CONTINUE=1: the step at which the checkpoint's schedule
                     began, for a run that had itself resumed a .pt on a new cosine (run05
                     started from gen768r_04 at 860000): the schedule then carries on from
                     gstep - NNUE_SCHED_START, not from gstep
  NNUE_LIVE_CLAMP [0]  tail activations pass this gradient slope outside [0, 1];
                     the forward pass (and so the engine's net) is unchanged
  NNUE_QAT [0]       1: train the network the way the engine runs it. The first layer
                     is rounded to the engine's int16 grid (x8192); each bucket's L2, L3
                     and output weights to int8 with quantize_i8.py's scale (127 / max|w|);
                     every activation into them to the engine's 1/127 steps. Gradients
                     pass straight through the rounding and the tail runs in fp32, so the
                     network trained is the network the engine loads after quantize_i8.py
  NNUE_QAT_FT8 [0]   with NNUE_QAT: SHIFT (1-8) rounds the first layer's weights instead to the
                     int8 table of an LNF8 net, steps of 2^(SHIFT-13) clipped at +-127 steps
                     (quantize_i8.py --ft8 SHIFT / --ft-grid SHIFT); the bias stays on x8192
  NNUE_FT_DENSE [0]  the feature layer as dense per-king-bucket matmuls on tensor cores
                     (positions sorted by bucket into tiles; X 0/1 times the bucket's
                     768 x L1 block) instead of EmbeddingBag: 1 = the weight gradient
                     only (the forward stays EmbeddingBag's, bit-identical), 2 = the
                     forward too (bf16 inputs, fp32 accumulation; with NNUE_QAT in
                     exact fp32, so it still equals the engine's int16 sums)
                     3 = ft_fused.py's Triton kernels: X tiles built on chip from piece
                     bitboards (no X in memory), both sides in one launch, bf16 forward and
                     bf16 weight gradient with fp32 accumulation (not with NNUE_QAT)
  NNUE_FT_FP8 [0]    with NNUE_FT_DENSE=3: 1 = the weight gradient in FP8 (G in e4m3, one
                     global scale and a power-of-two boost per row), 2 = the forward too
                     (the table in e4m3 with a scale per column); 0 = bf16 both ways
  NNUE_FT_TILE [1024]  rows per tile of the dense feature layer
  NNUE_FT_TF32 [1]   the dense gradient matmul in TF32 (0: exact fp32, slower)
  NNUE_TARGET_CLIP [1500]  |target| cap inside the loss (mates and big scores stay
                     in the data and train as "winning"; 1500 = 97.7% in the loss)
  NNUE_LOSS_WIDE ["" ]  "scale,weight": add weight * the same loss at a wider
                     sigmoid scale, which keeps a gradient in won positions
  NNUE_PROFILE [0]   profile this many steps after NNUE_PROFILE_WARMUP [300] steps of the
                     run (torch.profiler: GPU time by kernel and by kind, per step), write
                     the trace next to the net, and stop
  NNUE_OUT_DIR [.]  NNUE_OUT_NAME [luminex_resident.nnue]
"""
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("NNUE_BUCKETS", "8")
from luminex_nnue_train import (LNNUE, LOSS_SCALE, NUM_INPUTS,  # noqa: E402
                                parse_quiet_loss, save_nnue, stm_split_gate)

REC = 32   # LCR1 record bytes (src/featurize.cpp CompactRec)


def env(name, default):
    return os.environ.get(name, default)


L1 = int(env("NNUE_L1", "768"))
BS = int(env("NNUE_BS", "131072"))
LR = float(env("NNUE_LR", "1e-4"))
TAIL_LR_MULT = float(env("NNUE_TAIL_LR_MULT", "1"))
TAIL_WD = float(env("NNUE_TAIL_WD", "1e-2"))
GRAD_CLIP = float(env("NNUE_GRAD_CLIP", "1.0"))
SEED = int(env("NNUE_SEED", "1"))
T_MAX = int(env("NNUE_T_MAX_STEPS", "0"))
BUDGET = float(env("NNUE_BUDGET_SEC", "0"))
POWER = float(env("NNUE_LOSS_POWER", "2.6"))
QW = parse_quiet_loss(env("NNUE_QUIET_LOSS", ""))
VAL_FRAC = float(env("NNUE_VAL_FRAC", "0.002"))
VRAM_POS = int(float(env("NNUE_VRAM_POS", "0")))
FEED_THREADS = int(env("NNUE_FEED_THREADS", "8"))
VAL_DATA = env("NNUE_VAL_DATA", "")
VAL_MAX = int(float(env("NNUE_VAL_MAX", "5e6")))
EVAL_EVERY = int(env("NNUE_EVAL_EVERY", "1000"))
SAVE_EVERY = int(env("NNUE_SAVE_EVERY", "3000"))
KEEP_EVERY = int(env("NNUE_KEEP_EVERY", "0"))
CONTINUE = env("NNUE_CONTINUE", "0") == "1"
SCHED_START = int(float(env("NNUE_SCHED_START", "0")))
COMPILE = env("NNUE_COMPILE", "0") == "1"
LIVE = float(env("NNUE_LIVE_CLAMP", "0"))
QAT = env("NNUE_QAT", "0") == "1"
QAT_FT8 = int(env("NNUE_QAT_FT8", "0"))
if QAT_FT8 and not (QAT and 1 <= QAT_FT8 <= 8):
    raise SystemExit(f"FAIL: NNUE_QAT_FT8={QAT_FT8} needs NNUE_QAT=1 and a shift from 1 to 8")
FT_DENSE = int(env("NNUE_FT_DENSE", "0"))
FT_TILE = int(env("NNUE_FT_TILE", "1024"))
FT_FP8 = int(env("NNUE_FT_FP8", "0"))
PROFILE = int(env("NNUE_PROFILE", "0"))
PROFILE_WARMUP = int(env("NNUE_PROFILE_WARMUP", "300"))
FT_TF32 = env("NNUE_FT_TF32", "1") == "1"
WIDE = [float(x) for x in env("NNUE_LOSS_WIDE", "").split(",")] if env("NNUE_LOSS_WIDE", "") else None
TARGET_CLIP = float(env("NNUE_TARGET_CLIP", "1500"))   # |target| cap in the loss; data keeps raw scores
OUT_DIR = env("NNUE_OUT_DIR", ".")
OUT = env("NNUE_OUT_NAME", "luminex_resident.nnue")
OUT_BASE = OUT[:-5] if OUT.endswith(".nnue") else OUT
dev = env("NNUE_DEVICE", "cuda")   # cpu: a slow dry run of every code path on a machine without a GPU


# ---------------------------------------------------------------------------
# Data: LCR1 file -> VRAM, split by FEN hash
# ---------------------------------------------------------------------------
def build_data(src, dst):
    """fen-eval text -> LCR1 file through the featurizer's --compact mode."""
    feat = env("FEAT", os.path.join(HERE, "luminex-featurize"))
    threads = int(env("NNUE_FEAT_THREADS", str(os.cpu_count() or 8)))
    dec = "xz -T0 -dc" if src.endswith(".xz") else "zstd -dc" if src.endswith(".zst") else None
    tmp, text = dst + ".partial", None
    if dec and env("NNUE_DATA_VIA_FILE", "0") == "1":
        text = dst + ".text"
        print(f"[data] decompressing {src} -> {text}", flush=True)
        t = time.time()
        subprocess.run(["bash", "-c", f"{dec} '{src}' > '{text}'"], check=True)
        print(f"[data] decompressed {os.path.getsize(text) / 1e9:.1f} GB in {time.time() - t:.0f} s", flush=True)
        src, dec = text, None
    feat_cmd = f"'{feat}' --fen-eval --compact --threads {threads} --output '{tmp}'"
    cmd = f"set -o pipefail; {dec} '{src}' | {feat_cmd}" if dec else f"{feat_cmd} --input '{src}'"
    print(f"[data] building {dst}: {cmd}", flush=True)
    t = time.time()
    pr = subprocess.Popen(["bash", "-c", cmd], stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in pr.stderr:   # progress every 50M positions, and the summary line
        m = re.match(r"fen-eval: (\d+)M positions", line)
        if "featurized" in line or (m and int(m.group(1)) % 50 == 0):
            print("  " + line.strip(), flush=True)
    if pr.wait() != 0:
        raise SystemExit(f"FAIL: featurizer exited with {pr.returncode} ({cmd})")
    os.replace(tmp, dst)
    if text:
        os.remove(text)
    print(f"[data] {os.path.getsize(dst) // REC:,} positions in {time.time() - t:.0f} s", flush=True)


def data_files(spec):
    """NNUE_DATA: one LCR1 file, or a comma-separated list of files and glob
    patterns (binpack2lcr writes one part per thread: OUT.000.lcr, ...)."""
    import glob as _glob
    files = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        hits = sorted(_glob.glob(item)) if any(ch in item for ch in "*?[") else [item]
        files += hits
    return files


def load_data(paths):
    """LCR1 files -> (train, host, val). train: uint8 [n, 32] on the GPU; host: the
    training positions beyond NNUE_VRAM_POS, a uint8 [m, 32] CPU tensor (None when
    everything fits in VRAM). Positions whose hash field falls below VAL_FRAC * 2^32
    form the validation set (on the GPU); the rest are packed chunk by chunk, VRAM
    first, so neither pool ever holds a second copy. NNUE_MAX_POS caps the positions
    loaded (the files' order is kept)."""
    sizes = []
    for p in paths:
        s = os.path.getsize(p)
        if s % REC:
            raise SystemExit(f"FAIL: {p} is {s} bytes, not a whole number of {REC}-byte records")
        sizes.append(s // REC)
    n = sum(sizes)
    cap = int(float(env("NNUE_MAX_POS", "0"))) or n
    n = min(n, cap)
    n_gpu = min(n, VRAM_POS) if VRAM_POS > 0 else n
    thr = np.uint32(min(int(VAL_FRAC * 2 ** 32), 2 ** 32 - 1))
    train = torch.empty((n_gpu, REC), dtype=torch.uint8, device=dev)
    host = np.empty((n - n_gpu, REC), dtype=np.uint8) if n > n_gpu else None
    vals, nt, nh, nread, t = [], 0, 0, 0, time.time()
    chunk = 1 << 25   # records per read: 1 GiB
    for p in paths:
        with open(p, "rb") as f:
            while nread < n:
                a = np.fromfile(f, dtype=np.uint8, count=min(chunk, n - nread) * REC)
                if a.size == 0:
                    break
                a = a.reshape(-1, REC)
                nread += len(a)
                held = a[:, 28:32].copy().view(np.uint32).reshape(-1) < thr
                if held.any():
                    vals.append(a[held])
                keep = a[~held]
                k = min(len(keep), n_gpu - nt)
                if k > 0:
                    train[nt:nt + k].copy_(torch.from_numpy(keep[:k]))
                    nt += k
                if len(keep) > k:
                    host[nh:nh + len(keep) - k] = keep[k:]
                    nh += len(keep) - k
        if nread >= n:
            break
    val = torch.from_numpy(np.concatenate(vals)).to(dev) if vals else train[:0]
    where = f"VRAM {nt:,} + host RAM {nh:,}" if host is not None else f"{nt:,}"
    print(f"[data] {nread:,} positions from {len(paths)} file(s) -> {'VRAM and host RAM' if host is not None else 'VRAM'} "
          f"in {time.time() - t:.0f} s: train {where}, validation {len(val):,} ({VAL_FRAC:.2%} by hash), "
          f"{nread * REC / 1e9:.1f} GB{f' (capped at NNUE_MAX_POS {cap:,})' if cap < sum(sizes) else ''}", flush=True)
    return train[:nt], (torch.from_numpy(host[:nh]) if host is not None and nh else None), val


def load_records(paths, cap):
    """LCR1 files -> one uint8 [n, 32] tensor on the GPU, at most cap positions
    in file order (the game-disjoint validation set)."""
    parts, n = [], 0
    for p in paths:
        if n >= cap:
            break
        a = np.fromfile(p, dtype=np.uint8, count=(cap - n) * REC)
        if a.size % REC:
            raise SystemExit(f"FAIL: {p} is not a whole number of {REC}-byte records")
        parts.append(a.reshape(-1, REC))
        n += len(parts[-1])
    rec = (torch.from_numpy(np.concatenate(parts)).to(dev) if parts
           else torch.empty((0, REC), dtype=torch.uint8, device=dev))
    print(f"[data] game-disjoint validation: {len(rec):,} positions from {len(paths)} file(s)", flush=True)
    return rec


class HostShuffle:
    """Each pass's permutation of the training positions, made on the CPU in a
    background thread while the GPU trains the previous pass. torch.randperm on the
    GPU needs several times the permutation's size in temporary VRAM, which would
    cap the data at about 1.3B positions on a 96 GB card; this costs 4 bytes a
    position of pinned host memory and none of the VRAM."""

    def __init__(self, n, seed):
        import threading
        assert n < 2 ** 32, "uint32 permutation"
        self.n, self.seed, self.epoch = n, seed, 0
        self._next, self._thread = None, None
        self._threading = threading
        self._start()

    def _make(self, epoch):
        p = np.arange(self.n, dtype=np.uint32)
        np.random.default_rng([self.seed, epoch]).shuffle(p)
        t = torch.from_numpy(p.view(np.int32))
        self._next = t.pin_memory() if torch.cuda.is_available() else t

    def _start(self):
        self._thread = self._threading.Thread(target=self._make, args=(self.epoch,), daemon=True)
        self._thread.start()

    def next_pass(self):
        """The next pass's permutation; starts making the one after."""
        self._thread.join()
        perm = self._next
        self.epoch += 1
        self._start()
        return perm

    @staticmethod
    def batch(perm, i, bs):
        """Indices i..i+bs of a pass as int64 on the GPU (uint32 values travel as int32)."""
        return perm[i:i + bs].to(dev, non_blocking=True).to(torch.int64) & 0xFFFFFFFF


class KeyedShuffle:
    """A different pseudo-random order of 0..n-1 each pass that stores nothing: a
    4-round Feistel network on an even number of bits >= log2(n), keyed by
    (seed, pass), walking cycles past values >= n. It is a bijection, so each pass
    still visits every position exactly once, for any n (HostShuffle's permutation
    needs 4 bytes a position and n < 2^32)."""

    def __init__(self, n, seed):
        self.n, self.seed = n, seed
        bits = max(2, (n - 1).bit_length())
        bits += bits & 1
        self.h = np.uint64(bits // 2)
        self.mask = np.uint64((1 << (bits // 2)) - 1)

    def _keys(self, epoch):
        rng = np.random.default_rng([self.seed, epoch, 0x51CE])
        return [np.uint64(k) for k in rng.integers(0, 2 ** 63, size=4, dtype=np.uint64)]

    def _round(self, r, k):
        x = (r ^ k) * np.uint64(0x9E3779B97F4A7C15)   # wraps mod 2^64 (array arithmetic)
        x ^= x >> np.uint64(29)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(32)
        return x & self.mask

    def _perm(self, x, ks):
        left, right = x >> self.h, x & self.mask
        for k in ks:
            left, right = right, left ^ self._round(right, k)
        return (left << self.h) | right

    def indices(self, epoch, start, count):
        """Positions start..start+count-1 of pass `epoch`'s order (uint64 numpy)."""
        ks = self._keys(epoch)
        y = self._perm(np.arange(start, start + count, dtype=np.uint64), ks)
        out = y >= np.uint64(self.n)
        while out.any():
            y[out] = self._perm(y[out], ks)
            out = y >= np.uint64(self.n)
        return y


def feeder_lib():
    """feeder.cpp's feed_batch through ctypes (a call releases the GIL), built next to
    this file with g++ when the .so is missing or older than the source. None when
    NNUE_FEED_THREADS=0 or it can't be built or loaded: PoolFeeder then feeds through numpy."""
    if FEED_THREADS <= 0:
        return None
    import ctypes
    src, so = os.path.join(HERE, "feeder.cpp"), os.path.join(HERE, "libfeeder.so")
    try:
        if os.path.exists(src) and (not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src)):
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.run(["g++", "-O3", "-march=native", "-shared", "-fPIC", "-pthread", src, "-o", tmp],
                           check=True, capture_output=True, text=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
    except (OSError, subprocess.CalledProcessError) as e:
        why = e.stderr.strip().splitlines()[-1] if isinstance(e, subprocess.CalledProcessError) and e.stderr else e
        print(f"[feed] feeder.cpp unavailable ({why}): host-RAM batches go through numpy", flush=True)
        return None
    lib.feed_batch.restype = ctypes.c_int64
    lib.feed_batch.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_uint,
                               ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_int]
    return lib


class PoolFeeder:
    """Training batches drawn from a VRAM pool and a host-RAM pool together. A
    background thread takes each batch's positions from a KeyedShuffle over both
    pools, gathers the host-RAM rows into pinned memory and copies them and the VRAM
    indices to the GPU on a side stream, up to DEPTH batches ahead; next() waits on
    that copy and returns the [bs, 32] batch. A batch has bs positions; the share
    from each pool follows the pools' sizes. With feeder.cpp (NNUE_FEED_THREADS > 0) one
    call makes a batch, permutation and gather together, on several threads and without
    the GIL; numpy took ~26 ms a batch for the permutation alone at 6.6B positions
    (34 Feistel bits, 2.6 rounds of cycle walking), and the GPU waited for it."""

    DEPTH = 3

    def __init__(self, gpu_pool, host_pool, seed, bs):
        self.g, self.hp = gpu_pool, host_pool
        self.ng, self.n = len(gpu_pool), len(gpu_pool) + len(host_pool)
        self.bs, self.steps_per_pass = bs, (len(gpu_pool) + len(host_pool)) // bs
        self.shuffle = KeyedShuffle(self.n, seed)
        cuda = str(dev).startswith("cuda")
        self.stream = torch.cuda.Stream() if cuda else None
        nslot = self.DEPTH + 2
        self.pins = [torch.empty((bs, REC), dtype=torch.uint8, pin_memory=cuda) for _ in range(nslot)]
        self.lib, self.threads = feeder_lib(), FEED_THREADS
        if self.lib is not None:
            assert host_pool.is_contiguous() and host_pool.shape[1] == REC
            self.gpins = [torch.empty(bs, dtype=torch.int64, pin_memory=cuda) for _ in range(nslot)]
            self.scratch = np.empty(bs, dtype=np.uint64)
            self.kepoch, self.keys = -1, None
        print(f"[feed] host-RAM batches: "
              f"{f'feeder.cpp, {self.threads} threads' if self.lib is not None else 'numpy'}", flush=True)
        self.done_ev = [None] * nslot
        self.q = queue.Queue(maxsize=self.DEPTH)
        self.err = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            epoch, k, slot = 0, 0, 0
            while True:
                if self.lib is not None:
                    if self.kepoch != epoch:
                        self.kepoch, self.keys = epoch, np.array(self.shuffle._keys(epoch), dtype=np.uint64)
                    if self.done_ev[slot] is not None:
                        self.done_ev[slot].synchronize()    # this slot's last copies have landed
                    c = self.lib.feed_batch(self.hp.data_ptr(), self.ng, self.n, self.keys.ctypes.data,
                                            int(self.shuffle.h), int(self.shuffle.mask), k * self.bs, self.bs,
                                            self.gpins[slot].data_ptr(), self.pins[slot].data_ptr(),
                                            self.scratch.ctypes.data, self.threads)
                    gi, pin = self.gpins[slot][:c], self.pins[slot][:self.bs - c]
                else:
                    idx = self.shuffle.indices(epoch, k * self.bs, self.bs)
                    in_gpu = idx < np.uint64(self.ng)
                    gi = torch.from_numpy(idx[in_gpu].astype(np.int64))
                    hi = torch.from_numpy((idx[~in_gpu] - np.uint64(self.ng)).astype(np.int64))
                    if self.done_ev[slot] is not None:
                        self.done_ev[slot].synchronize()    # this slot's last copy has landed
                    pin = self.pins[slot][:len(hi)]
                    torch.index_select(self.hp, 0, hi, out=pin)
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        g_idx = (gi if self.lib is not None else gi.pin_memory()).to(dev, non_blocking=True)
                        h_rows = pin.to(dev, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(self.stream)
                    self.done_ev[slot] = ev
                else:
                    g_idx, h_rows, ev = gi.clone(), pin.clone(), None   # the slot's buffers are reused
                self.q.put((g_idx, h_rows, ev, epoch))
                slot = (slot + 1) % len(self.pins)
                k += 1
                if k == self.steps_per_pass:
                    k, epoch = 0, epoch + 1
        except Exception as e:   # surface in next(), never hang the trainer
            self.err = e
            self.q.put(None)

    def next(self):
        item = self.q.get()
        if item is None:
            raise RuntimeError(f"PoolFeeder thread failed: {self.err!r}")
        g_idx, h_rows, ev, epoch = item
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            g_idx.record_stream(torch.cuda.current_stream())
            h_rows.record_stream(torch.cuda.current_stream())
        return torch.cat([self.g[g_idx], h_rows]), epoch


KB = [-1, -1, -1, -1, 31, 30, 29, 28, -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20, -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12, -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4, -1, -1, -1, -1, 3, 2, 1, 0]
KB_T = torch.tensor(KB, dtype=torch.int64, device=dev).clamp(min=0)
SQ64 = torch.arange(64, dtype=torch.int64, device=dev)
PV_T = torch.tensor([1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=torch.int64, device=dev)   # by piece code
QW_T = torch.tensor(QW, device=dev) if QW is not None else None   # made once: a per-step copy would sync


def expand(rec):
    """uint8 [n, 32] LCR1 -> (white idx, black idx, stm, target, flag, pieces).

    Index rule = src/featurize.cpp halfka_idx: square XOR 7 when that side's
    king is on files a-d, XOR 56 more for black's view; plane = piece type * 2,
    +1 for the other side's pieces; bucket from the oriented king square."""
    n = rec.shape[0]
    occ = rec[:, 0:8].contiguous().view(torch.int64).view(n)
    nb = rec[:, 8:24].to(torch.int64)
    codes = torch.stack((nb & 15, nb >> 4), dim=2).view(n, 32)
    bits = (occ.view(n, 1) >> SQ64) & 1
    slot = torch.where(bits.bool(), torch.cumsum(bits, 1) - 1, 32)
    sqs = torch.full((n, 33), 64, dtype=torch.int64, device=rec.device)
    sqs.scatter_(1, slot, SQ64.expand(n, 64))
    sqs = sqs[:, :32]
    cnt = bits.sum(1)
    valid = SQ64[:32].view(1, 32) < cnt.view(n, 1)
    wk = torch.gather(sqs, 1, ((codes == 5) & valid).long().argmax(1, keepdim=True)).view(n)
    bk = torch.gather(sqs, 1, ((codes == 11) & valid).long().argmax(1, keepdim=True)).view(n)
    xw = torch.where((wk & 7) < 4, 7, 0)
    xb = torch.where((bk & 7) < 4, 7, 0) ^ 56
    pt, blk = codes % 6, codes // 6
    wf = (xw.view(n, 1) ^ sqs) + (pt * 2 + blk) * 64 + (KB_T[xw ^ wk] * 768).view(n, 1)
    bf = (xb.view(n, 1) ^ sqs) + (pt * 2 + 1 - blk) * 64 + (KB_T[xb ^ bk] * 768).view(n, 1)
    wf = torch.where(valid, wf, NUM_INPUTS)
    bf = torch.where(valid, bf, NUM_INPUTS)
    target = rec[:, 24:26].contiguous().view(torch.int16).view(n).float()
    flags = rec[:, 26].to(torch.int64)
    return wf, bf, (flags & 1).float(), target, (flags >> 1) & 3, cnt


def material_edge(rec):
    """White's material edge in pawns (1/3/3/5/9) of each LCR1 record."""
    n = rec.shape[0]
    nb = rec[:, 8:24].to(torch.int64)
    codes = torch.stack((nb & 15, nb >> 4), dim=2).view(n, 32)
    valid = SQ64[:32].view(1, 32) < rec[:, 27].to(torch.int64).view(n, 1)
    return (PV_T[codes] * valid).sum(1)


# ---------------------------------------------------------------------------
# Network pieces (parameters are LNNUE's; only the evaluation order differs)
# ---------------------------------------------------------------------------
class LiveSCReLU(torch.autograd.Function):
    """clamp(x, 0, 1)^2 forward; backward passes `slope` where the true
    derivative is zero, so a unit that is negative (or above 1) on every
    position still receives a gradient and can come back."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return torch.clamp(x, 0.0, 1.0).square()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        d = torch.where((x > 0) & (x < 1), 2.0 * x, torch.full_like(x, ctx.slope))
        return g * d, None


def tail_act(x):
    return LiveSCReLU.apply(x, LIVE) if LIVE > 0 else torch.clamp(x, 0.0, 1.0).square()


def fq_w(w, groups):
    """`w` as the engine holds it: each of `groups` equal blocks of rows (one per output
    bucket) rounded to int8 with quantize_i8.py's scale 127 / max|w|, then scaled back.
    The scale is a constant and the gradient passes straight through the rounding."""
    g = w.view(groups, -1)
    s = 127.0 / g.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    q = torch.round(g * s).clamp(-127.0, 127.0) / s
    return (g + (q - g).detach()).view_as(w)


def fq_a(a):
    """An activation in [0, 1] on the engine's 1/127 steps (src/nnue.cpp quant8 and the
    first layer's exact rounding), with a straight-through gradient."""
    return a + (torch.round(a * 127.0) / 127.0 - a).detach()


def forward_tail_qat(model, h, cnt):
    """The tail of `forward` with NNUE_QAT: int8 weights and 1/127 activations, in fp32.
    Products of the rounded values are exact in fp32, so up to the order of float sums
    this is the int8 net quantize_i8.py writes, as the engine computes it."""
    nbk = model.num_buckets
    with torch.autocast(device_type="cuda", enabled=False):
        h = fq_a(h.float())
        if nbk == 1:
            z = fq_a(tail_act(F.linear(h, fq_w(model.l2.weight, 1), model.l2.bias)))
            z = fq_a(tail_act(F.linear(z, fq_w(model.l3.weight, 1), model.l3.bias)))
            return F.linear(z, fq_w(model.out.weight, 1), model.out.bias).view(-1) * 300.0
        n, L2, L3 = h.shape[0], model.L2, model.L3
        bucket = ((cnt - 1) * nbk // 32).clamp(0, nbk - 1).view(n, 1)
        W2 = fq_w(torch.cat([m.weight for m in model.bl2]), nbk); b2 = torch.cat([m.bias for m in model.bl2])
        W3 = fq_w(torch.cat([m.weight for m in model.bl3]), nbk); b3 = torch.cat([m.bias for m in model.bl3])
        Wo = fq_w(torch.cat([m.weight for m in model.bout]), nbk); bo = torch.cat([m.bias for m in model.bout])
        z = F.linear(h, W2, b2).view(n, nbk, L2).gather(1, bucket.view(n, 1, 1).expand(n, 1, L2)).view(n, L2)
        z = fq_a(tail_act(z))
        z = F.linear(z, W3, b3).view(n, nbk, L3).gather(1, bucket.view(n, 1, 1).expand(n, 1, L3)).view(n, L3)
        z = fq_a(tail_act(z))
        return F.linear(z, Wo, bo).gather(1, bucket).view(n) * 300.0


PSQ = 768   # piece-square features per king bucket: feature index = bucket * PSQ + piece-square


def ft_plan(idx, tile):
    """Tile plan of one side's batch for the dense feature layer. All of a side's
    features lie in one king bucket, so per bucket the layer is a 0/1 [positions, 768]
    matrix times that bucket's [768, L1] block of the table. Positions are sorted by
    bucket into tiles of `tile` rows, each bucket padded to whole tiles: at most
    ceil(n / tile) + 32 tiles, so shapes are static and nothing syncs with the host.
    Returns X [ntile, tile, 768] (fp32, 0/1), row [n] (each position's row in the
    tiles), src [ntile * tile] (each tile row's position; padding rows name position
    0, harmless because their X rows are zero) and tile_kb [ntile] (each tile's
    bucket; unused tiles are all zero)."""
    n, dv = idx.shape[0], idx.device
    nkb = NUM_INPUTS // PSQ
    ntile = (n + tile - 1) // tile + nkb
    valid = idx < NUM_INPUTS
    kb = idx[:, 0] // PSQ                      # slot 0 always holds a piece
    counts = torch.bincount(kb, minlength=nkb)
    padded = (counts + tile - 1) // tile * tile
    end_pad = torch.cumsum(padded, 0)
    start_pad, start_cnt = end_pad - padded, torch.cumsum(counts, 0) - counts
    order = torch.argsort(kb, stable=True)
    kb_s = kb[order]
    row_sorted = start_pad[kb_s] + torch.arange(n, device=dv) - start_cnt[kb_s]
    row = torch.empty_like(row_sorted)
    row[order] = row_sorted
    src = torch.zeros(ntile * tile, dtype=row.dtype, device=dv)
    src[row_sorted] = order
    tile_kb = torch.searchsorted(end_pad, torch.arange(ntile, device=dv) * tile, right=True).clamp(max=nkb - 1)
    # A plain scatter of ones: a position's features are distinct, so only the spare
    # slot at the end, which takes every padding slot, is written twice. (Scattering
    # with accumulate=True instead runs PyTorch's sort-based kernel: ~1 ms a side.)
    size = ntile * tile * PSQ
    X = torch.zeros(size + 1, dtype=torch.float32, device=dv)
    X[torch.where(valid, row.view(-1, 1) * PSQ + idx % PSQ, size).view(-1)] = 1.0
    return X[:size].view(ntile, tile, PSQ), row, src, tile_kb


class DenseFT(torch.autograd.Function):
    """One side's feature layer (the sum of each position's active rows of `weight`)
    with its weight gradient, and with NNUE_FT_DENSE=2 its forward too, done as one
    batched matmul per tile on tensor cores instead of EmbeddingBag's sort and segment
    sums. NNUE_FT_DENSE=1 keeps EmbeddingBag's forward, so the forward pass is
    bit-identical to the default and only the gradient's summation order changes.

    The dense forward runs in bf16 on the GPU, except under NNUE_QAT (and on the
    CPU), where it runs in exact fp32 with autocast and TF32 off: QAT's weights sit
    on the engine's int16 grid (k / 8192), and any sum of a position's 32 of them is
    exact in fp32, so the layer equals the engine's integer sums. bf16 would round
    each weight to 8 significant bits and undo what QAT trains for."""

    @staticmethod
    def forward(ctx, idx, weight, dense_fwd):
        X, row, src, tile_kb = ft_plan(idx, FT_TILE)
        ctx.save_for_backward(X, src, tile_kb)
        ctx.nrows = weight.shape[0]
        if not dense_fwd:
            return F.embedding_bag(idx, weight, mode="sum", padding_idx=NUM_INPUTS)
        L1 = weight.shape[1]
        if QAT or not X.is_cuda:
            Wt = weight[:NUM_INPUTS].float().view(-1, PSQ, L1)[tile_kb]   # [ntile, 768, L1]
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                with torch.autocast(device_type=X.device.type, enabled=False):
                    out = torch.bmm(X, Wt).view(-1, L1)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            return out[row]
        Wt = weight[:NUM_INPUTS].to(torch.bfloat16).view(-1, PSQ, L1)[tile_kb]   # cast the table once, then copy blocks
        out = torch.bmm(X.to(torch.bfloat16), Wt).view(-1, L1)
        return out[row].float()

    @staticmethod
    def backward(ctx, g):
        X, src, tile_kb = ctx.saved_tensors
        ntile, tile, _ = X.shape
        L1 = g.shape[1]
        G = g.float()[src].view(ntile, tile, L1)   # one gather; padding rows meet zero rows of X
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = FT_TF32
        try:
            dWt = torch.bmm(X.transpose(1, 2), G)   # [ntile, 768, L1]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        dW = torch.zeros((NUM_INPUTS // PSQ, PSQ, L1), dtype=torch.float32, device=g.device)
        dW.index_add_(0, tile_kb, dWt)
        grad = torch.zeros((ctx.nrows, L1), dtype=torch.float32, device=g.device)
        grad[:NUM_INPUTS] = dW.view(NUM_INPUTS, L1)   # the padding row keeps a zero gradient
        return None, grad, None


@torch.compiler.disable
def ft_dense(idx, weight):
    return DenseFT.apply(idx, weight, FT_DENSE >= 2)


if FT_FP8 and FT_DENSE != 3:
    raise SystemExit('NNUE_FT_FP8 needs NNUE_FT_DENSE=3')
if FT_DENSE == 3:
    import ft_fused as FF
    if QAT:
        raise SystemExit('NNUE_FT_DENSE=3 has no QAT path (QAT needs the exact fp32 sums of FT_DENSE=2)')
    FUSED_FWD = dict(BM=128, BN=128, num_warps=4, num_stages=2)    # best of the 2026-11-12 sweeps: 1.247 ms
    FUSED_BWD = dict(BMF=128, BN=128, BK=32, num_warps=4, num_stages=4)
    FUSED_CH = 8192


class FusedFT(torch.autograd.Function):
    """Both sides' feature-layer sums straight from the LCR1 records (ft_fused.py):
    bitboards, a stable sort by king bucket, the forward kernel, and in backward the
    weight-gradient kernel. Returns White's and Black's [n, L1] sums in bf16, bias not
    added (the caller adds ft_bias, so autograd keeps its gradient)."""

    @staticmethod
    def forward(ctx, wf, bf, weight):
        n, L1 = wf.shape[0], weight.shape[1]
        planes, kb = FF.side_rows(wf, bf, NUM_INPUTS)   # 0.9 ms; rec_bitboards is exact but 4.1 ms until fixed
        order, rf = FF.plan(kb, FUSED_FWD['BM'])
        rb = FF.plan(kb, FUSED_CH)[1]
        planes_s = planes[order].contiguous()
        zero = torch.zeros(L1, dtype=torch.float32, device=wf.device)
        if FT_FP8 >= 2:   # the table in e4m3, scaled per column
            w8, wsc = FF.fp8_cols(weight[:NUM_INPUTS])
            out_s = FF.ft_fwd(planes_s, rf, w8, zero, fp8=True, wscale=wsc, **FUSED_FWD)
        else:
            out_s = FF.ft_fwd(planes_s, rf, weight[:NUM_INPUTS].to(torch.bfloat16), zero, **FUSED_FWD)
        out = torch.empty_like(out_s)
        out[order] = out_s                     # back to side-row order: White 0..n-1, Black n..2n-1
        ctx.save_for_backward(planes_s, rb, order)
        ctx.nrows = weight.shape[0]
        return out[:n], out[n:]

    @staticmethod
    def backward(ctx, gw, gb):
        planes_s, rb, order = ctx.saved_tensors
        g = torch.cat([gw, gb])[order]
        dw = torch.zeros((ctx.nrows, g.shape[1]), dtype=torch.float32, device=g.device)
        if FT_FP8 >= 1:   # X^T G = inv * X^T diag(s) G8 (ft_fused.fp8_rows)
            g8, gs, ginv = FF.fp8_rows(g)
            FF.ft_bwd(planes_s, rb, g8, dw[:NUM_INPUTS], fp8=True, gscale=gs, **FUSED_BWD)
            dw[:NUM_INPUTS] *= ginv
        else:
            FF.ft_bwd(planes_s, rb, g.to(torch.bfloat16).contiguous(), dw[:NUM_INPUTS], **FUSED_BWD)
        return None, None, dw   # the padding row keeps a zero gradient


@torch.compiler.disable
def ft_fused(wf, bf, weight):
    return FusedFT.apply(wf, bf, weight)


def forward(model, wf, bf, stm, cnt, acc=None):
    """LNNUE.forward with the per-bucket loop replaced by one matmul per layer
    over all buckets and a gather of each position's own bucket: same values,
    static shapes, no host synchronization."""
    if QAT:   # the first layer on the engine's int16 grid (src/nnue.cpp FT_WSCALE = 8192)
        w = model.ft.weight
        if QAT_FT8:   # an LNF8 net's int8 table: q << SHIFT on the x8192 grid, |q| <= 127
            k = float(2 ** (13 - QAT_FT8))
            wq = w + (torch.clamp(torch.round(w * k), -127.0, 127.0) / k - w).detach()
        else:
            wq = w + (torch.round(w * 8192.0) / 8192.0 - w).detach()
        bq = model.ft_bias + (torch.round(model.ft_bias * 8192.0) / 8192.0 - model.ft_bias).detach()
        if FT_DENSE:
            acc_w, acc_b = ft_dense(wf, wq) + bq, ft_dense(bf, wq) + bq
        else:
            acc_w = F.embedding_bag(wf, wq, mode="sum", padding_idx=NUM_INPUTS) + bq
            acc_b = F.embedding_bag(bf, wq, mode="sum", padding_idx=NUM_INPUTS) + bq
    elif acc is not None:   # NNUE_FT_DENSE=3: fwd_loss ran the fused kernels on the records
        acc_w = acc[0].float() + model.ft_bias
        acc_b = acc[1].float() + model.ft_bias
    elif FT_DENSE:
        acc_w = ft_dense(wf, model.ft.weight) + model.ft_bias
        acc_b = ft_dense(bf, model.ft.weight) + model.ft_bias
    else:
        acc_w = model.ft(wf) + model.ft_bias
        acc_b = model.ft(bf) + model.ft_bias
    s = stm.view(-1, 1) > 0.5
    h = torch.cat([torch.where(s, acc_w, acc_b), torch.where(s, acc_b, acc_w)], 1)
    h = torch.clamp(h, 0.0, 1.0).square()
    if QAT:
        return forward_tail_qat(model, h, cnt)
    nbk = model.num_buckets
    if nbk == 1:
        z = tail_act(model.l2(h))
        z = tail_act(model.l3(z))
        return model.out(z).view(-1).float() * 300.0
    n, L2, L3 = h.shape[0], model.L2, model.L3
    bucket = ((cnt - 1) * nbk // 32).clamp(0, nbk - 1).view(n, 1)
    W2 = torch.cat([m.weight for m in model.bl2]); b2 = torch.cat([m.bias for m in model.bl2])
    W3 = torch.cat([m.weight for m in model.bl3]); b3 = torch.cat([m.bias for m in model.bl3])
    Wo = torch.cat([m.weight for m in model.bout]); bo = torch.cat([m.bias for m in model.bout])
    z = F.linear(h, W2, b2).view(n, nbk, L2).gather(1, bucket.view(n, 1, 1).expand(n, 1, L2)).view(n, L2)
    z = F.linear(tail_act(z), W3, b3).view(n, nbk, L3).gather(1, bucket.view(n, 1, 1).expand(n, 1, L3)).view(n, L3)
    return F.linear(tail_act(z), Wo, bo).gather(1, bucket).view(n).float() * 300.0


def loss_terms(pred, tgt):
    """Per-position |sigmoid(pred/400) - sigmoid(target/400)|^power, plus the
    optional wide-scale term."""
    t = (torch.sigmoid(pred / LOSS_SCALE) - torch.sigmoid(tgt / LOSS_SCALE)).abs().pow(POWER)
    if WIDE:
        t = t + WIDE[1] * (torch.sigmoid(pred / WIDE[0]) - torch.sigmoid(tgt / WIDE[0])).abs().pow(POWER)
    return t


def batch_loss(pred, tgt, flag):
    t = loss_terms(pred, tgt)
    if QW is None:
        return t.mean()
    w = QW_T[flag.clamp(max=2)]
    return (t * w).sum() / w.sum()


def fwd_loss(model, rec):
    wf, bf, stm, tgt, flag, cnt = expand(rec)
    acc = ft_fused(wf, bf, model.ft.weight) if FT_DENSE == 3 else None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = forward(model, wf, bf, stm, cnt, acc)
    return batch_loss(pred.float(), tgt.clamp(-TARGET_CLIP, TARGET_CLIP), flag)


# Positions with a known large advantage for White to move. Held-out data
# shares its source's selection bias (the t91 dump dropped |score| >= 800,
# so material-up positions there are labelled near zero), so the network's
# value of material is checked against these instead of against labels.
PROBE = (("Q", "4r1k1/1p3ppp/p7/3Q4/8/1P4P1/P4P1P/3R2K1 w - - 0 30"),      # queen up
         ("R", "8/8/4k3/8/2R5/8/4K3/8 w - - 0 1"),                        # K+R vs K
         ("Rp", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),                 # rook up with pawns
         ("N", "2r2rk1/pp3ppp/2n1p3/3pP3/3P4/P1N2N2/1P3PPP/2R2RK1 w - - 0 18"))   # knight up
PIECE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}   # src/types.h Piece: WHITE_PAWN = 0 .. BLACK_KING = 11


def fen_record(fen):
    """One LCR1 record for a FEN (target 0), as luminex-featurize --compact writes it."""
    board, stm = fen.split()[:2]
    sqpc, r, f = {}, 7, 0
    for ch in board:
        if ch == "/":
            r, f = r - 1, 0
        elif ch.isdigit():
            f += int(ch)
        else:
            sqpc[r * 8 + f] = PIECE[ch]
            f += 1
    occ, nib = 0, bytearray(16)
    for k, sq in enumerate(sorted(sqpc)):
        occ |= 1 << sq
        nib[k >> 1] |= sqpc[sq] << (4 * (k & 1))
    return occ.to_bytes(8, "little") + bytes(nib) + bytes([0, 0, 1 if stm == "w" else 0, len(sqpc)]) + bytes(4)


@torch.no_grad()
def material_probe(model):
    rec = torch.tensor(list(b"".join(fen_record(f) for _, f in PROBE)), dtype=torch.uint8, device=dev).view(-1, REC)
    wf, bf, stm, _, _, cnt = expand(rec)
    p = forward(model, wf, bf, stm, cnt).float().tolist()
    return " ".join(f"{name} {v:+.0f}" for (name, _), v in zip(PROBE, p))


# ---------------------------------------------------------------------------
# Validation: the numbers that name our measured weaknesses
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val, gstep):
    if len(val) == 0:
        return
    P, T, FL = [], [], []
    for i in range(0, len(val), BS):
        wf, bf, stm, tgt, flag, cnt = expand(val[i:i + BS])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            P.append(forward(model, wf, bf, stm, cnt).float())
        T.append(tgt.clamp(-TARGET_CLIP, TARGET_CLIP))
        FL.append(flag)
    p, t, fl = torch.cat(P), torch.cat(T), torch.cat(FL)
    loss = float(batch_loss(p, t, fl))
    err = (p - t).abs()
    by_class = " ".join(f"{nm}={float(err[fl == c].mean()):.1f}" for nm, c in (("q", 0), ("c", 1), ("i", 2))
                        if bool((fl == c).any()))
    cov = float(((p - p.mean()) * (t - t.mean())).mean())
    slope, calib = cov / float(t.var()), cov / float(p.var())   # pred-on-target; target-on-pred (1 = calibrated)
    # winning side's view: how far big advantages are compressed, and whether
    # the net still orders positions inside the winning range
    sg, at = torch.sign(t), t.abs()
    bins = []
    for lo, hi in ((300, 500), (500, 700), (700, 1000), (1000, 1501)):   # empty bins are skipped
        m = (at >= lo) & (at < hi)
        if bool(m.any()):
            bins.append(f"{lo}-{hi - 1 if hi == 1501 else hi}:{float((p[m] * sg[m]).mean()):.0f}/{float(at[m].mean()):.0f}")
    win = (at >= 300)
    corr = float(torch.corrcoef(torch.stack([(p * sg)[win], at[win]]))[0, 1]) if int(win.sum()) > 100 else float("nan")
    k = min(32768, len(val))
    wf, bf, stm, _, _, _ = expand(val[:k])
    d2, s2, d3, s3, n2, n3 = model.tail_liveness(wf, bf, stm)
    print(f"[HEALTH step {gstep}] val loss={loss:.6f} MAE={float(err.mean()):.1f}cp ({by_class}) "
          f"slope={slope:.3f} calib={calib:.3f} predSTD={float(p.std()):.0f} tgtSTD={float(t.std()):.0f} | "
          f"won (pred/target) {' '.join(bins)} corr={corr:.3f} | "
          f"tail dead/stuck L2 {d2}/{s2} of {n2}, L3 {d3}/{s3} of {n3} | material (cp) {material_probe(model)}", flush=True)


@torch.no_grad()
def evaluate_disjoint(model, dval, gstep):
    """The game-disjoint set (NNUE_VAL_DATA): the training loss and errors on
    games the run never sees, the correlation where labels still order
    positions (|target| < 800), and, for quiet positions a piece, rook or queen
    up, the mean prediction against the mean label from the stronger side."""
    if dval is None or len(dval) == 0:
        return
    P, T, FL, E = [], [], [], []
    for i in range(0, len(dval), BS):
        r = dval[i:i + BS]
        wf, bf, stm, tgt, flag, cnt = expand(r)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            P.append(forward(model, wf, bf, stm, cnt).float())
        T.append(tgt.clamp(-TARGET_CLIP, TARGET_CLIP))
        FL.append(flag)
        E.append(material_edge(r) * torch.where(stm > 0.5, 1, -1))   # the side to move's edge
    p, t, fl, e = torch.cat(P), torch.cat(T), torch.cat(FL), torch.cat(E)
    loss = float(batch_loss(p, t, fl))
    err = (p - t).abs()
    by_class = " ".join(f"{nm}={float(err[fl == c].mean()):.1f}" for nm, c in (("q", 0), ("c", 1), ("i", 2))
                        if bool((fl == c).any()))
    m = t.abs() < 800
    corr = float(torch.corrcoef(torch.stack([p[m], t[m]]))[0, 1]) if int(m.sum()) > 100 else float("nan")
    mat = []
    for k, nm in ((3, "piece"), (5, "rook"), (9, "queen")):
        q = (e.abs() == k) & (fl == 0)
        if int(q.sum()) >= 100:
            s = torch.sign(e[q]).float()
            mat.append(f"{nm} {float((p[q] * s).mean()):+.0f}/{float((t[q] * s).mean()):+.0f}")
    print(f"[DISJOINT step {gstep}] loss={loss:.6f} MAE={float(err.mean()):.1f}cp ({by_class}) "
          f"corr(|t|<800)={corr:.3f} | quiet material up, pred/label: {' '.join(mat) or '-'} | "
          f"{len(dval):,} positions", flush=True)


# ---------------------------------------------------------------------------
def report_profile(prof, steps, seconds):
    """NNUE_PROFILE: GPU time per step by kernel (top 40) and by kind, against the step's wall time."""
    ev = prof.key_averages()
    def dev_self(e):
        return getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0) or 0
    kernels = [e for e in ev if str(getattr(e, 'device_type', '')).endswith('CUDA') and dev_self(e) > 0]
    kinds = (("fused FT kernels", ("_ft_fwd", "_ft_bwd")),
             ("cuBLAS / cutlass GEMM", ("gemm", "cutlass", "nvjet", "xmma", "cublas", "sm80_", "sm90_", "sm100_", "sm120_")),
             ("optimizer", ("adam", "Adam", "multi_tensor", "foreach")),
             ("copies and fills", ("Memcpy", "memcpy", "Memset", "memset", "fill", "copy_")),
             ("inductor (compiled glue)", ("triton_",)),
             ("index / gather / scatter / sort", ("index", "gather", "scatter", "sort", "cumsum", "radix", "embedding")))
    total = {k: 0.0 for k, _ in kinds}
    total["other"] = 0.0
    for e in kernels:
        kind = next((k for k, pats in kinds if any(p in e.key for p in pats)), "other")
        total[kind] += dev_self(e)
    gpu = sum(total.values())
    step_ms = seconds * 1e3 / steps
    print(f"[profile] {steps} steps of {BS:,} positions: {step_ms:.2f} ms a step (wall), GPU kernels "
          f"{gpu / 1e3 / steps:.2f} ms a step ({gpu / 1e3 / steps / step_ms:.0%} of the wall time)", flush=True)
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"[profile]   {k:34s} {v / 1e3 / steps:7.3f} ms a step  {v / max(gpu, 1e-9):6.1%}", flush=True)
    print("[profile] top kernels (GPU ms a step, launches a step, name):", flush=True)
    for e in sorted(kernels, key=lambda e: -dev_self(e))[:40]:
        print(f"[profile]   {dev_self(e) / 1e3 / steps:7.3f}  {e.count / steps:5.1f}  {e.key[:150]}", flush=True)
    path = os.path.join(OUT_DIR, OUT_BASE + "_profile.json")
    try:
        prof.export_chrome_trace(path)
        print(f"[profile] trace: {path}", flush=True)
    except Exception as e:
        print(f"[profile] trace not written ({e.__class__.__name__}: {e})", flush=True)


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    data = env("NNUE_DATA", "")
    src = env("NNUE_FENEVAL_FILE", "")
    if not data:
        if not src:
            raise SystemExit("FAIL: set NNUE_DATA (LCR1 file) or NNUE_FENEVAL_FILE")
        data = os.path.join(os.path.dirname(os.path.abspath(src)),
                            os.path.basename(src).split(".tsv")[0] + ".lcr")
    files = data_files(data)
    if not files or not all(os.path.exists(p) for p in files):
        if not src or len(files) > 1:
            raise SystemExit(f"FAIL: NNUE_DATA {data} not found and no NNUE_FENEVAL_FILE to build it from")
        build_data(src, data)
        files = [data]
    train, host, val = load_data(files)
    dval = None
    if VAL_DATA:
        vfiles = data_files(VAL_DATA)
        if not vfiles or not all(os.path.exists(p) for p in vfiles):
            raise SystemExit(f"FAIL: NNUE_VAL_DATA {VAL_DATA} not found")
        shared = set(map(os.path.realpath, vfiles)) & set(map(os.path.realpath, files))
        if shared:
            raise SystemExit(f"FAIL: NNUE_VAL_DATA shares {len(shared)} file(s) with NNUE_DATA, "
                             f"e.g. {sorted(shared)[0]}: those positions are trained on")
        dval = load_records(vfiles, VAL_MAX)
    n = len(train) + (len(host) if host is not None else 0)
    feeder = PoolFeeder(train, host, SEED, BS) if host is not None else None
    shuffle = HostShuffle(n, SEED) if feeder is None else None

    model = LNNUE(L1=L1).to(dev)
    gstep, opt_state = 0, None
    res = env("NNUE_RESUME", "")
    if res:
        if res.endswith(".nnue"):
            ck = {"model": {k: v.to(dev) for k, v in LNNUE.load_lnn1_state(res).items()}, "gstep": 0}
        else:
            ck = torch.load(res, map_location=dev, weights_only=False)
        sd = ck["model"]
        ck_l1 = sd["l2.weight"].shape[1] // 2
        if ck_l1 != L1:
            LNNUE.net2net_widen(model, sd, ck_l1)
            how = f"widened {ck_l1} -> {L1}"
        elif (sd["ft.weight"].shape != tuple(model.ft.weight.shape)
              or sd["l2.weight"].shape != tuple(model.l2.weight.shape)
              or (model.num_buckets > 1 and "bl2.0.weight" not in sd)):
            LNNUE.v2_expand(model, sd)
            how = "expanded (buckets/tail)"
        else:
            model.load_state_dict(sd, strict=False)
            gstep, opt_state = ck.get("gstep", 0), ck.get("opt")
            how = "loaded"
        with torch.no_grad():
            model.ft.weight[NUM_INPUTS].zero_()
        print(f"[RESUME] {how} from {res} at gstep {gstep}"
              f"{' with optimizer state' if opt_state else ' (optimizer fresh)'}", flush=True)
    model.probe_ft(dev)

    nbk = model.num_buckets
    tails = ([(model.bl2[b], model.bl3[b], model.bout[b]) for b in range(nbk)] if nbk > 1
             else [(model.l2, model.l3, model.out)])
    groups = [   # same groups and order as luminex_nnue_train.make_optimizer
        {"params": [model.ft.weight, model.ft_bias], "weight_decay": 0.0},
        {"params": [m.weight for tl in tails for m in tl], "weight_decay": TAIL_WD, "lr": LR * TAIL_LR_MULT},
        {"params": [m.bias for tl in tails for m in tl], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)
    if opt_state:
        opt.load_state_dict(opt_state)
        for g, mult in zip(opt.param_groups, (1.0, TAIL_LR_MULT, 1.0)):
            g["lr"] = g["initial_lr"] = LR * mult   # the scheduler would keep the checkpoint's initial_lr
    params = [p for g in opt.param_groups for p in g["params"]]

    steps_per_pass = n // BS
    t_max = T_MAX if T_MAX > 0 else steps_per_pass
    off = gstep - SCHED_START if CONTINUE else 0   # schedule steps already done by the run this checkpoint came from
    if off < 0:
        raise SystemExit(f"FAIL: NNUE_SCHED_START {SCHED_START:,} is past the checkpoint's step {gstep:,}")
    state = {"t_live": t_max}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s + off) / max(1, state["t_live"])))))

    with torch.no_grad():   # the label-convention gate of c500_train_stream.py, on held-out positions
        if res and len(val) >= 8192:
            wf, bf, stm, tgt, _, _ = expand(val[:8192])
            # clipped as in the loss: raw mate scores (up to ~26,600) would swamp a correlation
            gate = stm_split_gate(model, wf, bf, stm, tgt.clamp(-TARGET_CLIP, TARGET_CLIP))
            print(f"  [DATA] stm-split correlation {gate} (gate passed)", flush=True)
        wf, bf, stm, tgt, flag, cnt = expand(train[:BS])
        print(f"  [DATA] w {int(wf.min())}..{int(wf.max())} | b {int(bf.min())}..{int(bf.max())} | "
              f"pieces {int(cnt.min())}..{int(cnt.max())} | white to move {float(stm.mean()):.3f} | "
              f"target {float(tgt.min()):.0f}..{float(tgt.max()):.0f} std {float(tgt.std()):.0f} | "
              f"flags q/c/i {int((flag == 0).sum())}/{int((flag == 1).sum())}/{int((flag == 2).sum())}", flush=True)
    print(f"[train] {n:,} positions, {steps_per_pass:,} steps per pass of {BS:,} | horizon {t_max:,} steps "
          f"({t_max / max(1, steps_per_pass):.2f} passes){f', continuing from step {off:,}' if off else ''} | "
          f"lr {LR} tail x{TAIL_LR_MULT} (now {opt.param_groups[0]['lr']:.4g}) | "
          f"loss power {POWER}{' + wide ' + str(WIDE) if WIDE else ''} | quiet weights {QW} | "
          f"live clamp {LIVE} | compile {COMPILE} | budget {BUDGET:.0f} s", flush=True)

    step_fn = fwd_loss
    if COMPILE:
        step_fn = torch.compile(fwd_loss, dynamic=False)

    def save(reason):
        os.makedirs(OUT_DIR, exist_ok=True)
        nn_path, pt_path = os.path.join(OUT_DIR, OUT), os.path.join(OUT_DIR, OUT_BASE + ".pt")
        save_nnue(model, nn_path + ".tmp")
        os.replace(nn_path + ".tmp", nn_path)
        torch.save({"model": model.state_dict(), "gstep": gstep, "opt": opt.state_dict(), "swa": None},
                   pt_path + ".tmp")
        os.replace(pt_path + ".tmp", pt_path)
        print(f"  [save] {reason}: {nn_path} + .pt at step {gstep}", flush=True)

    def keep():
        """A numbered copy of the float net, so checkpoints along the run can be gated."""
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"{OUT_BASE}_s{gstep}.nnue")
        save_nnue(model, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"  [keep] {path} at step {gstep}", flush=True)

    evaluate(model, val, gstep)   # the starting point, for the curve
    evaluate_disjoint(model, dval, gstep)
    seg0, t0, last_t, last_s = gstep, time.time(), time.time(), gstep
    epoch, done = 0, None
    while done is None:
        perm = shuffle.next_pass() if feeder is None else None
        for i in range(0, steps_per_pass * BS, BS):
            batch = feeder.next()[0] if feeder is not None else train[HostShuffle.batch(perm, i, BS)]
            loss = step_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            sched.step()
            gstep += 1
            sess = gstep - seg0
            if PROFILE and sess == PROFILE_WARMUP:
                gpu = str(dev).startswith("cuda")
                if gpu:
                    torch.cuda.synchronize()
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]
                                              + ([torch.profiler.ProfilerActivity.CUDA] if gpu else []))
                prof.__enter__()
                p_t0 = time.time()
            elif PROFILE and sess == PROFILE_WARMUP + PROFILE:
                if gpu:
                    torch.cuda.synchronize()
                p_s = time.time() - p_t0
                prof.__exit__(None, None, None)
                report_profile(prof, PROFILE, p_s)
                done = "profile"
                break
            if gstep % 50 == 0:
                now = time.time()
                l2max = max(float(m.weight.detach().abs().max()) for m in (model.bl2 if nbk > 1 else [model.l2]))
                print(f"  e{epoch} step {gstep} loss={float(loss):.5f} L2|max|={l2max:.2f} | "
                      f"{sess * BS / 1e6:.0f}M | {(gstep - last_s) / (now - last_t):.1f} steps/s", flush=True)
                last_t, last_s = now, gstep
            if sess % 200 == 0 and BUDGET > 0:
                el = time.time() - t0
                fit = int(off + sess + sess / max(el, 1.0) * max(0.0, BUDGET - (time.time() - t_start) - 90))
                new = min(t_max, max(off + sess + 1, fit))
                if new != state["t_live"]:
                    print(f"  [sched] budget guard: horizon {state['t_live']:,} -> {new:,} steps "
                          f"({sess / max(el, 1.0):.2f} steps/s)", flush=True)
                    state["t_live"] = new
            if sess % EVAL_EVERY == 0:
                evaluate(model, val, gstep)
                evaluate_disjoint(model, dval, gstep)
            if sess % SAVE_EVERY == 0:
                save("checkpoint")
            if KEEP_EVERY > 0 and gstep % KEEP_EVERY == 0:
                keep()
            if off + sess >= state["t_live"]:
                done = "tmax"
                break
            if BUDGET > 0 and time.time() - t_start >= BUDGET:
                done = "budget"
                break
        epoch += 1
    evaluate(model, val, gstep)
    evaluate_disjoint(model, dval, gstep)
    save("final")
    print(f"DONE ({done}) - {OUT}: {gstep} steps, {(gstep - seg0) * BS:,} positions in "
          f"{time.time() - t_start:.0f} s ({(gstep - seg0) / (time.time() - t0):.2f} steps/s)", flush=True)


if __name__ == "__main__":
    main()
