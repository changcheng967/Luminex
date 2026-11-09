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
  NNUE_LIVE_CLAMP [0]  tail activations pass this gradient slope outside [0, 1];
                     the forward pass (and so the engine's net) is unchanged
  NNUE_TARGET_CLIP [1500]  |target| cap inside the loss (mates and big scores stay
                     in the data and train as "winning"; 1500 = 97.7% in the loss)
  NNUE_LOSS_WIDE ["" ]  "scale,weight": add weight * the same loss at a wider
                     sigmoid scale, which keeps a gradient in won positions
  NNUE_OUT_DIR [.]  NNUE_OUT_NAME [luminex_resident.nnue]
"""
import math
import os
import re
import subprocess
import sys
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
VAL_DATA = env("NNUE_VAL_DATA", "")
VAL_MAX = int(float(env("NNUE_VAL_MAX", "5e6")))
EVAL_EVERY = int(env("NNUE_EVAL_EVERY", "1000"))
SAVE_EVERY = int(env("NNUE_SAVE_EVERY", "3000"))
KEEP_EVERY = int(env("NNUE_KEEP_EVERY", "0"))
CONTINUE = env("NNUE_CONTINUE", "0") == "1"
COMPILE = env("NNUE_COMPILE", "0") == "1"
LIVE = float(env("NNUE_LIVE_CLAMP", "0"))
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
    """LCR1 files -> (train, val) uint8 [n, 32] on the GPU. Positions whose hash
    field falls below VAL_FRAC * 2^32 form the validation set; the rest are packed
    into one training tensor, chunk by chunk, so the GPU never holds a second copy.
    NNUE_MAX_POS caps the positions loaded (the files' order is kept)."""
    sizes = []
    for p in paths:
        s = os.path.getsize(p)
        if s % REC:
            raise SystemExit(f"FAIL: {p} is {s} bytes, not a whole number of {REC}-byte records")
        sizes.append(s // REC)
    n = sum(sizes)
    cap = int(float(env("NNUE_MAX_POS", "0"))) or n
    n = min(n, cap)
    thr = np.uint32(min(int(VAL_FRAC * 2 ** 32), 2 ** 32 - 1))
    train = torch.empty((n, REC), dtype=torch.uint8, device=dev)
    vals, nt, nread, t = [], 0, 0, time.time()
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
                train[nt:nt + len(keep)].copy_(torch.from_numpy(keep))
                nt += len(keep)
        if nread >= n:
            break
    val = torch.from_numpy(np.concatenate(vals)).to(dev) if vals else train[:0]
    print(f"[data] {nread:,} positions from {len(paths)} file(s) -> VRAM in {time.time() - t:.0f} s: "
          f"train {nt:,}, validation {len(val):,} ({VAL_FRAC:.2%} by hash), {nread * REC / 1e9:.1f} GB"
          f"{f' (capped at NNUE_MAX_POS {cap:,})' if cap < sum(sizes) else ''}", flush=True)
    return train[:nt], val


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


def forward(model, wf, bf, stm, cnt):
    """LNNUE.forward with the per-bucket loop replaced by one matmul per layer
    over all buckets and a gather of each position's own bucket: same values,
    static shapes, no host synchronization."""
    acc_w = model.ft(wf) + model.ft_bias
    acc_b = model.ft(bf) + model.ft_bias
    s = stm.view(-1, 1) > 0.5
    h = torch.cat([torch.where(s, acc_w, acc_b), torch.where(s, acc_b, acc_w)], 1)
    h = torch.clamp(h, 0.0, 1.0).square()
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
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = forward(model, wf, bf, stm, cnt)
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
    train, val = load_data(files)
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
    n = len(train)
    shuffle = HostShuffle(n, SEED)

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
    off = gstep if CONTINUE else 0   # schedule steps already done by the run this checkpoint came from
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
        perm = shuffle.next_pass()
        for i in range(0, steps_per_pass * BS, BS):
            loss = step_fn(model, train[HostShuffle.batch(perm, i, BS)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            sched.step()
            gstep += 1
            sess = gstep - seg0
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
