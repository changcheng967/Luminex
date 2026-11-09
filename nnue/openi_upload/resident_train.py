#!/usr/bin/env python3
"""GPU-resident NNUE trainer for one large-memory CUDA GPU.

The whole dataset sits in VRAM as 32-byte LCR1 records, written by
`luminex-featurize --fen-eval --compact`. Each batch is expanded to
HalfKAv2_hm feature indices on the GPU, so there is no featurizing and no pipe
traffic while the network trains, and no host synchronization inside a step.
Network, loss, optimizer groups, learning-rate schedule and output files are
those of c500_train_stream.py: a .pt from either trainer resumes in the other.

Environment (default in brackets):
  NNUE_DATA          LCR1 file; built from NNUE_FENEVAL_FILE first when missing
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
  NNUE_EVAL_EVERY [1000] steps between [HEALTH] validation lines
  NNUE_SAVE_EVERY [3000] steps between checkpoint saves
  NNUE_COMPILE [0]   1: torch.compile the step from the expanded batch to the loss
  NNUE_LIVE_CLAMP [0]  tail activations pass this gradient slope outside [0, 1];
                     the forward pass (and so the engine's net) is unchanged
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
from luminex_nnue_train import (LNNUE, LOSS_SCALE, NUM_INPUTS, TARGET_CLIP,  # noqa: E402
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
EVAL_EVERY = int(env("NNUE_EVAL_EVERY", "1000"))
SAVE_EVERY = int(env("NNUE_SAVE_EVERY", "3000"))
COMPILE = env("NNUE_COMPILE", "0") == "1"
LIVE = float(env("NNUE_LIVE_CLAMP", "0"))
WIDE = [float(x) for x in env("NNUE_LOSS_WIDE", "").split(",")] if env("NNUE_LOSS_WIDE", "") else None
OUT_DIR = env("NNUE_OUT_DIR", ".")
OUT = env("NNUE_OUT_NAME", "luminex_resident.nnue")
OUT_BASE = OUT[:-5] if OUT.endswith(".nnue") else OUT
dev = "cuda"


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


def load_data(path):
    """LCR1 file -> (train, val) uint8 [n, 32] on the GPU. Positions whose FEN hash
    falls below VAL_FRAC * 2^32 form the validation set; the rest are packed into
    one training tensor, chunk by chunk, so the GPU never holds a second copy."""
    size = os.path.getsize(path)
    if size % REC:
        raise SystemExit(f"FAIL: {path} is {size} bytes, not a whole number of {REC}-byte records")
    n = size // REC
    thr = np.uint32(min(int(VAL_FRAC * 2 ** 32), 2 ** 32 - 1))
    train = torch.empty((n, REC), dtype=torch.uint8, device=dev)
    vals, nt, t = [], 0, time.time()
    chunk = 1 << 25   # records per read: 1 GiB
    with open(path, "rb") as f:
        while True:
            a = np.fromfile(f, dtype=np.uint8, count=chunk * REC)
            if a.size == 0:
                break
            a = a.reshape(-1, REC)
            held = a[:, 28:32].copy().view(np.uint32).reshape(-1) < thr
            if held.any():
                vals.append(a[held])
            keep = a[~held]
            train[nt:nt + len(keep)].copy_(torch.from_numpy(keep))
            nt += len(keep)
    val = torch.from_numpy(np.concatenate(vals)).to(dev) if vals else train[:0]
    print(f"[data] {n:,} positions -> VRAM in {time.time() - t:.0f} s: train {nt:,}, "
          f"validation {len(val):,} ({VAL_FRAC:.2%} by FEN hash), {n * REC / 1e9:.1f} GB", flush=True)
    return train[:nt], val


KB = [-1, -1, -1, -1, 31, 30, 29, 28, -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20, -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12, -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4, -1, -1, -1, -1, 3, 2, 1, 0]
KB_T = torch.tensor(KB, dtype=torch.int64, device=dev).clamp(min=0)
SQ64 = torch.arange(64, dtype=torch.int64, device=dev)
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
    if not os.path.exists(data):
        if not src:
            raise SystemExit(f"FAIL: {data} missing and no NNUE_FENEVAL_FILE to build it from")
        build_data(src, data)
    train, val = load_data(data)
    n = len(train)

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
            g["lr"] = LR * mult
    params = [p for g in opt.param_groups for p in g["params"]]

    steps_per_pass = n // BS
    t_max = T_MAX if T_MAX > 0 else steps_per_pass
    state = {"t_live": t_max}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(1.0, s / max(1, state["t_live"])))))

    with torch.no_grad():   # the label-convention gate of c500_train_stream.py, on held-out positions
        if res and len(val) >= 8192:
            wf, bf, stm, tgt, _, _ = expand(val[:8192])
            print(f"  [DATA] stm-split correlation {stm_split_gate(model, wf, bf, stm, tgt)} (gate passed)", flush=True)
        wf, bf, stm, tgt, flag, cnt = expand(train[:BS])
        print(f"  [DATA] w {int(wf.min())}..{int(wf.max())} | b {int(bf.min())}..{int(bf.max())} | "
              f"pieces {int(cnt.min())}..{int(cnt.max())} | white to move {float(stm.mean()):.3f} | "
              f"target {float(tgt.min()):.0f}..{float(tgt.max()):.0f} std {float(tgt.std()):.0f} | "
              f"flags q/c/i {int((flag == 0).sum())}/{int((flag == 1).sum())}/{int((flag == 2).sum())}", flush=True)
    print(f"[train] {n:,} positions, {steps_per_pass:,} steps per pass of {BS:,} | horizon {t_max:,} steps "
          f"({t_max / max(1, steps_per_pass):.2f} passes) | lr {LR} tail x{TAIL_LR_MULT} | "
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

    evaluate(model, val, gstep)   # the starting point, for the curve
    seg0, t0, last_t, last_s = gstep, time.time(), time.time(), gstep
    epoch, done = 0, None
    while done is None:
        perm = torch.randperm(n, device=dev)
        for i in range(0, steps_per_pass * BS, BS):
            loss = step_fn(model, train[perm[i:i + BS]])
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
                fit = int(sess + sess / max(el, 1.0) * max(0.0, BUDGET - (time.time() - t_start) - 90))
                new = min(t_max, max(sess + 1, fit))
                if new != state["t_live"]:
                    print(f"  [sched] budget guard: horizon {state['t_live']:,} -> {new:,} steps "
                          f"({sess / max(el, 1.0):.2f} steps/s)", flush=True)
                    state["t_live"] = new
            if sess % EVAL_EVERY == 0:
                evaluate(model, val, gstep)
            if sess % SAVE_EVERY == 0:
                save("checkpoint")
            if sess >= state["t_live"]:
                done = "tmax"
                break
            if BUDGET > 0 and time.time() - t_start >= BUDGET:
                done = "budget"
                break
        epoch += 1
    evaluate(model, val, gstep)
    save("final")
    print(f"DONE ({done}) - {OUT}: {gstep} steps, {(gstep - seg0) * BS:,} positions in "
          f"{time.time() - t_start:.0f} s ({(gstep - seg0) / (time.time() - t0):.2f} steps/s)", flush=True)


if __name__ == "__main__":
    main()
