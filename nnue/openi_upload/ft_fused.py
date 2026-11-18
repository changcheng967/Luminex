"""Fused feature-layer kernels for resident_train.py (Triton; FP8 needs sm_89+).

A side-row is one position seen from one side: about 30 active piece-squares, all in one
king bucket. Its feature-layer sum is a 0/1 row X times that bucket's [768, L1] block of
the table. The dense path (NNUE_FT_DENSE=2) builds X in HBM (an fp32 scatter, then a bf16
copy) and runs cuBLAS bmm over 1024-row tiles, and the copies and casts around the
matmuls cost more than the matmuls. Here X never exists in memory: each kernel builds its
X tile in registers from the side-row's 12 piece bitboards (24 int32 words, 96 bytes,
instead of 768 values), so the matmuls read only the weights or the gradient.

    ft_fwd:  A[r, :] = X[r] @ W[kb(r)] + bias           bf16 out; ranges of <= BM rows of one bucket
    ft_bwd:  dW[kb] += X[rows of kb]^T @ G[rows of kb]   fp32 atomics; ranges of <= CH rows of one bucket

Both run over side-rows sorted by bucket (plan()). FP8 variants: the forward takes W in
e4m3 with a per-column scale; the backward takes G in e4m3 with one global scale and a
power-of-two boost per row, the boost's inverse riding in X's entries (exact in e4m3).

Activation (ft_fwd(act=...), act_bwd): the forward also adds the bias, applies SCReLU and
writes each side-row's values straight to its half of the next layer's input h [n, 2 * L1]
(side to move first, act_dst()), so the sums never go back to side-row order and no
elementwise pass reads them. The values are the trainer's bit for bit: x = (the stored sum)
+ bias in fp32, h = clamp(x, 0, 1)^2 rounded to bf16 as autocast rounds it for the matmul.
act_bwd() is autograd's backward of the same ops: G = where(0 <= x <= 1, dL/dh * 2 *
clamp(x, 0, 1), 0) in fp32, rounded to bf16 in sorted order for ft_bwd, and the bias
gradient from the fp32 G (the one difference: its sum runs in another order).

Exact forward (QAT): a weight on the engine's int16 grid, k / 2^13 with |k| < 2^15, is the
sum of two bf16 values, hi = bf16(w) and lo = w - hi (|lo| <= 64 / 2^13: at most 7 significant
bits), split_bf16(). The forward then runs two bf16 dots into one fp32 accumulator and stores
fp32: every product and partial sum is a multiple of 2^-13 below 2^7, which fp32 holds exactly,
so the output is the engine's integer sum whatever the order of the additions.
"""
import os

import torch
import triton
import triton.language as tl

# Triton's CPU interpreter (TRITON_INTERPRET=1, check_lanes_cpu.py) computes bf16 dots wrong (3.8.0: a 0/1 x bf16
# product off by 2e10 where the answer is 3.6); under it the kernels run their dots in fp32, the same values.
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"

PSQ = 768   # piece-square features per king bucket
NKB = 32    # king buckets


def side_rows(wf, bf, num_inputs):
    """[n, 32] feature indices per side (num_inputs marks an empty slot) -> planes [2n, 24]
    int32 (plane p's squares 0-31 in word 2p, 32-63 in word 2p+1) and kb [2n] int64.
    Rows 0..n-1 are White's view, n..2n-1 Black's."""
    feats = torch.cat([wf, bf])
    valid = feats < num_inputs
    kb = feats[:, 0] // PSQ
    local = feats % PSQ
    bits = torch.where(valid, torch.ones_like(local) << (local % 64), torch.zeros_like(local))
    planes = torch.zeros(feats.shape[0], 12, dtype=torch.int64, device=feats.device)
    planes.scatter_add_(1, torch.where(valid, local // 64, 0), bits)   # distinct bits: the sum is the OR
    return planes.view(torch.int32), kb


@triton.jit
def _lsb_index(x):
    """Index of the lowest set bit of each int64 (x != 0): the exponent of x & -x as an fp64."""
    f = (x & (-x)).to(tl.float64).to(tl.int64, bitcast=True)
    return ((f >> 52) & 0x7FF) - 1023


@triton.jit
def _hflip(x):   # square ^ 7 on every bit: reverse the bits of each byte
    x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1)
    x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2)
    return ((x >> 4) & 0x0F0F0F0F0F0F0F0F) | ((x & 0x0F0F0F0F0F0F0F0F) << 4)


@triton.jit
def _vflip(x):   # square ^ 56 on every bit: reverse the bytes
    x = ((x >> 8) & 0x00FF00FF00FF00FF) | ((x & 0x00FF00FF00FF00FF) << 8)
    x = ((x >> 16) & 0x0000FFFF0000FFFF) | ((x & 0x0000FFFF0000FFFF) << 16)
    return ((x >> 32) & 0xFFFFFFFF) | (x << 32)


@triton.jit
def _rec_bitboards(rec32, kbt, planes, kbo, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pos = pid * BLOCK + tl.arange(0, BLOCK)
    m = pos < n
    p = pos.to(tl.int64)
    lo = tl.load(rec32 + p * 8, mask=m, other=0).to(tl.int64) & 0xFFFFFFFF
    hi = tl.load(rec32 + p * 8 + 1, mask=m, other=0).to(tl.int64)
    occ = lo | (hi << 32)
    q = tl.arange(0, 16)
    bw = tl.zeros((BLOCK, 16), dtype=tl.int64)   # White's planes 2 * type + colour, squares unoriented
    bb = tl.zeros((BLOCK, 16), dtype=tl.int64)   # Black's planes 2 * type + 1 - colour
    for k in tl.static_range(4):                 # nibble words: pieces 8k .. 8k+7 in square order
        nw = tl.load(rec32 + p * 8 + 2 + k, mask=m, other=0)
        for i in tl.static_range(8):
            code = (nw >> (4 * i)) & 15
            have = occ != 0
            sq = _lsb_index(occ)
            bit = tl.where(have, tl.full((BLOCK,), 1, tl.int64) << (sq & 63), 0)
            occ = occ & (occ - 1)
            pt = code % 6
            colour = code // 6
            bw = bw | tl.where(q[None, :] == (2 * pt + colour)[:, None], bit[:, None], 0)
            bb = bb | tl.where(q[None, :] == (2 * pt + 1 - colour)[:, None], bit[:, None], 0)
    wk = _lsb_index(tl.sum(tl.where(q[None, :] == 10, bw, 0), axis=1))   # White king: White's plane 10
    bk = _lsb_index(tl.sum(tl.where(q[None, :] == 10, bb, 0), axis=1))   # Black king: Black's plane 10
    mw = (wk & 7) < 4
    mb = (bk & 7) < 4
    bw = tl.where(mw[:, None], _hflip(bw), bw)
    bb = _vflip(tl.where(mb[:, None], _hflip(bb), bb))
    kw = tl.load(kbt + (tl.where(mw, 7, 0) ^ wk), mask=m, other=0)
    kbb = tl.load(kbt + ((tl.where(mb, 7, 0) ^ 56) ^ bk), mask=m, other=0)
    cm = m[:, None] & (q[None, :] < 12)
    tl.store(planes + p[:, None] * 12 + q[None, :], bw, mask=cm)
    tl.store(planes + (p + n)[:, None] * 12 + q[None, :], bb, mask=cm)
    tl.store(kbo + p, kw, mask=m)
    tl.store(kbo + p + n, kbb, mask=m)


def rec_bitboards(rec, kb_table, BLOCK=128):
    """LCR1 records [n, 32] uint8 -> (planes [2n, 24] int32, kb [2n] int64), the same as
    side_rows(*expand(rec)[:2]) but straight from the record, without the feature indices.
    kb_table: resident_train.KB_T (oriented king square -> bucket, int64 [64])."""
    n = rec.shape[0]
    planes = torch.empty(2 * n, 12, dtype=torch.int64, device=rec.device)
    kb = torch.empty(2 * n, dtype=torch.int64, device=rec.device)
    _rec_bitboards[(triton.cdiv(n, BLOCK),)](rec.view(torch.int32), kb_table, planes, kb, n, BLOCK=BLOCK)
    return planes.view(torch.int32), kb


def plan(kb, block, nkb=NKB):
    """Sort side-rows by bucket and cut each bucket into ranges of at most `block` rows.
    Returns order [R] (sorted slot -> side-row; the same for every block, the sort is
    stable) and ranges [R // block + nkb, 3] int32 (start, end, bucket), unused ranges
    empty, so every shape is static and nothing syncs with the host."""
    R, dv = kb.shape[0], kb.device
    order = torch.argsort(kb, stable=True)
    counts = torch.bincount(kb, minlength=nkb)
    starts = torch.cumsum(counts, 0) - counts
    nr = (counts + block - 1) // block
    rend = torch.cumsum(nr, 0)
    t = torch.arange(R // block + nkb, device=dv)
    b = torch.searchsorted(rend, t, right=True)
    ok = b < nkb
    bc = b.clamp(max=nkb - 1)
    s = starts[bc] + (t - (rend[bc] - nr[bc])) * block
    e = torch.minimum(s + block, starts[bc] + counts[bc])
    ranges = torch.stack([torch.where(ok, s, 0), torch.where(ok, e, 0), bc], 1)
    return order, ranges.to(torch.int32).contiguous()


@triton.jit
def _lanes(lane, order, rows, rmask, cols, T: tl.constexpr):
    """Each sorted side-row's extra lane sums on lanes 0..T-1 (lane [R, T] fp32 in side-row order, read through
    the sort), 0 on the other lanes: the tile's [BM, BN] share, and the mask of the lanes it covers."""
    orow = tl.load(order + rows, mask=rmask, other=0)
    m = rmask[:, None] & (cols < T)[None, :]
    return tl.load(lane + orow[:, None] * T + cols[None, :], mask=m, other=0.0), orow, m


@triton.jit
def _ft_fwd(planes, ranges, w, bias, wscale, out, wlo, dst, h, lane, order,
            L1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr, SPLIT: tl.constexpr,
            ACT: tl.constexpr, T: tl.constexpr, INTERP: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    r0 = tl.load(ranges + 3 * t)
    r1 = tl.load(ranges + 3 * t + 1)
    kb = tl.load(ranges + 3 * t + 2).to(tl.int64)
    if r0 >= r1:
        return
    rows = (r0 + tl.arange(0, BM)).to(tl.int64)
    rmask = rows < r1
    cols = nb * BN + tl.arange(0, BN)
    j = tl.arange(0, 64)
    sh = j & 31
    upper = j >= 32
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wrow = w + kb * (768 * L1)
    wrow_lo = wlo + kb * (768 * L1)
    for p in range(12):
        lo = tl.load(planes + rows * 24 + 2 * p, mask=rmask, other=0)
        hi = tl.load(planes + rows * 24 + 2 * p + 1, mask=rmask, other=0)
        word = tl.where(upper[None, :], hi[:, None], lo[:, None])
        bit = ((word >> sh[None, :]) & 1).to(tl.float32)
        k = p * 64 + j
        wt = tl.load(wrow + k[:, None] * L1 + cols[None, :])
        if FP8:
            acc = tl.dot(bit.to(tl.float8e4nv), wt, acc)
        elif INTERP:   # Triton's CPU interpreter gets bf16 dots wrong (3.8.0); the same values in fp32 there
            acc = tl.dot(bit, wt.to(tl.float32), acc)
            if SPLIT:
                acc = tl.dot(bit, tl.load(wrow_lo + k[:, None] * L1 + cols[None, :]).to(tl.float32), acc)
        else:
            acc = tl.dot(bit.to(tl.bfloat16), wt, acc)
            if SPLIT:   # the rest of each weight: hi + lo is exact (split_bf16)
                acc = tl.dot(bit.to(tl.bfloat16), tl.load(wrow_lo + k[:, None] * L1 + cols[None, :]), acc)
    if FP8:
        acc = acc * tl.load(wscale + cols)[None, :]
    if ACT:   # the sums without the bias (act_bwd's x), then bias + SCReLU into each row's half of h
        if SPLIT:
            s = acc
        else:
            s = acc.to(tl.bfloat16)
        tl.store(out + rows[:, None] * L1 + cols[None, :], s, mask=rmask[:, None])
        x = s.to(tl.float32) + tl.load(bias + cols)[None, :]
        if T > 0:   # extra lane sums (the relation rows) on lanes 0..T-1, after the bias as forward() adds them
            if nb * BN < T:
                xl, orow, lm = _lanes(lane, order, rows, rmask, cols, T)
                x = x + xl
        c = tl.minimum(tl.maximum(x, 0.0), 1.0)
        y = c * c
        hp = h + tl.load(dst + rows, mask=rmask, other=0)[:, None] + cols[None, :]
        if SPLIT:
            tl.store(hp, y, mask=rmask[:, None])
        else:
            tl.store(hp, y.to(tl.bfloat16), mask=rmask[:, None])
    else:
        acc = acc + tl.load(bias + cols)[None, :]
        if SPLIT:
            tl.store(out + rows[:, None] * L1 + cols[None, :], acc, mask=rmask[:, None])
        else:
            tl.store(out + rows[:, None] * L1 + cols[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _ft_bwd(planes, ranges, g, gscale, dw,
            L1: tl.constexpr, BMF: tl.constexpr, NW: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            FP8: tl.constexpr, INTERP: tl.constexpr):
    t = tl.program_id(0)
    fb = tl.program_id(1)
    nb = tl.program_id(2)
    r0 = tl.load(ranges + 3 * t)
    r1 = tl.load(ranges + 3 * t + 1)
    kb = tl.load(ranges + 3 * t + 2).to(tl.int64)
    if r0 >= r1:
        return
    fl = tl.arange(0, BMF)   # feature within this block: word fl // 32, bit fl % 32
    wl = fl // 32
    sh = fl & 31
    cols = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((BMF, BN), dtype=tl.float32)
    for r in range(r0, r1, BK):
        rows = (r + tl.arange(0, BK)).to(tl.int64)
        rmask = rows < r1
        # only the block's NW distinct words per row are loaded ([BK] each), then spread over
        # the 32 features each one holds: no [BMF, BK] gather (it repeated every word 32 times)
        word = tl.zeros((BMF, BK), dtype=tl.int32)
        for i in tl.static_range(NW):
            wi = tl.load(planes + rows * 24 + (fb * NW + i), mask=rmask, other=0)
            word = tl.where((wl == i)[:, None], wi[None, :], word)
        bit = ((word >> sh[:, None]) & 1).to(tl.float32)              # X^T tile [BMF, BK]
        gt = tl.load(g + rows[:, None] * L1 + cols[None, :], mask=rmask[:, None], other=0.0)
        if FP8:
            s = tl.load(gscale + rows, mask=rmask, other=0.0)
            acc = tl.dot((bit * s[None, :]).to(tl.float8e4nv), gt, acc)
        elif INTERP:   # the interpreter's bf16 dot, as in _ft_fwd
            acc = tl.dot(bit, gt.to(tl.float32), acc)
        else:
            acc = tl.dot(bit.to(tl.bfloat16), gt, acc)
    dst = dw + (kb * 768 + fb * BMF + fl)[:, None] * L1 + cols[None, :]
    tl.atomic_add(dst, acc, sem="relaxed")


@triton.jit
def _act_bwd(g, g2, sums, dst, bias, gs, part, lane, order, glane, R,
             L1: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, TWO: tl.constexpr, T: tl.constexpr):
    pb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = (pb * BR + tl.arange(0, BR)).to(tl.int64)
    rmask = rows < R
    cols = cb * BC + tl.arange(0, BC)
    gp = tl.load(dst + rows, mask=rmask, other=0)[:, None] + cols[None, :]
    gv = tl.load(g + gp, mask=rmask[:, None], other=0.0).to(tl.float32)
    if TWO:   # a second consumer's gradient, added in fp32 as autograd added the two
        gv = gv + tl.load(g2 + gp, mask=rmask[:, None], other=0.0).to(tl.float32)
    sp = rows[:, None] * L1 + cols[None, :]
    x = tl.load(sums + sp, mask=rmask[:, None], other=0.0).to(tl.float32) + tl.load(bias + cols)[None, :]
    if T > 0:   # the forward's x: the extra lane sums on lanes 0..T-1 (0 elsewhere, masked off)
        xl, orow, lm = _lanes(lane, order, rows, rmask, cols, T)
        x = x + xl
    c = tl.minimum(tl.maximum(x, 0.0), 1.0)
    gx = tl.where((x >= 0.0) & (x <= 1.0) & rmask[:, None], gv * 2.0 * c, 0.0)   # clamp's mask is inclusive
    tl.store(gs + sp, gx.to(tl.bfloat16), mask=rmask[:, None])
    if T > 0:   # the lane sums' gradient is the sums' gradient on their lanes, fp32, in side-row order
        tl.store(glane + orow[:, None] * T + cols[None, :], gx, mask=lm)
    tl.store(part + pb * L1 + cols, tl.sum(gx, axis=0))


def lane_args(lanes, R, L1):
    """lanes = (sums [R, T] fp32 in side-row order, order from plan()) or None -> the kernels' (lane, order, T)."""
    if lanes is None:
        return None, None, 0
    lane, order = lanes
    if lane.dim() != 2 or lane.shape[0] != R or order.shape[0] != R or not 0 < lane.shape[1] <= L1:
        raise ValueError(f"lanes: {tuple(lane.shape)} sums and {order.shape[0]} sort slots for {R} side-rows of {L1}")
    if lane.dtype != torch.float32:
        raise ValueError(f"lanes: the sums must be fp32, not {lane.dtype}")
    return lane.contiguous(), order.contiguous(), lane.shape[1]


def ft_fwd(planes_s, ranges, w, bias, BM, BN, fp8=False, wscale=None, num_warps=8, num_stages=3, out=None,
           w_lo=None, act=None, lanes=None):
    """planes_s: side-rows in sorted order; ranges from plan(kb, BM). w: [32 * 768, L1]
    bf16 (or e4m3 with wscale [L1] fp32 = 1 / column scale). Returns [R, L1] bf16 in sorted order.
    With w_lo (bf16, split_bf16's second part), the table is w + w_lo and the output fp32: exact
    for weights on the int16 grid.
    With act = (dst, n) (act_dst()), the activation too: returns (sums, h), the sums as above but
    without the bias, and h [n, 2 * L1] = clamp(sums + bias, 0, 1)^2, each side-row's values at
    offset dst[r] (bf16; fp32 with w_lo).
    With act, lanes = (sums [R, T] fp32 in side-row order, order from plan()) adds extra sums on lanes
    0..T-1 after the bias (the relation inputs): h = clamp(sums + bias + lane sums, 0, 1)^2, the
    returned sums still without either."""
    R, L1 = planes_s.shape[0], w.shape[1]
    split = w_lo is not None
    if split and fp8:
        raise ValueError("the exact (split) forward is bf16 only")
    if lanes is not None and act is None:
        raise ValueError("lanes: only with act (the sums are returned without them)")
    lane, order, T = lane_args(lanes, R, L1)
    if out is None:
        out = torch.empty(R, L1, dtype=torch.float32 if split else torch.bfloat16, device=planes_s.device)
    h = out
    if act is not None:
        dst, n = act
        if R != 2 * n or dst.shape[0] != R:
            raise ValueError(f"act: {R} side-rows, {dst.shape[0]} offsets for {n} positions")
        h = torch.empty(n, 2 * L1, dtype=out.dtype, device=planes_s.device)
    grid = (ranges.shape[0], L1 // BN)
    _ft_fwd[grid](planes_s, ranges, w, bias, wscale if fp8 else bias, out, w_lo if split else w,
                  act[0] if act is not None else ranges, h,
                  lane if T else bias, order if T else ranges,
                  L1=L1, BM=BM, BN=BN, FP8=fp8, SPLIT=split, ACT=act is not None, T=T, INTERP=INTERP,
                  num_warps=num_warps, num_stages=num_stages)
    return out if act is None else (out, h)


def act_dst(order, stm, n, L1):
    """Where each sorted side-row's L1 activations start in h [n, 2 * L1]: its position's row,
    the first half for the side to move (stm [n], 1 = White) and the second for the other, as
    the trainer's torch.cat([where(s, white, black), where(s, black, white)], 1) lays them out.
    order from plan(): side-rows 0..n-1 are White's views, n..2n-1 Black's."""
    black = order >= n
    p = order - black.long() * n
    half = torch.where(stm[p] > 0.5, black, ~black)
    return (p * (2 * L1) + half.long() * L1).contiguous()


def act_bwd(g, g2, sums, dst, bias, BR=32, BC=256, num_warps=4, lanes=None):
    """Backward of ft_fwd(act=...). g [n, 2 * L1] = dL/dh (bf16 or fp32), g2 = a second
    consumer's dL/dh or None, sums, bias and lanes as ft_fwd took them. Returns (gs, db, gl): gs
    [R, L1] bf16 in sorted order for ft_bwd, the gradient of the sums, db [L1] fp32, the bias's,
    and gl [R, T] fp32 in side-row order, the lane sums' (None without lanes)."""
    R, L1 = sums.shape
    BC = min(BC, L1 & -L1)   # a power of two that divides L1
    lane, order, T = lane_args(lanes, R, L1)
    g = g.contiguous()
    g2 = g2.contiguous() if g2 is not None else None
    gs = torch.empty(R, L1, dtype=torch.bfloat16, device=sums.device)
    gl = torch.empty(R, T, dtype=torch.float32, device=sums.device) if T else None   # every row is written once
    nrb = triton.cdiv(R, BR)
    part = torch.empty(nrb, L1, dtype=torch.float32, device=sums.device)
    _act_bwd[(nrb, L1 // BC)](g, g2 if g2 is not None else g, sums, dst, bias, gs, part,
                              lane if T else bias, order if T else dst, gl if T else part, R,
                              L1=L1, BR=BR, BC=BC, TWO=g2 is not None, T=T, num_warps=num_warps)
    return gs, part.sum(0), gl


def split_bf16(w):
    """w [NI, L1] fp32 -> (hi, lo) bf16 with hi + lo == w exactly for weights on the int16
    grid (k / 2^13, |k| < 2^15): hi = w rounded to bf16, lo = what is left (<= 7 bits)."""
    hi = w.to(torch.bfloat16)
    return hi.contiguous(), (w - hi.float()).to(torch.bfloat16).contiguous()


def ft_bwd(planes_s, ranges, g_s, dw, BMF, BN, BK, fp8=False, gscale=None, num_warps=8, num_stages=3):
    """Adds X^T G into dw [32 * 768, L1] fp32 (zero it first). g_s: [R, L1] bf16 in sorted
    order (or e4m3 with gscale [R] fp32, see fp8_rows); ranges from plan(kb, CH)."""
    L1 = g_s.shape[1]
    grid = (ranges.shape[0], 768 // BMF, L1 // BN)
    _ft_bwd[grid](planes_s, ranges, g_s, gscale if fp8 else g_s, dw,
                  L1=L1, BMF=BMF, NW=BMF // 32, BN=BN, BK=BK, FP8=fp8, INTERP=INTERP,
                  num_warps=num_warps, num_stages=num_stages)


def fp8_cols(w):
    """w [NI, L1] -> (e4m3 copy scaled per column to +-448, dequant [L1] fp32)."""
    m = w.abs().amax(0).float().clamp_min(1e-30)
    sc = 448.0 / m
    return (w.float() * sc[None, :]).to(torch.float8_e4m3fn).contiguous(), (1.0 / sc).contiguous()


def fp8_rows(g):
    """g [R, L1] -> (g8 e4m3, s [R] fp32, inv): X^T g = inv * X^T diag(s) g8. One global
    scale puts the largest |g| at 448; each row is then boosted by 2^d (0 <= d <= 9) so its
    own max sits in [224, 448]; s = 2^-d is exact in e4m3 (down to its 2^-9 subnormal)."""
    m = g.abs().amax(1).float()
    S = 448.0 / m.max().clamp_min(1e-30)
    d = torch.floor(torch.log2(448.0 / (m * S).clamp_min(1e-30))).clamp(0, 9)
    boost = torch.exp2(d)
    g8 = (g.float() * (S * boost)[:, None]).to(torch.float8_e4m3fn).contiguous()
    return g8, (1.0 / boost).float().contiguous(), float(1.0 / S)
