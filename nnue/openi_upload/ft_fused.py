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
"""
import torch
import triton
import triton.language as tl

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
def _ft_fwd(planes, ranges, w, bias, wscale, out,
            L1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr):
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
    for p in range(12):
        lo = tl.load(planes + rows * 24 + 2 * p, mask=rmask, other=0)
        hi = tl.load(planes + rows * 24 + 2 * p + 1, mask=rmask, other=0)
        word = tl.where(upper[None, :], hi[:, None], lo[:, None])
        bit = ((word >> sh[None, :]) & 1).to(tl.float32)
        k = p * 64 + j
        wt = tl.load(wrow + k[:, None] * L1 + cols[None, :])
        if FP8:
            acc = tl.dot(bit.to(tl.float8e4nv), wt, acc)
        else:
            acc = tl.dot(bit.to(tl.bfloat16), wt, acc)
    if FP8:
        acc = acc * tl.load(wscale + cols)[None, :]
    acc = acc + tl.load(bias + cols)[None, :]
    tl.store(out + rows[:, None] * L1 + cols[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _ft_bwd(planes, ranges, g, gscale, dw,
            L1: tl.constexpr, BMF: tl.constexpr, NW: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            FP8: tl.constexpr):
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
        else:
            acc = tl.dot(bit.to(tl.bfloat16), gt, acc)
    dst = dw + (kb * 768 + fb * BMF + fl)[:, None] * L1 + cols[None, :]
    tl.atomic_add(dst, acc, sem="relaxed")


def ft_fwd(planes_s, ranges, w, bias, BM, BN, fp8=False, wscale=None, num_warps=8, num_stages=3, out=None):
    """planes_s: side-rows in sorted order; ranges from plan(kb, BM). w: [32 * 768, L1]
    bf16 (or e4m3 with wscale [L1] fp32 = 1 / column scale). Returns [R, L1] bf16 in sorted order."""
    R, L1 = planes_s.shape[0], w.shape[1]
    if out is None:
        out = torch.empty(R, L1, dtype=torch.bfloat16, device=planes_s.device)
    grid = (ranges.shape[0], L1 // BN)
    _ft_fwd[grid](planes_s, ranges, w, bias, wscale if fp8 else bias, out,
                  L1=L1, BM=BM, BN=BN, FP8=fp8, num_warps=num_warps, num_stages=num_stages)
    return out


def ft_bwd(planes_s, ranges, g_s, dw, BMF, BN, BK, fp8=False, gscale=None, num_warps=8, num_stages=3):
    """Adds X^T G into dw [32 * 768, L1] fp32 (zero it first). g_s: [R, L1] bf16 in sorted
    order (or e4m3 with gscale [R] fp32, see fp8_rows); ranges from plan(kb, CH)."""
    L1 = g_s.shape[1]
    grid = (ranges.shape[0], 768 // BMF, L1 // BN)
    _ft_bwd[grid](planes_s, ranges, g_s, gscale if fp8 else g_s, dw,
                  L1=L1, BMF=BMF, NW=BMF // 32, BN=BN, BK=BK, FP8=fp8, num_warps=num_warps, num_stages=num_stages)


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
