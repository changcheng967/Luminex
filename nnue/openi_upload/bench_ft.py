"""Benchmark of the fused feature-layer kernels (ft_fused.py) against the trainer's dense
path (resident_train.DenseFT, NNUE_FT_DENSE=2) on one real batch: 131,072 positions of
t91 Feb, both sides, L1 = 768. Reports each kernel config's time and dense TFLOP/s
(2 * side-rows * 768 * L1 per pass), the error against exact fp32 references, and the
feature layer's share of a step before and after. usage: python bench_ft.py DATA_DIR"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ.update(NNUE_BUCKETS="8", NNUE_FT_DENSE="2", NNUE_FT_TF32="1", NNUE_FT_TILE="1024")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resident_train as RT   # noqa: E402
import ft_fused as FF         # noqa: E402

dev, L1, BS, NI = "cuda", 768, 131072, RT.NUM_INPUTS
ONLY = sys.argv[2] if len(sys.argv) > 2 else ""   # "bwd": skip the forward sweep (measured on 2026-11-12: best 1.459 ms)
BF16_PEAK, FP8_PEAK = 408.5, 804.6   # TFLOP/s, cuBLAS 16384^3 on this card (gpu_bench, 2026-11-12)
p = torch.cuda.get_device_properties(0)
print(f"{p.name} | torch {torch.__version__} | triton {FF.triton.__version__} | sm_{p.major}{p.minor}", flush=True)

torch.manual_seed(1)
part = sorted(glob.glob(sys.argv[1] + "/t91_*.lcr"))[0]
rec = torch.from_numpy(np.fromfile(part, dtype=np.uint8, count=4_000_000 * 32).reshape(-1, 32)).to(dev)
batch = rec[torch.randint(0, rec.shape[0], (BS,), device=dev)]
wf, bf, stm, tgt, flag, cnt = RT.expand(batch)
planes, kb = FF.side_rows(wf, bf, NI)
R = planes.shape[0]
feats = torch.cat([wf, bf])
FLOP = 2.0 * R * 768 * L1
counts = torch.bincount(kb, minlength=32)
print(f"batch {BS:,} positions = {R:,} side-rows; {int((feats < NI).sum()) / R:.1f} features a side-row; "
      f"largest king bucket {int(counts.max()) / R:.1%} of side-rows; {FLOP / 1e9:.0f} GFLOP a pass", flush=True)


def timeit(fn, iters=10, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


W = torch.randn(NI + 1, L1, device=dev) * 0.05   # fp32 master table (+ the zero padding row)
W[NI] = 0
bias = torch.randn(L1, device=dev) * 0.1
Wb = W[:NI].to(torch.bfloat16).contiguous()
G = torch.randn(R, L1, device=dev) * torch.exp2(torch.randint(-10, 1, (R, 1), device=dev).float()) * 1e-3
G = G * (torch.rand(R, L1, device=dev) < 0.6)    # like dL/dacc: units outside (0, 1) pass no gradient

# --- the trainer's current dense path, both sides -------------------------------------------
print("\n== current path: DenseFT (ft_plan scatter + bf16 bmm forward, fp32 gather + TF32 bmm + index_add backward)", flush=True)
Wp = W.clone().requires_grad_(True)
Gw, Gb = G[:BS].contiguous(), G[BS:].contiguous()


def cur_fwd():
    return RT.DenseFT.apply(wf, Wp, True), RT.DenseFT.apply(bf, Wp, True)


def cur_step():
    Wp.grad = None
    a, b = cur_fwd()
    torch.autograd.backward([a, b], [Gw, Gb])


t_cf = timeit(cur_fwd)
t_cs = timeit(cur_step)
print(f">>> current: forward {t_cf:.2f} ms, backward {t_cs - t_cf:.2f} ms, both {t_cs:.2f} ms "
      f"({2 * FLOP / t_cs / 1e9:.0f} TFLOP/s effective)", flush=True)
del Wp
torch.cuda.empty_cache()

# --- cuBLAS on the same FLOPs, for reference ------------------------------------------------
ntile = R // 1024
Xr = (torch.rand(ntile, 1024, 768, device=dev) < 0.04).to(torch.bfloat16)
Wr = torch.randn(ntile, 768, L1, device=dev).to(torch.bfloat16)
t = timeit(lambda: torch.bmm(Xr, Wr))
print(f">>> cuBLAS bf16 bmm [{ntile} x 1024 x 768 x {L1}], operands in memory: {t:.3f} ms {FLOP / t / 1e9:.0f} TFLOP/s", flush=True)
try:
    A8 = Xr.view(R, 768).to(torch.float8_e4m3fn)
    B8 = Wr[0].to(torch.float8_e4m3fn).t().contiguous().t()
    one = torch.tensor(1.0, device=dev)
    t = timeit(lambda: torch._scaled_mm(A8, B8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
    print(f">>> cuBLAS fp8 [{R} x 768] x [768 x {L1}] (one bucket's block, operands in memory): {t:.3f} ms "
          f"{FLOP / t / 1e9:.0f} TFLOP/s", flush=True)
    del A8, B8
except Exception as ex:
    print(f"  cuBLAS fp8: {ex.__class__.__name__}: {str(ex)[:120]}", flush=True)
del Xr, Wr
torch.cuda.empty_cache()

# --- fused: per-step plan cost --------------------------------------------------------------
order, _ = FF.plan(kb, 128)
planes_s = planes[order].contiguous()
feats_s = feats[order]
t_prep = timeit(lambda: FF.side_rows(wf, bf, NI))
t_plan = timeit(lambda: FF.plan(kb, 128))
t_gath = timeit(lambda: planes[order].contiguous())
print(f"\n== fused: bitboards {t_prep:.3f} ms, sort and plan {t_plan:.3f} ms, bitboard gather {t_gath:.3f} ms a step", flush=True)

# --- fused forward ---------------------------------------------------------------------------
Wf = torch.cat([Wb.float(), torch.zeros(1, L1, device=dev)])
ref_f = F.embedding_bag(feats_s, Wf, mode="sum", padding_idx=NI) + bias
ref_master = F.embedding_bag(feats_s, W, mode="sum", padding_idx=NI) + bias
W8, wsc = FF.fp8_cols(W[:NI])
fwd_cfgs = [(128, 128, 4, 3), (128, 128, 8, 3), (128, 128, 4, 4), (128, 256, 8, 2), (128, 256, 8, 3),
            (64, 128, 4, 3), (64, 256, 4, 3), (256, 128, 8, 3), (128, 64, 4, 4), (256, 256, 8, 2)]
ranges_by = {}
best_f = {False: (1.459, 256, 128, 8, 3, 1.7e-3), True: (1.137, 128, 128, 4, 3, 2.4e-2)} if ONLY == 'bwd' else {}
for fp8 in (() if ONLY == 'bwd' else (False, True)):
    print(f"\n== fused forward ({'fp8 e4m3 W, per-column scale' if fp8 else 'bf16 W'}): X tiles built from bitboards", flush=True)
    for BM, BN, nw, ns in fwd_cfgs:
        if BM not in ranges_by:
            ranges_by[BM] = FF.plan(kb, BM)[1]
        rg = ranges_by[BM]
        try:
            fn = (lambda: FF.ft_fwd(planes_s, rg, W8, bias, BM, BN, fp8=True, wscale=wsc, num_warps=nw, num_stages=ns)) if fp8 \
                else (lambda: FF.ft_fwd(planes_s, rg, Wb, bias, BM, BN, num_warps=nw, num_stages=ns))
            out = fn()
            torch.cuda.synchronize()
            t = timeit(fn)
            err = rel(out, ref_master if fp8 else ref_f)
            peak = FP8_PEAK if fp8 else BF16_PEAK
            print(f"  BM={BM:3d} BN={BN:3d} warps={nw} stages={ns}: {t:.3f} ms {FLOP / t / 1e9:5.0f} TFLOP/s "
                  f"({FLOP / t / 1e9 / peak:.0%} of cuBLAS peak) | rel err {err:.1e}", flush=True)
            if t < best_f.get(fp8, (1e9,))[0]:
                best_f[fp8] = (t, BM, BN, nw, ns, err)
        except Exception as ex:
            print(f"  BM={BM:3d} BN={BN:3d} warps={nw} stages={ns}: {ex.__class__.__name__}: {str(ex).splitlines()[0][:110]}", flush=True)
del ref_f, ref_master
torch.cuda.empty_cache()

# --- fused backward --------------------------------------------------------------------------
G_s = G[order]
Gb_s = G_s.to(torch.bfloat16).contiguous()
Wr = torch.zeros(NI + 1, L1, device=dev, requires_grad=True)
F.embedding_bag(feats_s, Wr, mode="sum", padding_idx=NI).backward(G_s)
ref_exact = Wr.grad[:NI].clone()        # X^T G in fp32 from the unrounded G
Wr.grad = None
F.embedding_bag(feats_s, Wr, mode="sum", padding_idx=NI).backward(Gb_s.float())
ref_bf = Wr.grad[:NI].clone()           # the same from G rounded to bf16 (what the kernel sees)
del Wr
G8, gs, ginv = FF.fp8_rows(G_s)
dw = torch.zeros(NI, L1, device=dev)
t_zero = timeit(lambda: dw.zero_())
bwd_cfgs = [(128, 128, 64, 4, 3, 8192), (128, 128, 64, 8, 3, 8192), (128, 128, 32, 4, 4, 8192), (256, 128, 64, 8, 2, 8192),
            (128, 256, 64, 8, 2, 8192), (256, 64, 64, 4, 3, 8192), (128, 128, 64, 4, 3, 4096), (128, 128, 64, 4, 3, 16384),
            (128, 64, 64, 4, 4, 8192), (256, 128, 32, 8, 3, 8192), (256, 128, 64, 8, 3, 16384), (128, 128, 64, 4, 2, 8192)]
best_b = {}
for fp8 in (False, True):
    print(f"\n== fused backward ({'fp8 e4m3 G, row-scaled' if fp8 else 'bf16 G'}): X^T tiles built from bitboards, "
          f"fp32 atomics per range (zeroing dW: {t_zero:.3f} ms, counted below)", flush=True)
    for BMF, BN, BK, nw, ns, CH in bwd_cfgs:
        if CH not in ranges_by:
            ranges_by[CH] = FF.plan(kb, CH)[1]
        rg = ranges_by[CH]
        try:
            if fp8:
                def fn():
                    dw.zero_()
                    FF.ft_bwd(planes_s, rg, G8, dw, BMF, BN, BK, fp8=True, gscale=gs, num_warps=nw, num_stages=ns)
            else:
                def fn():
                    dw.zero_()
                    FF.ft_bwd(planes_s, rg, Gb_s, dw, BMF, BN, BK, num_warps=nw, num_stages=ns)
            fn()
            torch.cuda.synchronize()
            got = dw * ginv if fp8 else dw.clone()
            t = timeit(fn)
            peak = FP8_PEAK if fp8 else BF16_PEAK
            e_exact = rel(got, ref_exact)
            e_same = rel(got, ref_bf) if not fp8 else float("nan")
            print(f"  BMF={BMF:3d} BN={BN:3d} BK={BK:2d} warps={nw} stages={ns} CH={CH:5d}: {t:.3f} ms "
                  f"{FLOP / t / 1e9:5.0f} TFLOP/s ({FLOP / t / 1e9 / peak:.0%} of cuBLAS peak) | rel err vs exact "
                  f"{e_exact:.1e}" + ("" if fp8 else f", vs same bf16 G {e_same:.1e}"), flush=True)
            if t < best_b.get(fp8, (1e9,))[0]:
                best_b[fp8] = (t, BMF, BN, BK, nw, ns, CH, e_exact)
        except Exception as ex:
            print(f"  BMF={BMF:3d} BN={BN:3d} BK={BK:2d} warps={nw} stages={ns} CH={CH:5d}: {ex.__class__.__name__}: "
                  f"{str(ex).splitlines()[0][:110]}", flush=True)

# --- summary ---------------------------------------------------------------------------------
print("\n== summary (one step's feature layer, both sides; the rest of the step is unchanged here)", flush=True)
over = t_prep + t_plan + t_gath   # bitboards, the sort and plan, and the sorted gather of the bitboards
print(f">>> current dense path: {t_cs:.2f} ms", flush=True)
for name, ff, bb in (("bf16 forward + bf16 backward", False, False), ("bf16 forward + fp8 backward", False, True),
                     ("fp8 forward + fp8 backward", True, True)):
    if ff in best_f and bb in best_b:
        tot = best_f[ff][0] + best_b[bb][0] + over
        print(f">>> fused {name}: forward {best_f[ff][0]:.3f} + backward {best_b[bb][0]:.3f} + plan and gathers "
              f"~{over:.2f} = {tot:.2f} ms -> {t_cs / tot:.1f}x on the feature layer, "
              f"{18.62 - t_cs + tot:.1f} ms a step instead of 18.62 if nothing else changed", flush=True)
for k, v in best_f.items():
    print(f">>> best forward {'fp8' if k else 'bf16'}: BM={v[1]} BN={v[2]} warps={v[3]} stages={v[4]}, {v[0]:.3f} ms, "
          f"{FLOP / v[0] / 1e9:.0f} TFLOP/s, rel err {v[5]:.1e}", flush=True)
for k, v in best_b.items():
    print(f">>> best backward {'fp8' if k else 'bf16'}: BMF={v[1]} BN={v[2]} BK={v[3]} warps={v[4]} stages={v[5]} CH={v[6]}, "
          f"{v[0]:.3f} ms, {FLOP / v[0] / 1e9:.0f} TFLOP/s, rel err vs exact {v[7]:.1e}", flush=True)
print("== FINISHED: fused feature-layer benchmark", flush=True)
