#!/usr/bin/env python3
# Luminex NNUE v3 trainer — "LNI8-QAT": Quantization-Aware Training + capacity rebalance.
#
# GOAL: stronger AND faster than v2.
#   v2 = L1=512 L2=16 L3=32, trained in float, quantized to int8 POST-HOC (loses precision).
#        -> 990K NPS, -13.9 Elo vs HCE.
#   v3 = L1=256 L2=32 L3=64, trained WITH simulated int8/int16 quantization (QAT).
#        -> ~1.15-1.2M NPS (SCReLU + the incremental update both halve with L1) AND stronger
#           (QAT eliminates post-hoc quantization loss; L2=32 doubles L2 capacity at the SAME
#           dot-product MAC count as v2: 32*512 = 16*1024; L3=64 adds cheap final-layer capacity
#           since the L3 dot is tiny: L3 x L2 = 64 x 32).
#
# ###############################################################################
# THE THREE INNOVATIONS
# ###############################################################################
# 1) CAPACITY REBALANCE  (speed + strength, zero engine change)
#    L1 512->256 : the FT accumulator width. This is what the SCReLU pass (2*L1 values)
#                  AND the incremental update (delta over L1 values per feature) iterate over.
#                  Halving L1 ~halves BOTH -> the dominant per-node NNUE costs drop ~40%.
#    L2 16->32   : the L2 dot product is (L2_outputs) x (2*L1_inputs) MACs. With L1=256,
#                  L2=32 gives 32*512 = 16384 MACs — IDENTICAL to v2's 16*1024. So the L2
#                  dot (the eval's biggest single cost, VPDPBUSD-throughput-bound) is unchanged,
#                  but we get 2x the L2 output neurons (more expressive) for free.
#    Net: faster inference (update+SCReLU halved) + more L2 capacity. Same L2-dot budget.
#
# 2) QUANTIZATION-AWARE TRAINING (QAT)  — the headline STRENGTH innovation
#    v2 trained in float, then quantize_int8.py crushed each layer to int8 afterwards. The
#    shipped int8 net is an APPROXIMATION of the trained float net — it loses precision,
#    especially in L2/L3/out (8-bit) and the FT (16-bit). This is silent Elo leakage.
#    v3 trains with FAKE QUANTIZATION inserted in the forward pass: every weight and activation
#    is passed through round()/clamp() to simulate the EXACT int8/int16 arithmetic the C++
#    engine performs at inference. Gradients flow through via the straight-through estimator
#    (STE), so backprop OPTIMIZES the quantized-representable weights directly. The shipped
#    int8 net then performs like the trained net — no post-hoc surprise.
#    The fake-quant scales MATCH the engine EXACTLY:
#      - FT weights : round(w*8192).clamp(int16)/8192   (engine FT_WSCALE = 8192)
#      - L2/L3/out w: round(w * (127/max_abs))/s         (engine + quantize_int8.py per-layer)
#      - activations: round(h*127)/127  after each SCReLU (engine "quant8" pass, h already in [0,1])
#    Because the simulation matches quantize_int8.py bit-for-bit, you STILL run the existing
#    quantize_int8.py on the v3 float export — and it reproduces what training simulated. No
#    engine change, no new format.
#
# 3) QAT WARMUP + MODERN OPTIMIZATION  (training stability / final quality)
#    - Warmup: first `qat_warmup_frac` of epochs train in PURE FLOAT (qat=False) so the net
#      learns structure; then QAT switches on so it adapts to int8. Training with fake-quant
#      from step 0 is noisy/unstable; the warmup converges fast then refines under quantization.
#    - Cosine LR (always), SWA over the last 25%, bf16 AMP, AdamW, gradient clipping.
#    - L1=256 is ~2x cheaper per step than v2's L1=512, so MORE epochs fit the C500 session.
#
# ###############################################################################
# ENGINE COMPATIBILITY — ZERO C++ CHANGES REQUIRED
# ###############################################################################
# L1/L2/L3 are read from the net header. NNUE_L1_MAX=512 covers L1=256. The engine's
# dot_i8_vnni_small already handles small-n dots (L3 dot n=L2=32, out dot n=L3=32 via the
# 32-byte zero-extend VPDPBUSD path). So: train -> quantize_int8.py -> load luminex_v3_i8.nnue.
#
# ###############################################################################
# USAGE (on C500 / MetaX mcPyTorch)
# ###############################################################################
#   export MACA_PATH=/opt/maca/ ; export LD_LIBRARY_PATH=...   (see mcPyTorch guide)
#   export PYTORCH_DEFAULT_NCHW=1
#   # Reuse the SAME featurized 280M-position dataset as v2 (features are FT inputs,
#   # independent of L1/L2/L3). Set NNUE_NPY_DIR to the dir with feat_w/feat_b/feat_s/feat_t.npy.
#   export NNUE_NPY_DIR=/tmp/code
#   python luminex_nnue_train_v3.py /tmp/code/sf_dataset_big.txt \
#       --epochs 30 --batch-size 16384 --lr 1.2e-3 --swa \
#       --L1 256 --L2 32 --L3 64 --qat-warmup-frac 0.25 \
#       --out /tmp/code/luminex_v3.nnue --ckpt /tmp/code/v3_ckpt.pt
#   # Then quantize (unchanged from v2):
#   python quantize_int8.py /tmp/code/luminex_v3.nnue /tmp/code/luminex_v3_i8.nnue
#   # Upload luminex_v3_i8.nnue to hyperai, load via "setoption name NNUEFile value ...".

import sys, math, struct, time, os
import numpy as np
import chess
import torch
from torch import nn

# ============================================================================
# HalfKAv2_hm feature indexer  (identical to v2; proven correct vs SF nnue-pytorch)
# ============================================================================
KingBuckets = [
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1,  3,  2,  1,  0,
]
NUM_SQ, NUM_PT, NUM_BUCKETS = 64, 12, 32
NUM_PLANES = NUM_SQ * NUM_PT
NUM_INPUTS = NUM_PLANES * NUM_BUCKETS   # 24576 per half
MAX_PIECES = 32
FT_WSCALE = 8192.0   # MUST match src/nnue.cpp FT_WSCALE (engine int16 FT quantization step)


def _orient(is_white_pov, sq, ksq):
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def _halfka_idx(is_white_pov, king_sq, sq, piece_color, piece_type):
    p_idx = (piece_type - 1) * 2 + (1 if piece_color != is_white_pov else 0)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 768


def active_features(board):
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    w, b = [], []
    for sq, p in board.piece_map().items():
        pc, pt = (1 if p.color == chess.WHITE else 0), int(p.piece_type)
        w.append(_halfka_idx(True,  wk, sq, pc, pt))
        b.append(_halfka_idx(False, bk, sq, pc, pt))
    return w, b


# ============================================================================
# QAT fake-quantization — simulates the C++ engine's int8/int16 inference EXACTLY.
# Each returns a value whose FORWARD equals the quantized-dequantized input, but whose
# BACKWARD is the identity (straight-through estimator): grad flows to the float weight
# as if no rounding happened. This is what lets SGD optimize quantization-aware weights.
# ============================================================================
def fake_quant_i8(w):
    """Per-LAYER int8: round(w * s)/s with s = 127/max_abs(w). Matches quantize_int8.py.
    Forced to fp32 (autocast disabled) so the QAT simulation is EXACT — the rounding step
    must not itself suffer bf16 error, or we'd be training on a wrong quantization model."""
    with torch.autocast(device_type=w.device.type, enabled=False):
        w = w.float()
        s = (127.0 / w.abs().max().clamp(min=1e-8)).detach()
        wq = torch.round(w * s) / s
        return w + (wq - w).detach()


def fake_quant_int16(w):
    """FT int16: round(w*8192).clamp(int16)/8192. Matches engine FT load (FT_WSCALE=8192)."""
    with torch.autocast(device_type=w.device.type, enabled=False):
        w = w.float()
        wq = torch.round(w * FT_WSCALE).clamp(-32768.0, 32767.0) / FT_WSCALE
        return w + (wq - w).detach()


def fake_quant_u8(h):
    """Activation uint8 after SCReLU: round(h*127)/127. h is already in [0,1] (screlu output).
    Matches the engine 'quant8' pass (screlu then round(x*127))."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        h = h.float()
        wq = torch.round(h * 127.0) / 127.0
        return h + (wq - h).detach()


# ============================================================================
# Model — QAT-aware L-NNUE
# ============================================================================
class QLinear(nn.Module):
    """nn.Linear whose weights are fake-quantized to int8 when qat=True (L2/L3/out layers).
    Bias stays float (the engine stores biases as float — no bias quantization)."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x, qat):
        w = fake_quant_i8(self.weight) if qat else self.weight
        return torch.nn.functional.linear(x, w, self.bias)


class LNNUEv3(nn.Module):
    def __init__(self, L1=256, L2=32, L3=64):
        super().__init__()
        self.ft = nn.Linear(NUM_INPUTS, L1)
        self.l2 = QLinear(2 * L1, L2)
        self.l3 = QLinear(L2, L3)
        self.out = QLinear(L3, 1)
        self.L1, self.L2, self.L3 = L1, L2, L3
        nn.init.zeros_(self.ft.bias)
        # FT init uses EFFECTIVE fan-in (~32 active features/half), not NUM_INPUTS=24576.
        # std ~ 1/sqrt(32) -> accumulator std ~1 (active SCReLU range). (Proven in v2.)
        nn.init.normal_(self.ft.weight, mean=0.0, std=0.2)

    def forward(self, w_idx, b_idx, stm, qat=True):
        # FT: int16 fake-quant weights (when qat), embedding_bag = fused gather+sum.
        ftw = self.ft.weight
        if qat:
            ftw = fake_quant_int16(ftw)
        ft_w = torch.cat([ftw.t(),
                          torch.zeros(1, self.L1, device=ftw.device)], dim=0)  # zero-pad row
        acc_w = torch.nn.functional.embedding_bag(w_idx, ft_w, mode='sum') + self.ft.bias
        acc_b = torch.nn.functional.embedding_bag(b_idx, ft_w, mode='sum') + self.ft.bias
        stm_mask = stm.view(-1, 1).float()
        stm_acc = stm_mask * acc_w + (1 - stm_mask) * acc_b
        nstm_acc = (1 - stm_mask) * acc_w + stm_mask * acc_b
        h = torch.cat([stm_acc, nstm_acc], dim=1)
        # Three SCReLU blocks, each followed by uint8 activation fake-quant (matches engine
        # screlu+quant8 after L1, L2, L3). The out layer has NO activation (final scalar).
        h = torch.clamp(h, 0.0, 1.0) ** 2
        if qat:
            h = fake_quant_u8(h)
        h = torch.clamp(self.l2(h, qat), 0.0, 1.0) ** 2
        if qat:
            h = fake_quant_u8(h)
        h = torch.clamp(self.l3(h, qat), 0.0, 1.0) ** 2
        if qat:
            h = fake_quant_u8(h)
        # Output scale: raw out (~+/-1) -> eval-cp range so sigmoid(out/SCALE) spans the
        # target range and gradients flow. (Same as v2.)
        return self.out(h, qat).squeeze(-1) * 300.0


# ============================================================================
# Data loading (identical to v2 — reuses the 280M-position featurized cache as-is).
# ============================================================================
def load_dataset(path, max_positions=None):
    fens, evals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            fen, _, ev = line.rpartition('|')
            try:
                ev = int(ev)
            except ValueError:
                continue
            fens.append(fen)
            evals.append(ev)
            if max_positions and len(fens) >= max_positions:
                break
    return fens, evals


def featurize_all(fens, evals, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        try:
            d = torch.load(cache_path, map_location='cpu', weights_only=False)
            if int(d['w'].shape[0]) == len(fens):
                print(f"  loaded featurize cache {cache_path} ({len(fens)} pos)", flush=True)
                return d['w'], d['b'], d['stm'], d['target']
        except Exception as e:
            print(f"  cache load failed ({e}), re-featurizing", flush=True)
    N = len(fens)
    w_idx = torch.full((N, MAX_PIECES), NUM_INPUTS, dtype=torch.int32)
    b_idx = torch.full((N, MAX_PIECES), NUM_INPUTS, dtype=torch.int32)
    stm = torch.zeros(N, dtype=torch.float32)
    target = torch.zeros(N, dtype=torch.float32)
    t0 = time.time()
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        w, b = active_features(board)
        w = w[:MAX_PIECES]; b = b[:MAX_PIECES]
        w_idx[i, :len(w)] = torch.tensor(w, dtype=torch.int32)
        b_idx[i, :len(b)] = torch.tensor(b, dtype=torch.int32)
        stm[i] = 1.0 if board.turn == chess.WHITE else 0.0
        target[i] = evals[i]
        if (i + 1) % 200000 == 0:
            print(f"  featurized {i+1}/{N} ({(i+1)/(time.time()-t0):.0f} pos/s)", flush=True)
    if cache_path:
        try:
            torch.save({'w': w_idx, 'b': b_idx, 'stm': stm, 'target': target}, cache_path)
        except Exception as e:
            print(f"  (cache save failed: {e})", flush=True)
    return w_idx, b_idx, stm, target


def load_vram(npy_path, device):
    """Stream the pre-featurized .npy cache (from c500_featurize.py) into VRAM in parallel.
    The C500 has 64GB VRAM; the 280M-position dataset is ~38GB and fits with headroom."""
    import numpy as _np
    meta_path = os.path.join(npy_path, "featurized_meta.txt")
    if os.path.exists(meta_path):
        with open(meta_path) as mf:
            n_valid = int(mf.readline().strip())
    else:
        n_valid = _np.load(os.path.join(npy_path, "feat_s.npy"), mmap_mode='r').shape[0]
    print(f"  streaming {n_valid:,} positions (~38GB) into VRAM...", flush=True)
    t_load = time.time()
    w_all = torch.empty((n_valid, MAX_PIECES), dtype=torch.int16, device=device)
    b_all = torch.empty((n_valid, MAX_PIECES), dtype=torch.int16, device=device)
    stm_all = torch.empty(n_valid, dtype=torch.float32, device=device)
    target_all = torch.empty(n_valid, dtype=torch.float32, device=device)
    from concurrent.futures import ThreadPoolExecutor
    WORKERS = int(os.environ.get("NNUE_LOAD_WORKERS", "8"))
    CH = 500_000

    def vram_load(name, dst):
        src = _np.load(os.path.join(npy_path, name), mmap_mode='r')
        ranges = [(i, min(i + CH, n_valid)) for i in range(0, n_valid, CH)]

        def read(ij):
            i, j = ij
            return i, torch.from_numpy(_np.array(src[i:j]))

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(read, ij) for ij in ranges]
            done = 0
            for fut in futs:
                i, t = fut.result()
                end = i + t.shape[0]
                dst[i:end] = t.to(device)
                del t
                done += 1
                if done % 50 == 0:
                    print(f"    {name}: {end:,}/{n_valid:,} ({time.time()-t_load:.0f}s)", flush=True)
        del src

    vram_load("feat_w.npy", w_all)
    vram_load("feat_b.npy", b_all)
    vram_load("feat_s.npy", stm_all)
    vram_load("feat_t.npy", target_all)
    print(f"  VRAM cache ready in {time.time()-t_load:.0f}s ({WORKERS} readers)", flush=True)
    return w_all, b_all, stm_all, target_all


# ============================================================================
# Train
# ============================================================================
def train(data, out, epochs=30, batch_size=16384, lr=1.2e-3, device='cuda',
          max_positions=None, L1=256, L2=32, L3=64, swa=False, qat_warmup_frac=0.25,
          feat_arrays=None, time_budget_sec=None, ckpt_path=None, use_amp=True):
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"=== Luminex NNUE v3 (QAT) ===", flush=True)
    print(f"device: {device} | L1={L1} L2={L2} L3={L3} | lr={lr} | swa={swa} | "
          f"qat_warmup={qat_warmup_frac} | amp={use_amp} | budget={time_budget_sec}s", flush=True)

    print("loading + pre-featurizing data...", flush=True)
    if feat_arrays is not None:
        w_np, b_np, stm_np, tgt_np = feat_arrays
        w_all = torch.from_numpy(w_np); b_all = torch.from_numpy(b_np)
        stm_all = torch.from_numpy(stm_np); target_all = torch.from_numpy(tgt_np)
        print(f"  {w_all.shape[0]:,} positions (in-RAM arrays)", flush=True)
    else:
        npy_path = os.environ.get("NNUE_NPY_DIR", "/tmp/code")
        if os.path.exists(os.path.join(npy_path, "feat_w.npy")):
            w_all, b_all, stm_all, target_all = load_vram(npy_path, device)
        else:
            fens, evals = load_dataset(data, max_positions)
            print(f"  {len(fens)} positions (FEN|eval)", flush=True)
            cache_path = os.environ.get("NNUE_CACHE", "/tmp/luminex_featurized_v3.pt")
            w_all, b_all, stm_all, target_all = featurize_all(fens, evals, cache_path)
    n = w_all.shape[0]
    # Hold out a validation set. The train-vs-val loss curves are the ONLY honest way to
    # decide whether to add capacity (underfitting: train plateaus high) or data
    # (overfitting: val diverges from train). Don't increase L1 or data without this evidence.
    VAL_N = min(500000, n // 200)   # ~0.5%, capped at 500K
    train_n = n - VAL_N
    val_w = w_all[train_n:]; val_b = b_all[train_n:]
    val_stm = stm_all[train_n:]; val_tgt = target_all[train_n:]
    print(f"  train={train_n:,} val={VAL_N:,} (val loss tracked per epoch -> capacity/data diagnosis)", flush=True)

    model = LNNUEv3(L1=L1, L2=L2, L3=L3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    SCALE = 400.0
    bs = batch_size
    steps_per_epoch = (train_n + bs - 1) // bs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    # SWA: average weights over the last 25% of training -> flatter minimum, better generalization.
    swa_model = None
    swa_start = 0
    if swa and epochs >= 2:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start = steps_per_epoch * epochs * 3 // 4
    # QAT warmup: pure float for the first fraction, then fake-quant on for the rest.
    qat_on_step = int(steps_per_epoch * epochs * qat_warmup_frac)
    grad_clip = 1.0   # stabilizes QAT (STE gradients can spike)

    gstep = 0
    start_epoch = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            scheduler.load_state_dict(ck['sched'])
            if swa_model is not None and 'swa' in ck:
                swa_model.load_state_dict(ck['swa'])
            start_epoch = ck['epoch']
            gstep = ck.get('gstep', 0)
            print(f"  RESUMED from checkpoint at epoch {start_epoch} (gstep {gstep})", flush=True)
        except Exception as e:
            print(f"  checkpoint load failed ({e}) — starting fresh", flush=True)

    train_t0 = time.time()
    completed_all = False
    for epoch in range(start_epoch, epochs):
        if time_budget_sec and (time.time() - train_t0) >= time_budget_sec:
            print(f"  time budget {time_budget_sec}s reached before epoch {epoch} — pausing", flush=True)
            break
        perm = torch.randperm(train_n, device=device)   # train positions only (val held out)
        running, t0 = 0.0, time.time()
        for s in range(steps_per_epoch):
            if time_budget_sec and (time.time() - train_t0) >= time_budget_sec:
                break
            idx = perm[s*bs:(s+1)*bs]
            w_idx = w_all[idx].to(device).long()
            b_idx = b_all[idx].to(device).long()
            stm = stm_all[idx].to(device)
            target = target_all[idx].to(device)
            qat = (gstep >= qat_on_step)
            opt.zero_grad()
            if use_amp:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    pred = model(w_idx, b_idx, stm, qat=qat)
                    sig_p = torch.sigmoid(pred / SCALE)
                    sig_t = torch.sigmoid(target / SCALE)
                    loss = ((sig_p - sig_t) ** 2).mean()
                loss.backward()
            else:
                pred = model(w_idx, b_idx, stm, qat=qat)
                sig_p = torch.sigmoid(pred / SCALE)
                sig_t = torch.sigmoid(target / SCALE)
                loss = ((sig_p - sig_t) ** 2).mean()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            gstep += 1
            if swa_model is not None and gstep >= swa_start:
                swa_model.update_parameters(model)
            running += 1
            if s == 0 or (s + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  epoch {epoch} step {s+1}/{steps_per_epoch} loss={loss.item():.5f} "
                      f"qat={'ON' if qat else 'off'} ({(s+1)*bs/dt:.0f} pos/s)", flush=True)
        # Validation loss (no grad, qat=ON = deployed mode). Compare to train loss:
        #   val >> train & rising  -> OVERFITTING -> data is the limit (need more positions)
        #   train plateaus high    -> UNDERFITTING -> capacity is the limit (need bigger L1)
        #   both converge & close  -> at the sweet spot for this L1+data
        model.eval()
        with torch.no_grad():
            vls = []
            for vi in range(0, VAL_N, bs):
                vw = val_w[vi:vi+bs]; vb = val_b[vi:vi+bs]
                vs = val_stm[vi:vi+bs]; vt = val_tgt[vi:vi+bs]
                if use_amp:
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        vp = model(vw.to(device).long(), vb.to(device).long(), vs.to(device), qat=True)
                else:
                    vp = model(vw.to(device).long(), vb.to(device).long(), vs.to(device), qat=True)
                vls.append(((torch.sigmoid(vp/SCALE) - torch.sigmoid(vt.to(device)/SCALE))**2).mean().item())
            val_loss = sum(vls) / len(vls)
        model.train()
        print(f"  >>> epoch {epoch} VAL loss={val_loss:.5f}  (train ~{loss.item():.5f}, gap={val_loss-loss.item():.5f})", flush=True)
        if ckpt_path:
            ck = {'epoch': epoch + 1, 'model': model.state_dict(), 'opt': opt.state_dict(),
                  'sched': scheduler.state_dict(), 'gstep': gstep}
            if swa_model is not None:
                ck['swa'] = swa_model.state_dict()
            torch.save(ck, ckpt_path)
            # Save a usable net every epoch so a killed session still leaves a net behind.
            _sm = swa_model.module if (swa_model is not None and gstep >= swa_start) else model
            _latest = os.path.join(os.path.dirname(ckpt_path) or '.', "luminex_v3_latest.nnue")
            save_nnue(_sm, _latest, qat=True)   # save the QAT-trained (quantization-matched) weights
            print(f"  [epoch {epoch+1}/{epochs} saved — ckpt + {_latest}]", flush=True)
    else:
        completed_all = True

    save_model = swa_model.module if (swa_model is not None and gstep >= swa_start) else model
    if save_model is getattr(swa_model, 'module', None):
        print("  using SWA-averaged weights", flush=True)
    save_nnue(save_model, out, qat=True)
    print(f"saved {out}", flush=True)
    if completed_all and ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("  training complete — checkpoint cleared", flush=True)
    return out


def save_nnue(model, path, qat=True):
    """Export float32 LNN1 (same format as v2 -> existing quantize_int8.py converts to LNI8).
    The weights are QAT-trained, so they quantize to int8 with minimal precision loss.
    NOTE: we save the RAW float parameters (the engine/quantizer apply their own quantization).
    Saving the already-fake-quantized values would double-quantize; the trained float values,
    when passed through quantize_int8.py, reproduce exactly what QAT simulated."""
    model = model.module if hasattr(model, 'module') else model  # unwrap SWA
    with open(path, 'wb') as f:
        f.write(b'LNN1')
        f.write(struct.pack('iiii', model.L1, model.L2, model.L3, NUM_INPUTS))
        for name in ['ft.weight', 'ft.bias', 'l2.weight', 'l2.bias',
                     'l3.weight', 'l3.bias', 'out.weight', 'out.bias']:
            t = dict(model.named_parameters())[name].detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack('i', t.size))
            f.write(t.tobytes())


def main():
    import argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('data', nargs='?', default='', help='FEN|eval_cp file (ignored if .npy cache exists)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1.2e-3)
    ap.add_argument('--max-positions', type=int, default=None)
    ap.add_argument('--out', default='luminex_v3.nnue')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--L1', type=int, default=256)
    ap.add_argument('--L2', type=int, default=32)
    ap.add_argument('--L3', type=int, default=64)
    ap.add_argument('--swa', action='store_true')
    ap.add_argument('--qat-warmup-frac', type=float, default=0.25)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--time-budget', type=int, default=None, help='stop before N seconds (resumes)')
    ap.add_argument('--no-amp', action='store_true')
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.batch_size, args.lr, args.device,
          args.max_positions, args.L1, args.L2, args.L3, args.swa, args.qat_warmup_frac,
          None, args.time_budget, args.ckpt, not args.no_amp)


if __name__ == '__main__':
    main()
