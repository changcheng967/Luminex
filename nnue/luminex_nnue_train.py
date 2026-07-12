#!/usr/bin/env python3
# Luminex NNUE trainer — lean custom trainer built for OUR situation.
#
# Why custom (not stockfish nnue-pytorch): SF's toolchain is built for their
# 100B-position scale (binpack + data_loader + lightning + conversion tools).
# We have 8.4M positions already in simple FEN|eval_cp format. A lean trainer
# that reads that directly is simpler, runs on MetaX mcPyTorch with zero
# extra infra, and trains a small net in hours.
#
# Architecture (L-NNUE): HalfKAv2_hm feature transformer (proven, from SF)
#   -> L1=256 -> ClippedReLU -> L2=16 -> ClippedReLU -> L3=32 -> ClippedReLU -> 1
# Small on purpose: bullet TC (eval called millions of times) + limited data
# (smaller net infers faster and overfits less).
#
# Objective: distillation — predict the Stockfish eval (the label). This is
# exactly how SF trains (net learns a stronger engine's judgment), and avoids
# the self-referential trap that killed our Texel attempts.
#
# Runs on MetaX C500 via mcPyTorch (CUDA-compatible). Set:
#   export MACA_PATH=/opt/maca/ ; export LD_LIBRARY_PATH=...  (see mcPyTorch guide)
#   export PYTORCH_DEFAULT_NCHW=1   (avoid .view() NCHW quirk)

import sys, math, struct, time, os
import numpy as np
import chess
import torch
from torch import nn

# ============================================================================
# HalfKAv2_hm feature indexer  (adapted from official-stockfish/nnue-pytorch)
# Proven correct; do not change the math without re-deriving against SF.
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
NUM_PLANES = NUM_SQ * NUM_PT          # 768
NUM_INPUTS = NUM_PLANES * NUM_BUCKETS # 24576 per half
MAX_PIECES = 32                       # max active features per half


def _orient(is_white_pov, sq, ksq):
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def _halfka_idx(is_white_pov, king_sq, sq, piece_color, piece_type):
    # p_idx: 0..11 ; own pieces (== pov color) take the lower of each pair
    p_idx = (piece_type - 1) * 2 + (1 if piece_color != is_white_pov else 0)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 768


def active_features(board):
    """Return (white_pov_indices, black_pov_indices) as python lists."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    w, b = [], []
    for sq, p in board.piece_map().items():
        pc, pt = (1 if p.color == chess.WHITE else 0), int(p.piece_type)
        w.append(_halfka_idx(True,  wk, sq, pc, pt))
        b.append(_halfka_idx(False, bk, sq, pc, pt))
    return w, b


# ============================================================================
# Model
# ============================================================================
class LNNUE(nn.Module):
    def __init__(self, L1=256, L2=16, L3=32):
        super().__init__()
        self.ft = nn.Linear(NUM_INPUTS, L1)            # feature transformer
        self.l2 = nn.Linear(2 * L1, L2)                # stm + nstm concatenated
        self.l3 = nn.Linear(L2, L3)
        self.out = nn.Linear(L3, 1)
        self.L1, self.L2, self.L3 = L1, L2, L3
        nn.init.zeros_(self.ft.bias)
        # Feature transformer init must use the EFFECTIVE fan-in (~32 active
        # features per half), not NUM_INPUTS=24576. Default nn.Linear init
        # gives weights ~0.004 -> accumulator std ~0.02 -> vanishing gradients.
        # std ~ 1/sqrt(32) gives accumulator std ~1 (active ClippedReLU range).
        nn.init.normal_(self.ft.weight, mean=0.0, std=0.2)

    def forward(self, w_idx, b_idx, stm):  # idx: (B, MAX) padded with NUM_INPUTS; stm: (B,)
        # embedding_bag = FUSED gather+sum over the 32 feature slots → (B, L1), with NO
        # giant (B, 32, L1) intermediate. This is the SF-standard feature transformer and
        # the key speedup (the old ft_w[w_idx].sum(1) built a ~2GB tensor every batch and
        # ran at 29K pos/s; embedding_bag is typically 10-50x faster). Row NUM_INPUTS is the
        # zero-padding row (appended by the cat), so padded slots contribute nothing.
        ft_w = torch.cat([self.ft.weight.t(),
                          torch.zeros(1, self.L1, device=w_idx.device)], dim=0)
        acc_w = torch.nn.functional.embedding_bag(w_idx, ft_w, mode='sum') + self.ft.bias
        acc_b = torch.nn.functional.embedding_bag(b_idx, ft_w, mode='sum') + self.ft.bias
        # Order: side-to-move first
        stm_mask = stm.view(-1, 1).float()
        stm_acc = stm_mask * acc_w + (1 - stm_mask) * acc_b
        nstm_acc = (1 - stm_mask) * acc_w + stm_mask * acc_b
        h = torch.cat([stm_acc, nstm_acc], dim=1)
        # SCReLU: (clamp(x, 0, 1))² — proven +15-30 Elo over ClippedReLU (Seer/SF v9+)
        h = torch.clamp(h, 0.0, 1.0) ** 2
        h = torch.clamp(self.l2(h), 0.0, 1.0) ** 2
        h = torch.clamp(self.l3(h), 0.0, 1.0) ** 2
        # Output scale: raw out (~±1) -> eval-cp range so sigmoid(out/SCALE)
        # spans the target range and gradients flow. Without this the net is
        # stuck predicting 0.5 (sigmoid(±tiny/400)≈0.5, near-zero gradient).
        return self.out(h).squeeze(-1) * 300.0


# ============================================================================
# Data: read FEN|eval_cp, build padded feature-index tensors
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


def featurize_batch(fens, evals, device):
    B = len(fens)
    # Pad with NUM_INPUTS — a zero-weight row in the forward pass (no-op).
    w_idx = torch.full((B, MAX_PIECES), NUM_INPUTS, dtype=torch.long, device=device)
    b_idx = torch.full((B, MAX_PIECES), NUM_INPUTS, dtype=torch.long, device=device)
    stm = torch.zeros(B, dtype=torch.float, device=device)
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        w, b = active_features(board)
        w = w[:MAX_PIECES]; b = b[:MAX_PIECES]
        w_idx[i, :len(w)] = torch.tensor(w, dtype=torch.long)
        b_idx[i, :len(b)] = torch.tensor(b, dtype=torch.long)
        stm[i] = 1.0 if board.turn == chess.WHITE else 0.0
    target = torch.tensor(evals, dtype=torch.float, device=device)
    return w_idx, b_idx, stm, target


def featurize_all(fens, evals, cache_path=None):
    """Pre-featurize ALL positions ONCE into CPU tensors. The python-chess
    parsing (~6k pos/s) was the per-epoch bottleneck; doing it once upfront
    means every subsequent epoch is GPU-fast -> many more epochs in a fixed
    GPU session -> a much stronger net.

    If cache_path is given and a valid cache exists there, load it (skip the
    ~15 min featurize). Otherwise featurize and save to cache_path."""
    if cache_path and os.path.exists(cache_path):
        try:
            d = torch.load(cache_path, map_location='cpu', weights_only=False)
            if int(d['w'].shape[0]) == len(fens):
                print(f"  loaded featurize cache {cache_path} ({len(fens)} pos) — skipping featurize", flush=True)
                return d['w'], d['b'], d['stm'], d['target']
            print(f"  cache stale (size {d['w'].shape[0]} != {len(fens)}), re-featurizing", flush=True)
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
    print(f"  featurize done in {time.time()-t0:.0f}s", flush=True)

    if cache_path:
        try:
            torch.save({'w': w_idx, 'b': b_idx, 'stm': stm, 'target': target}, cache_path)
            print(f"  saved featurize cache {cache_path}", flush=True)
        except Exception as e:
            print(f"  (cache save failed: {e})", flush=True)
    return w_idx, b_idx, stm, target


# ============================================================================
# Train
# ============================================================================
def train(data, out, epochs=1, batch_size=16384, lr=1e-3, device='cuda', max_positions=None, L1=256, lr_schedule=False, swa=False, feat_arrays=None, time_budget_sec=None, ckpt_path=None, use_amp=False):
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"device: {device} | L1={L1} | lr_schedule={lr_schedule} | swa={swa} | "
          f"amp={use_amp} | budget={time_budget_sec}s | ckpt={ckpt_path}", flush=True)

    print("loading + pre-featurizing data (one-time; epochs after are GPU-fast)...", flush=True)

    if feat_arrays is not None:
        # In-RAM arrays passed directly (combined featurize+train, no disk)
        w_np, b_np, stm_np, tgt_np = feat_arrays
        w_all = torch.from_numpy(w_np)
        b_all = torch.from_numpy(b_np)
        stm_all = torch.from_numpy(stm_np)
        target_all = torch.from_numpy(tgt_np)
        print(f"  {w_all.shape[0]:,} positions (in-RAM arrays, no disk)", flush=True)
        drop_fn = lambda: None   # RAM-backed — nothing to evict
    else:
        # Pre-featurized .npy cache (from c500_featurize.py). Cache the WHOLE dataset in
        # VRAM — the C500 has 64GB VRAM, the dataset is ~38GB, so it fits with headroom.
        # One-time JuiceFS read, then every batch is a pure GPU gather: no CPU→GPU
        # transfer, no disk reads, no page-cache thrash. This is the fix for slow JuiceFS.
        npy_path = os.environ.get("NNUE_NPY_DIR", "/tmp/code")
        if os.path.exists(os.path.join(npy_path, "feat_w.npy")):
            import numpy as _np
            meta_path = os.path.join(npy_path, "featurized_meta.txt")
            if os.path.exists(meta_path):
                with open(meta_path) as mf:
                    n_valid = int(mf.readline().strip())
            else:
                n_valid = _np.load(os.path.join(npy_path, "feat_s.npy"), mmap_mode='r').shape[0]
            print(f"  streaming {n_valid:,} positions (~38GB) into VRAM...", flush=True)
            t_load = time.time()
            w_all     = torch.empty((n_valid, MAX_PIECES), dtype=torch.int16,   device=device)
            b_all     = torch.empty((n_valid, MAX_PIECES), dtype=torch.int16,   device=device)
            stm_all   = torch.empty(n_valid,               dtype=torch.float32, device=device)
            target_all= torch.empty(n_valid,               dtype=torch.float32, device=device)
            # Parallel JuiceFS read: object storage scales with concurrency, so N workers
            # read disjoint chunks concurrently (CPU-side, GIL released by numpy I/O) while
            # the main thread copies each finished chunk to VRAM. Minimizes the one-time load.
            from concurrent.futures import ThreadPoolExecutor
            WORKERS = int(os.environ.get("NNUE_LOAD_WORKERS", "8"))
            CH = 500_000
            def vram_load(name, dst):
                src = _np.load(os.path.join(npy_path, name), mmap_mode='r')
                ranges = [(i, min(i + CH, n_valid)) for i in range(0, n_valid, CH)]
                def read(ij):
                    i, j = ij
                    return i, torch.from_numpy(_np.array(src[i:j]))   # JuiceFS read happens here
                with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                    futs = [ex.submit(read, ij) for ij in ranges]
                    done = 0
                    for fut in futs:                                   # in order; reads run concurrently
                        i, t = fut.result()
                        end = i + t.shape[0]
                        dst[i:end] = t.to(device)                      # CUDA on main thread only
                        del t
                        done += 1
                        if done % 50 == 0:
                            print(f"    {name}: {end:,}/{n_valid:,} ({time.time()-t_load:.0f}s)", flush=True)
                del src
            vram_load("feat_w.npy", w_all)
            vram_load("feat_b.npy", b_all)
            vram_load("feat_s.npy", stm_all)
            vram_load("feat_t.npy", target_all)
            print(f"  VRAM cache ready in {time.time()-t_load:.0f}s ({n_valid:,} positions, {WORKERS} readers)", flush=True)
            drop_fn = lambda: None   # VRAM-resident — nothing to evict from page cache
        else:
            fens, evals = load_dataset(data, max_positions)
            print(f"  {len(fens)} positions", flush=True)
            cache_path = os.environ.get("NNUE_CACHE", "/tmp/luminex_featurized.pt")
            w_all, b_all, stm_all, target_all = featurize_all(fens, evals, cache_path)
    n = w_all.shape[0]

    model = LNNUE(L1=L1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)   # AdamW: better generalization than plain Adam
    SCALE = 400.0
    bs = batch_size
    steps_per_epoch = (n + bs - 1) // bs
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
                 if lr_schedule else None)
    # SWA: average weights over the last 25% of training -> flatter minimum,
    # better generalization on limited data (SF/PyTorch-standard technique).
    swa_model = None
    swa_start = 0
    if swa and epochs >= 2:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start = steps_per_epoch * epochs * 3 // 4
    gstep = 0
    start_epoch = 0

    # Resume from checkpoint — lets training span multiple 4h sessions.
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            if scheduler is not None and 'sched' in ck: scheduler.load_state_dict(ck['sched'])
            if swa_model is not None and 'swa' in ck: swa_model.load_state_dict(ck['swa'])
            start_epoch = ck['epoch']
            gstep = ck.get('gstep', 0)
            print(f"  RESUMED from checkpoint at epoch {start_epoch} (gstep {gstep})", flush=True)
        except Exception as e:
            print(f"  checkpoint load failed ({e}) — starting fresh", flush=True)
            start_epoch = 0; gstep = 0

    train_t0 = time.time()
    completed_all = False
    for epoch in range(start_epoch, epochs):
        # Time-budget guard: stop before the session window expires; re-running resumes here.
        if time_budget_sec and (time.time() - train_t0) >= time_budget_sec:
            print(f"  time budget {time_budget_sec}s reached before epoch {epoch} — pausing "
                  f"(net saved; re-run to resume from epoch {epoch})", flush=True)
            break
        perm = torch.randperm(n, device=device)   # GPU-side shuffle (data is in VRAM)
        running, window, t0 = 0.0, 0, time.time()
        for s in range(steps_per_epoch):
            idx = perm[s*bs:(s+1)*bs]
            w_idx = w_all[idx].to(device).long()
            b_idx = b_all[idx].to(device).long()
            stm = stm_all[idx].to(device)
            target = target_all[idx].to(device)
            drop_fn()   # evict the just-read memmap pages → keeps RAM low on 40GB cgroup
            opt.zero_grad()
            if use_amp:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    pred = model(w_idx, b_idx, stm)
                    sig_p = torch.sigmoid(pred / SCALE)
                    sig_t = torch.sigmoid(target / SCALE)
                    loss = ((sig_p - sig_t) ** 2).mean()
                loss.backward()
            else:
                pred = model(w_idx, b_idx, stm)
                sig_p = torch.sigmoid(pred / SCALE)
                sig_t = torch.sigmoid(target / SCALE)
                loss = ((sig_p - sig_t) ** 2).mean()
                loss.backward()
            opt.step()
            if scheduler is not None: scheduler.step()
            gstep += 1
            if swa_model is not None and gstep >= swa_start:
                swa_model.update_parameters(model)
            window += 1   # don't call loss.item() every step — it forces a GPU sync and serializes batches
            if s == 0 or (s + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  epoch {epoch} step {s+1}/{steps_per_epoch} "
                      f"loss={loss.item():.5f} ({(s+1)*bs/dt:.0f} pos/s)", flush=True)
                window = 0
        # Checkpoint after each epoch (persistent → survives restart)
        if ckpt_path:
            ck = {'epoch': epoch + 1, 'model': model.state_dict(), 'opt': opt.state_dict(), 'gstep': gstep}
            if scheduler is not None: ck['sched'] = scheduler.state_dict()
            if swa_model is not None: ck['swa'] = swa_model.state_dict()
            torch.save(ck, ckpt_path)
            print(f"  [checkpoint saved: epoch {epoch+1}/{epochs}]", flush=True)
            # ALSO save a usable .nnue net every epoch → if the session is killed mid-training,
            # the latest net still exists next to the checkpoint. Uses SWA weights once in SWA phase.
            _latest = os.path.join(os.path.dirname(ckpt_path), "luminex_v2_latest.nnue")
            _sm = swa_model.module if (swa_model is not None and gstep >= swa_start) else model
            save_nnue(_sm, _latest)
            print(f"  [net saved: epoch {epoch+1} -> {_latest}]", flush=True)
    else:
        completed_all = True   # loop ran to completion (no time-budget break)

    save_model = swa_model.module if (swa_model is not None and gstep >= swa_start) else model
    if save_model is swa_model.module:
        print("  using SWA-averaged weights", flush=True)
    save_nnue(save_model, out)
    print(f"saved {out}", flush=True)
    if completed_all and ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("  training complete — checkpoint cleared", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('data', help='FEN|eval_cp file (e.g. sf_dataset.txt)')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-positions', type=int, default=None)
    ap.add_argument('--out', default='luminex_v1.nnue')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--L1', type=int, default=256)
    ap.add_argument('--lr-schedule', action='store_true')
    ap.add_argument('--swa', action='store_true')
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.batch_size, args.lr, args.device, args.max_positions, args.L1, args.lr_schedule, args.swa)


def save_nnue(model, path):
    """Export weights as float32 binary (C++ loader reads this). Quantization
    to int8/int16 happens in the C++ inference / a separate export step."""
    with open(path, 'wb') as f:
        # header: magic + architecture dims
        f.write(b'LNN1')
        f.write(struct.pack('iiii', model.L1, model.L2, model.L3, NUM_INPUTS))
        for name in ['ft.weight', 'ft.bias', 'l2.weight', 'l2.bias',
                     'l3.weight', 'l3.bias', 'out.weight', 'out.bias']:
            t = dict(model.named_parameters())[name].detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack('i', t.size))
            f.write(t.tobytes())


if __name__ == '__main__':
    main()
