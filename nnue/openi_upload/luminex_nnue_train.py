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
try:
    import chess          # only needed by active_features/featurize_batch (FEN-text path);
except ImportError:       # the C500 streaming trainer uses the C featurizer instead.
    chess = None
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
        # Sparse accumulator: gather active feature weights, sum. A zero-weight
        # row at index NUM_INPUTS makes padding contribute nothing.
        ft_w = torch.cat([self.ft.weight.t(),
                          torch.zeros(1, self.L1, device=w_idx.device)], dim=0)
        acc_w = ft_w[w_idx].sum(dim=1) + self.ft.bias   # (B, L1)
        acc_b = ft_w[b_idx].sum(dim=1) + self.ft.bias
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
def train(data, out, epochs=1, batch_size=16384, lr=1e-3, device='cuda', max_positions=None, L1=256, lr_schedule=False, swa=False):
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"device: {device} | L1={L1} | lr_schedule={lr_schedule} | swa={swa}", flush=True)

    print("loading + pre-featurizing data (one-time; epochs after are GPU-fast)...", flush=True)

    # Check for pre-featurized .npy memmap files (from c500_featurize.py)
    # These load with ZERO RAM copy (disk-backed via mmap)
    npy_path = os.environ.get("NNUE_NPY_DIR", "/tmp/code")
    if os.path.exists(os.path.join(npy_path, "feat_w.npy")):
        print("  loading .npy memmap cache (zero-RAM)...", flush=True)
        import numpy as _np
        # Read metadata
        meta_path = os.path.join(npy_path, "featurized_meta.txt")
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                n_valid = int(mf.readline().strip())
        else:
            n_valid = _np.load(os.path.join(npy_path, "feat_s.npy"), mmap_mode='r').shape[0]
        w_all = torch.from_numpy(_np.load(os.path.join(npy_path, "feat_w.npy"), mmap_mode='r')[:n_valid])
        b_all = torch.from_numpy(_np.load(os.path.join(npy_path, "feat_b.npy"), mmap_mode='r')[:n_valid])
        stm_all = torch.from_numpy(_np.load(os.path.join(npy_path, "feat_s.npy"), mmap_mode='r')[:n_valid])
        target_all = torch.from_numpy(_np.load(os.path.join(npy_path, "feat_t.npy"), mmap_mode='r')[:n_valid])
        print(f"  {n_valid:,} positions (memmap, ~0 RAM)", flush=True)
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

    for epoch in range(epochs):
        perm = torch.randperm(n)
        running, window, t0 = 0.0, 0, time.time()
        for s in range(steps_per_epoch):
            idx = perm[s*bs:(s+1)*bs]
            w_idx = w_all[idx].to(device).long()
            b_idx = b_all[idx].to(device).long()
            stm = stm_all[idx].to(device)
            target = target_all[idx].to(device)
            pred = model(w_idx, b_idx, stm)
            sig_p = torch.sigmoid(pred / SCALE)
            sig_t = torch.sigmoid(target / SCALE)
            loss = ((sig_p - sig_t) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler is not None: scheduler.step()
            gstep += 1
            if swa_model is not None and gstep >= swa_start:
                swa_model.update_parameters(model)
            running += loss.item(); window += 1
            if s == 0 or (s + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  epoch {epoch} step {s+1}/{steps_per_epoch} "
                      f"loss={running/window:.5f} ({(s+1)*bs/dt:.0f} pos/s)", flush=True)
                running = 0.0; window = 0

    save_model = swa_model.module if (swa_model is not None and gstep >= swa_start) else model
    if save_model is swa_model.module:
        print("  using SWA-averaged weights", flush=True)
    save_nnue(save_model, out)
    print(f"saved {out}", flush=True)
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
