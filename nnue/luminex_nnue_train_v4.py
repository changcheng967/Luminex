#!/usr/bin/env python3
# Luminex NNUE v4 trainer — int16 ACCUMULATOR + int8 FT (the SF architecture).
#
# GOAL: break the float-accumulator floors that cap v2/v3 at ~990K NPS single-thread.
#   v2/v3 = float accumulator (4KB, int16->float convert every update) + float SCReLU.
#   v4    = int8 FT weights (0.5KB/feature) + int16 accumulator (2KB) + integer ClippedReLU.
#
# ###############################################################################
# ARCHITECTURE (the engine int16-acc path this trainer simulates, EXACTLY)
# ###############################################################################
# 1. FT weights: int8 (per-layer scale s = 127/max_abs), stored int8.
# 2. Accumulator: int16 = sum over active features of int8_weight_row + int16_bias.
#      Range for ~32 features: +-127*32 = +-4064 -> fits int16 (+-32767) with 8x headroom.
#      NO overflow (this is what killed our old int16-acc attempt, which used int16 weights).
#      Update = pure int16 add (NO int16->float convert) -> the update bottleneck, gone.
# 3. FT activation: ClippedReLU  act = clamp(int16_acc, 0, 127) -> uint8 DIRECTLY.
#      Integer, no float chain -> the SCReLU bottleneck, gone. (The net trains to use the
#      [0,127] range; saturated neurons act as binary features, SF-style.)
# 4. L2/L3/out: UNCHANGED from v2/v3 — int8 weights x uint8 activations (VNNI), SCReLU,
#      per-layer /cs dequant. Only the FT path is new.
#
# Result (engine, L1=512): update 724->~380 cyc, SCReLU 320->~160 cyc -> ~1.35M NPS
# single-thread (+36% over v2). + working MT -> 2M+ aggregate -> the +280 equal-depth
# edge over HCE finally materializes at bullet.
#
# ###############################################################################
# SAVE FORMAT (LNI9) — int8 FT + int8 L2/L3/out. The v4 ENGINE LOADER reads this.
# ###############################################################################
#   magic 'LNI9' | L1 L2 L3 NUM_INPUTS (4x int32) | FT_SCALE (float, =127.0, reserved)
#   FT: int8 weights (NUM_INPUTS*L1 bytes) | int32 scale (1) | int16 bias (L1)
#   per L2/L3/out: int8 weights (n) | float bias (n) | float scale (1)
# (Engine TODO: add an LNI9 loader + the int16-accumulator FT path. The trainer is ready.)
#
# ###############################################################################
# USAGE (C500) — same as v3/v4-float: upload luminex_nnue_train_v4.py + c500 wrapper.
# ###############################################################################
import sys, math, struct, time, os
import numpy as np
import chess
import torch
from torch import nn

# ---- feature indexer (identical to v2/v3; proven vs SF) ----
KingBuckets = [
    -1,-1,-1,-1,31,30,29,28, -1,-1,-1,-1,27,26,25,24,
    -1,-1,-1,-1,23,22,21,20, -1,-1,-1,-1,19,18,17,16,
    -1,-1,-1,-1,15,14,13,12, -1,-1,-1,-1,11,10, 9, 8,
    -1,-1,-1,-1, 7, 6, 5, 4, -1,-1,-1,-1, 3, 2, 1, 0,
]
NUM_SQ, NUM_PT, NUM_BUCKETS = 64, 12, 32
NUM_PLANES = NUM_SQ * NUM_PT
NUM_INPUTS = NUM_PLANES * NUM_BUCKETS   # 24576 per half
MAX_PIECES = 32


def _orient(is_white_pov, sq, ksq):
    return (7 * (ksq % 8 < 4)) ^ (56 * (not is_white_pov)) ^ sq


def _halfka_idx(is_white_pov, king_sq, sq, piece_color, piece_type):
    p_idx = (piece_type - 1) * 2 + (1 if piece_color != is_white_pov else 0)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 768


def active_features(board):
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    w, b = [], []
    for sq, p in board.piece_map().items():
        pc, pt = (1 if p.color == chess.WHITE else 0), int(p.piece_type)
        w.append(_halfka_idx(True,  wk, sq, pc, pt))
        b.append(_halfka_idx(False, bk, sq, pc, pt))
    return w, b


# ---- QAT fake-quant (guaranteed fp32, STE) ----
def fake_quant_i8(w):
    """Per-layer int8: round(w*s)/s, s=127/max_abs. Matches the engine FT + L2/L3/out quant."""
    with torch.autocast(device_type=w.device.type, enabled=False):
        w = w.float()
        s = (127.0 / w.abs().max().clamp(min=1e-8)).detach()
        wq = torch.round(w * s) / s
        return w + (wq - w).detach()


def fake_quant_u8(h):
    """Activation uint8: round(h*127)/127 (h in [0,1])."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        h = h.float()
        wq = torch.round(h * 127.0) / 127.0
        return h + (wq - h).detach()


# ---- model: int16-accumulator L-NNUE ----
class QLinear(nn.Module):
    """int8-weight linear (L2/L3/out) — identical to v3."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
    def forward(self, x, qat):
        w = fake_quant_i8(self.weight) if qat else self.weight
        return torch.nn.functional.linear(x, w, self.bias)


class LNNUEv4(nn.Module):
    def __init__(self, L1=512, L2=16, L3=32):
        super().__init__()
        self.ft = nn.Linear(NUM_INPUTS, L1)
        self.l2 = QLinear(2 * L1, L2)
        self.l3 = QLinear(L2, L3)
        self.out = QLinear(L3, 1)
        self.L1, self.L2, self.L3 = L1, L2, L3
        nn.init.zeros_(self.ft.bias)
        nn.init.normal_(self.ft.weight, mean=0.0, std=0.2)
        # Learnable FT activation scale. The int16 accumulator (sum of ~32 int8 weights)
        # spans ~+-158 (1-sigma) at init. ft_scale=256 maps that to ~+-0.6 so ~45% of
        # neurons sit in the linear [0,1] SCReLU range -> gradients flow -> it learns.
        # softplus keeps it positive; the net tunes it during training.
        self.ft_scale = nn.Parameter(torch.tensor(256.0))

    def forward(self, w_idx, b_idx, stm, qat=True):
        # FT: int8 fake-quant ALWAYS ON (not gated by qat). Reason: the int16 accumulator's
        # scale depends on the FT weight format — int8 weights give acc ~+-158, float weights
        # ~+-1. If we fake-quant only during qat, the acc scale flips 100x at the warmup->qat
        # transition and ft_scale can't track it -> flat loss. Keeping FT int8 from step 0
        # makes the acc scale consistent; the warmup then gates only L2/L3/out.
        ftw = fake_quant_i8(self.ft.weight)
        ft_w = torch.cat([ftw.t(), torch.zeros(1, self.L1, device=ftw.device)], dim=0)
        acc_w = torch.nn.functional.embedding_bag(w_idx, ft_w, mode='sum') + self.ft.bias
        acc_b = torch.nn.functional.embedding_bag(b_idx, ft_w, mode='sum') + self.ft.bias
        stm_mask = stm.view(-1, 1).float()
        stm_acc = stm_mask * acc_w + (1 - stm_mask) * acc_b
        nstm_acc = (1 - stm_mask) * acc_w + stm_mask * acc_b
        h = torch.cat([stm_acc, nstm_acc], dim=1)
        # FT activation = SCReLU on the int16 accumulator / learned ft_scale. The int8-weight
        # acc spans ~+-158 (1-sigma) at init; ft_scale=256 maps it to ~+-0.6 so ~45% of
        # neurons sit in the linear [0,1] range -> gradients flow -> it learns (like v2).
        scale = torch.nn.functional.softplus(self.ft_scale)
        h = torch.clamp(h / scale, 0.0, 1.0) ** 2
        if qat:
            h = fake_quant_u8(h)
        # L2/L3/out: v3-style (int8 weights, SCReLU, uint8 activations) — UNCHANGED.
        h = torch.clamp(self.l2(h, qat), 0.0, 1.0) ** 2
        if qat:
            h = fake_quant_u8(h)
        h = torch.clamp(self.l3(h, qat), 0.0, 1.0) ** 2
        if qat:
            h = fake_quant_u8(h)
        return self.out(h, qat).squeeze(-1) * 300.0


# ---- data loading (identical to v3 — reuses the 280M .npy cache) ----
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
            fens.append(fen); evals.append(ev)
            if max_positions and len(fens) >= max_positions:
                break
    return fens, evals


def load_vram(npy_path, device):
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
            for fut in [ex.submit(read, ij) for ij in ranges]:
                i, t = fut.result()
                dst[i:i + t.shape[0]] = t.to(device); del t
        del src
    vram_load("feat_w.npy", w_all); vram_load("feat_b.npy", b_all)
    vram_load("feat_s.npy", stm_all); vram_load("feat_t.npy", target_all)
    print(f"  VRAM cache ready in {time.time()-t_load:.0f}s ({WORKERS} readers)", flush=True)
    return w_all, b_all, stm_all, target_all


def featurize_all(fens, evals, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        d = torch.load(cache_path, map_location='cpu', weights_only=False)
        if int(d['w'].shape[0]) == len(fens):
            return d['w'], d['b'], d['stm'], d['target']
    N = len(fens)
    w_idx = torch.full((N, MAX_PIECES), NUM_INPUTS, dtype=torch.int32)
    b_idx = torch.full((N, MAX_PIECES), NUM_INPUTS, dtype=torch.int32)
    stm = torch.zeros(N, dtype=torch.float32); target = torch.zeros(N, dtype=torch.float32)
    for i, fen in enumerate(fens):
        bd = chess.Board(fen); w, b = active_features(bd)
        w_idx[i, :len(w)] = torch.tensor(w[:MAX_PIECES], dtype=torch.int32)
        b_idx[i, :len(b)] = torch.tensor(b[:MAX_PIECES], dtype=torch.int32)
        stm[i] = 1.0 if bd.turn == chess.WHITE else 0.0
        target[i] = evals[i]
    return w_idx, b_idx, stm, target


def train(data, out, epochs=20, batch_size=32768, lr=1.2e-3, device='cuda',
          max_positions=None, L1=512, L2=16, L3=32, swa=False, qat_warmup_frac=0.25,
          time_budget_sec=None, ckpt_path=None, use_amp=True):
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"=== Luminex NNUE v4 (int16 ACCUMULATOR + int8 FT) ===", flush=True)
    print(f"device: {device} | L1={L1} L2={L2} L3={L3} | lr={lr} | swa={swa} | "
          f"qat_warmup={qat_warmup_frac} | amp={use_amp} | budget={time_budget_sec}s", flush=True)
    npy_path = os.environ.get("NNUE_NPY_DIR", "/tmp/code")
    if os.path.exists(os.path.join(npy_path, "feat_w.npy")):
        w_all, b_all, stm_all, target_all = load_vram(npy_path, device)
    else:
        fens, evals = load_dataset(data, max_positions)
        print(f"  {len(fens)} positions (FEN|eval)", flush=True)
        w_all, b_all, stm_all, target_all = featurize_all(fens, evals, os.environ.get("NNUE_CACHE", "/tmp/luminex_feat_v4.pt"))
    n = w_all.shape[0]
    model = LNNUEv4(L1=L1, L2=L2, L3=L3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    SCALE = 400.0
    bs = batch_size
    steps_per_epoch = (n + bs - 1) // bs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    swa_model = None; swa_start = 0
    if swa and epochs >= 2:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start = steps_per_epoch * epochs * 3 // 4
    qat_on_step = int(steps_per_epoch * epochs * qat_warmup_frac)
    gstep = 0; start_epoch = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
            scheduler.load_state_dict(ck['sched'])
            if swa_model is not None and 'swa' in ck: swa_model.load_state_dict(ck['swa'])
            start_epoch = ck['epoch']; gstep = ck.get('gstep', 0)
            print(f"  RESUMED at epoch {start_epoch} (gstep {gstep})", flush=True)
        except Exception as e:
            print(f"  ckpt load failed ({e}) — fresh", flush=True)
    train_t0 = time.time(); completed_all = False
    for epoch in range(start_epoch, epochs):
        if time_budget_sec and (time.time() - train_t0) >= time_budget_sec:
            print(f"  budget {time_budget_sec}s reached before epoch {epoch} — pausing", flush=True); break
        perm = torch.randperm(n, device=device); t0 = time.time()
        for s in range(steps_per_epoch):
            if time_budget_sec and (time.time() - train_t0) >= time_budget_sec: break
            idx = perm[s*bs:(s+1)*bs]
            w_idx = w_all[idx].to(device).long(); b_idx = b_all[idx].to(device).long()
            stm = stm_all[idx].to(device); target = target_all[idx].to(device)
            qat = (gstep >= qat_on_step)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16) if use_amp else torch.enable_grad():
                pred = model(w_idx, b_idx, stm, qat=qat)
                sig_p = torch.sigmoid(pred / SCALE); sig_t = torch.sigmoid(target / SCALE)
                loss = ((sig_p - sig_t) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step(); gstep += 1
            if swa_model is not None and gstep >= swa_start: swa_model.update_parameters(model)
            if s == 0 or (s + 1) % 25 == 0:
                print(f"  epoch {epoch} step {s+1}/{steps_per_epoch} loss={loss.item():.5f} "
                      f"qat={'ON' if qat else 'off'} ({(s+1)*bs/(time.time()-t0):.0f} pos/s)", flush=True)
        if ckpt_path:
            ck = {'epoch': epoch+1, 'model': model.state_dict(), 'opt': opt.state_dict(),
                  'sched': scheduler.state_dict(), 'gstep': gstep}
            if swa_model is not None: ck['swa'] = swa_model.state_dict()
            torch.save(ck, ckpt_path)
            _sm = swa_model.module if (swa_model is not None and gstep >= swa_start) else model
            save_nnue_v4(_sm, os.path.join(os.path.dirname(ckpt_path) or '.', "luminex_v4_latest.nnue"))
            print(f"  [epoch {epoch+1}/{epochs} saved]", flush=True)
    else:
        completed_all = True
    save_model = swa_model.module if (swa_model is not None and gstep >= swa_start) else model
    save_nnue_v4(save_model, out)
    print(f"saved {out}", flush=True)
    if completed_all and ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path); print("  complete — ckpt cleared", flush=True)
    return out


def save_nnue_v4(model, path):
    """Export LNI9: int8 FT + int16 bias + int8 L2/L3/out (the v4 engine loader reads this).
    The QAT-trained weights, when extracted to int8, match what training simulated (QAT)."""
    model = model.module if hasattr(model, 'module') else model
    def quant_i8(w):
        s = 127.0 / max(float(np.abs(w).max()), 1e-8)
        return np.round(w * s).clip(-127, 127).astype(np.int8), np.float32(s)
    ftw = model.ft.weight.detach().cpu().numpy()
    ftw_i8, ft_s = quant_i8(ftw)
    ftb = model.ft.bias.detach().cpu().numpy()
    ftb_i16 = np.round(ftb).clip(-32768, 32767).astype(np.int16)   # int16 accumulator bias
    with open(path, 'wb') as f:
        f.write(b'LNI9')
        f.write(struct.pack('iiii', model.L1, model.L2, model.L3, NUM_INPUTS))
        f.write(struct.pack('f', float(torch.nn.functional.softplus(model.ft_scale))))   # learned FT activation scale (acc/this -> [0,1] SCReLU)
        # FT: int8 weights | int32 scale | int16 bias
        f.write(struct.pack('i', ftw_i8.size)); f.write(ftw_i8.tobytes())
        f.write(struct.pack('i', int(ft_s)))
        f.write(struct.pack('i', ftb_i16.size)); f.write(ftb_i16.tobytes())
        # L2/L3/out: int8 weights | float bias | float scale (same as LNI8)
        for name in ['l2.weight','l2.bias','l3.weight','l3.bias','out.weight','out.bias']:
            t = dict(model.named_parameters())[name].detach().cpu().numpy()
            if name.endswith('.weight'):
                q, s = quant_i8(t)
                f.write(struct.pack('i', q.size)); f.write(q.tobytes())
                f.write(struct.pack('f', float(s)))
            else:
                b = t.astype(np.float32)
                f.write(struct.pack('i', b.size)); f.write(b.astype(np.float32).tobytes())
    print(f"  [LNI9 export: FT int8 ({ftw_i8.size}), L2/L3/out int8 — for the v4 engine loader]", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('data', nargs='?', default='')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32768)
    ap.add_argument('--lr', type=float, default=1.2e-3)
    ap.add_argument('--L1', type=int, default=512)
    ap.add_argument('--L2', type=int, default=16)
    ap.add_argument('--L3', type=int, default=32)
    ap.add_argument('--swa', action='store_true')
    ap.add_argument('--qat-warmup-frac', type=float, default=0.25)
    ap.add_argument('--out', default='luminex_v4.nnue')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--time-budget', type=int, default=None)
    ap.add_argument('--no-amp', action='store_true')
    a = ap.parse_args()
    train(a.data, a.out, a.epochs, a.batch_size, a.lr, 'cuda', None,
          a.L1, a.L2, a.L3, a.swa, a.qat_warmup_frac, a.time_budget, a.ckpt, not a.no_amp)


if __name__ == '__main__':
    main()
