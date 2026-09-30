#!/usr/bin/env python3
"""Luminex V13 model (amended spec): factorized FT (8 buckets + shared),
structural side features, ClippedReLU, tail (16, 64), no FM interaction.

Same data pipeline as gen768 (identical 24576 HalfKAv2 index space — the
factorization remap lives inside the model), so the c500 stream trainer
drives it unchanged apart from the model class and the export call.
"""
import math
import numpy as np
import torch
import torch.nn as nn

from luminex_nnue_train import NUM_INPUTS

FT_FACTOR = 8
FT_ROWS = FT_FACTOR * 768 + 768          # effective buckets + shared
L1_DIM = 512
TAIL = (16, 64)
OUT_SCALE = 300.0


def side_features(w_idx):
    """Derive the 33 structural side features from white-pov feature rows.

    plane = (idx % 768) // 64: 0=wp 1=bp ... 10=wk 11=bk (orientation-frame
    squares — consistent per perspective, mirror-invariant as inputs).
    """
    valid = (w_idx != NUM_INPUTS).float()
    planes = ((w_idx % 768) // 64).clamp(0, 11)
    counts = torch.zeros(w_idx.shape[0], 12, device=w_idx.device).scatter_add_(
        1, planes, valid)
    sq = w_idx & 63
    f = (sq & 7).clamp(0, 7)
    wfiles = torch.zeros(w_idx.shape[0], 8, device=w_idx.device).scatter_add_(
        1, f, (planes == 0).float() * valid)
    bfiles = torch.zeros(w_idx.shape[0], 8, device=w_idx.device).scatter_add_(
        1, f, (planes == 1).float() * valid)
    wk = (sq * (planes == 10).long()).sum(1)
    bk = (sq * (planes == 11).long()).sum(1)
    men = (w_idx != NUM_INPUTS).sum(1)
    return torch.cat([
        counts / 8.0, wfiles, bfiles,
        (wk & 7).unsqueeze(1).float() / 7.0, (wk >> 3).unsqueeze(1).float() / 7.0,
        (bk & 7).unsqueeze(1).float() / 7.0, (bk >> 3).unsqueeze(1).float() / 7.0,
        men.unsqueeze(1).float() / 32.0], dim=1)


class V13NNUE(nn.Module):
    def __init__(self, dual_head=True):
        super().__init__()
        self.dual_head = dual_head
        self.ft = nn.EmbeddingBag(FT_ROWS + 1, L1_DIM, mode="sum",
                                  padding_idx=FT_ROWS)
        self.ft_bias = nn.Parameter(torch.zeros(L1_DIM))
        nn.init.normal_(self.ft.weight, std=0.2)
        self.ft.weight.data[FT_ROWS].zero_()
        self.side_proj = nn.Linear(33, 32)
        nn.init.normal_(self.side_proj.weight, std=0.05)
        nn.init.zeros_(self.side_proj.bias)
        self.l2 = nn.Linear(2 * L1_DIM + 32, TAIL[0])
        self.l3 = nn.Linear(TAIL[0], TAIL[1])
        self.out = nn.Linear(TAIL[1], 1)
        if dual_head:
            self.lin = nn.Embedding(NUM_INPUTS + 1, 1, padding_idx=NUM_INPUTS)
            nn.init.zeros_(self.lin.weight)

    def _acc(self, idx):
        valid = idx < NUM_INPUTS
        plane = idx % 768
        bkt = (idx // 768).clamp(max=FT_FACTOR - 1)
        pad = FT_ROWS
        base = torch.where(valid, bkt * 768 + plane, torch.full_like(idx, pad))
        shared = torch.where(valid, torch.full_like(idx, FT_FACTOR * 768) + plane,
                             torch.full_like(idx, pad))
        return self.ft(base) + self.ft(shared) + self.ft_bias

    def forward(self, w_idx, b_idx, stm):
        acc_w = self._acc(w_idx)
        acc_b = self._acc(b_idx)
        m = stm.view(-1, 1).float()
        side = side_features(w_idx)
        h = torch.cat([m * acc_w + (1 - m) * acc_b,
                       (1 - m) * acc_w + m * acc_b,
                       torch.clamp(self.side_proj(side), 0.0, 1.0)], dim=1)
        h = torch.clamp(h, 0.0, 1.0)
        h = torch.clamp(self.l2(h), 0.0, 1.0)
        h = torch.clamp(self.l3(h), 0.0, 1.0)
        return self.out(h).squeeze(-1) * OUT_SCALE

    def probe_ft(self, device):
        """v7-postmortem guard: the active FT path must equal a manual
        per-bag sum of the same embedding rows."""
        idx = torch.randint(0, NUM_INPUTS, (64, 32), device=device)
        idx[:, -5:] = FT_ROWS
        bag = self._acc(idx)
        manual = (self.ft.weight[((idx // 768).clamp(max=FT_FACTOR - 1) * 768
                                  + idx % 768)].sum(1)
                  + self.ft.weight[FT_FACTOR * 768 + idx % 768].sum(1)
                  + self.ft_bias)
        d = (bag - manual).abs().max().item()
        assert d < 1e-4, f"probe_ft drift {d}"
        return d


def warm_start_from_gen768(v13, ckpt_path):
    """Initialize V13 from a 768-wide gen768 checkpoint (best-effort,
    function-approximate): SVD column shrink of the FT, tail remap,
    mean-split factorization of the lin head."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    W = sd["ft.weight"][:NUM_INPUTS]              # [24576, 768]
    b = sd["ft_bias"]                             # [768]
    # rank-512 least-squares shrink of the accumulator column space
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Vr = Vh[:L1_DIM].T                            # [768, 512]
    W512 = (W @ Vr).numpy()
    b512 = (b @ Vr)
    ft_new = np.zeros((FT_ROWS + 1, L1_DIM), dtype=np.float32)
    plane = (np.arange(NUM_INPUTS) % 768)
    bkt = np.minimum(np.arange(NUM_INPUTS) // 768, FT_FACTOR - 1)
    np.add.at(ft_new, bkt * 768 + plane, W512)
    shared = W512.mean(axis=0)
    np.subtract.at(ft_new, bkt * 768 + plane, shared[None, :])
    ft_new[FT_FACTOR * 768:FT_ROWS] = shared
    with torch.no_grad():
        v13.ft.weight.copy_(torch.from_numpy(ft_new))
        v13.ft_bias.copy_(b512)
        l2w = sd["l2.weight"]                     # [L2, 1536]
        half = l2w.shape[1] // 2
        new_l2 = torch.zeros(TAIL[0], 2 * L1_DIM + 32)
        new_l2[:, :L1_DIM] = l2w[:, :half] @ Vr
        new_l2[:, L1_DIM:2 * L1_DIM] = l2w[:, half:] @ Vr
        v13.l2.weight.copy_(new_l2)
        v13.l2.bias.copy_(sd["l2.bias"])
        # tail shape mismatch (old L3=32 vs V13 64): pad function-preservingly —
        # new l3 rows / out columns are zero, so the extra dims contribute nothing
        old_l3w, old_l3b = sd["l3.weight"], sd["l3.bias"]
        l3w = torch.zeros(TAIL[1], TAIL[0]); l3w[:old_l3w.shape[0]] = old_l3w
        l3b = torch.zeros(TAIL[1]); l3b[:old_l3b.shape[0]] = old_l3b
        v13.l3.weight.copy_(l3w); v13.l3.bias.copy_(l3b)
        old_ow, old_ob = sd["out.weight"], sd["out.bias"]
        ow = torch.zeros(1, TAIL[1]); ow[:, :old_ow.shape[1]] = old_ow
        v13.out.weight.copy_(ow); v13.out.bias.copy_(old_ob)
        if v13.dual_head and "lin.weight" in sd:
            v13.lin.weight.copy_(sd["lin.weight"])   # stays full-index-space
    return v13


def export_nnue(model, path):
    """V13 export: header + factorized FT + side projection + tail + lin.

    Layout (all little-endian):
      magic 'LXV3' | i32 version=1 | i32 L1 | i32 l2 | i32 l3 | i32 ft_factor
      FT rows [ft_factor*768+768, L1] float | ft_bias [L1] float
      side_proj w [32, 33] float | side_proj b [32] float
      l2/l3/out weights + biases float
      'LINH' + lin rows [ft_factor*768+768] float  (optional)
    The engine-side quantization constants live with the loader contract.
    """
    m = model.cpu().eval()
    import struct
    with open(path, "wb") as f:
        f.write(b"LXV3")
        f.write(struct.pack("<iiiii", 1, L1_DIM, TAIL[0], TAIL[1], FT_FACTOR))
        f.write(m.ft.weight.detach().numpy().astype(np.float32).tobytes())
        f.write(m.ft_bias.detach().numpy().astype(np.float32).tobytes())
        f.write(m.side_proj.weight.detach().numpy().astype(np.float32).tobytes())
        f.write(m.side_proj.bias.detach().numpy().astype(np.float32).tobytes())
        for lin in (m.l2, m.l3, m.out):
            f.write(lin.weight.detach().numpy().astype(np.float32).tobytes())
            f.write(lin.bias.detach().numpy().astype(np.float32).tobytes())
        if m.dual_head:
            f.write(b"LINH")
            f.write(m.lin.weight.detach().numpy().astype(np.float32).tobytes())
    print(f"v13 export: {path} (L1={L1_DIM} tail={TAIL} ft_factor={FT_FACTOR}"
          f"{' +lin' if m.dual_head else ''})", flush=True)
