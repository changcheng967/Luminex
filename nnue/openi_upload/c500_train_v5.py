#!/usr/bin/env python3
"""C500: v5 ISOLATION TEST — v2's trainer (NO QAT), 30 epochs. One session, no CLI args.

PURPOSE: isolate whether QAT or epoch-count caused v4's -81 Elo loss to v2.
  v2 = no-QAT trainer, 15 epochs  -> BEST net
  v4 = QAT trainer,    30 epochs  -> -81 Elo vs v2
  v5 = no-QAT trainer, 30 epochs  -> THIS RUN (matches v2 exactly except 2x epochs)
If v5 ~= v2  -> 30 epochs is fine, QAT was the culprit (drop QAT).
If v5 ~= v4  -> 30 epochs overfits regardless of QAT (cap epochs at ~15).

Uses luminex_nnue_train.py (the v2 trainer: pure-float forward, NO fake-quant / QAT).
Everything else matched to v2: L1=512, lr=1e-3, bf16 (AMP=1), cosine LR, SWA.

Quantization is INLINED here (no separate quantize_int8.py dependency — that broke v4's run).

RUN (no arguments):
    python c500_train_v5.py

Files required on /tmp/code:
    luminex_nnue_train.py        (the v2 trainer — no QAT)
    feat_w.npy feat_b.npy feat_s.npy feat_t.npy  (existing 280M cache)
"""
import os, sys, struct
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
sys.path.insert(0, "/tmp/code")
os.environ["NNUE_NPY_DIR"] = "/tmp/code"

from c2net.context import prepare, upload_output
ctx = prepare()
out_dir = ctx.output_path

import torch
print(f"torch {torch.__version__}, cuda: {torch.cuda.is_available()}", flush=True)

if not os.path.exists("/tmp/code/feat_w.npy"):
    print("ERROR: no featurized cache at /tmp/code.", flush=True); sys.exit(1)

import luminex_nnue_train as T   # the v2 trainer (NO QAT)

# ---- v5 = v2 matched except epochs=30. Env-overridable; no CLI args. ----
L1      = int(os.environ.get("NNUE_L1", "512"))    # v2's size
EPOCHS  = int(os.environ.get("NNUE_EPOCHS", "30"))  # 2x v2's 15 (the isolation variable)
BS      = int(os.environ.get("NNUE_BS", "32768"))
LR      = float(os.environ.get("NNUE_LR", "1e-3"))  # v2's lr
BUDGET  = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.7 * 3600))))
out_file = os.path.join(out_dir, "luminex_v5.nnue")
i8_file  = os.path.join(out_dir, "luminex_v5_i8.nnue")

print(f"\nv5 ISOLATION: v2 trainer (NO QAT), {EPOCHS} epochs, L1={L1}, lr={LR}, bf16, SWA.", flush=True)
print("If v5~=v2 -> QAT was the culprit. If v5~=-81 -> 30ep overfits (cap at 15).", flush=True)

T.train(
    data=None, out=out_file,
    epochs=EPOCHS, batch_size=BS, lr=LR, device="cuda",
    L1=L1, lr_schedule=True, swa=True,           # cosine + SWA (matches v2)
    time_budget_sec=BUDGET,
    ckpt_path="/tmp/code/v5_ckpt.pt",
    use_amp=True,                                 # bf16 (matches v2)
)


# ---- INLINE int8 quantization (no external script dependency) ----
def quantize_i8(src, dst):
    import numpy as np
    def rt(f):
        (n,) = struct.unpack('i', f.read(4)); return np.frombuffer(f.read(n*4), dtype=np.float32).copy()
    def q(w):
        s = 127.0 / max(float(np.abs(w).max()), 1e-8)
        return np.round(w*s).clip(-127, 127).astype(np.int8), np.float32(s)
    with open(src, 'rb') as f:
        assert f.read(4) == b'LNN1', "not LNN1"
        L1, L2, L3, NI = struct.unpack('iiii', f.read(16))
        ft_w = rt(f).reshape(L1, NI); ft_b = rt(f)
        l2_w = rt(f).reshape(L2, 2*L1); l2_b = rt(f)
        l3_w = rt(f).reshape(L3, L2);  l3_b = rt(f)
        out_w = rt(f).reshape(1, L3);  out_b = rt(f)[0]
    l2w, s2 = q(l2_w); l3w, s3 = q(l3_w); outw, so = q(out_w)
    with open(dst, 'wb') as f:
        f.write(b'LNI8'); f.write(struct.pack('iiii', L1, L2, L3, NI))
        f.write(struct.pack('i', ft_w.size)); f.write(ft_w.astype(np.float32).tobytes())
        f.write(struct.pack('i', ft_b.size)); f.write(ft_b.astype(np.float32).tobytes())
        for w8, b, s in [(l2w, l2_b, s2), (l3w, l3_b, s3), (outw, np.array([out_b], np.float32), so)]:
            f.write(struct.pack('i', w8.size)); f.write(w8.tobytes())
            f.write(struct.pack('i', b.size)); f.write(b.astype(np.float32).tobytes())
            f.write(struct.pack('f', float(s)))
    print(f"  [inlined quantize] wrote {dst}: L1={L1} L2={L2} L3={L3}", flush=True)

print(f"\nQuantizing {out_file} -> int8 (inlined)...", flush=True)
quantize_i8(out_file, i8_file)

print(f"\nUploading nets from {out_dir}...", flush=True)
upload_output()
print("DONE! — luminex_v5_i8.nnue (30 ep, NO QAT). Bench vs v2: ~=v2 means QAT was the issue.", flush=True)
