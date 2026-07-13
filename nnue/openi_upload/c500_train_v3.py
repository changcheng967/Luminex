#!/usr/bin/env python3
"""C500: train v3 NNUE (Net A: L1=256 L2=32 L3=64, QAT) from the featurized cache on
/tmp/code — ONE session, NO command-line arguments.

This is the v3 twin of c500_train.py (the v2 fast path). It:
  - reads the .npy featurized cache straight from /tmp/code (persistent JuiceFS) — NOT a text
    dataset; data=None, exactly like v2,
  - VRAM-caches the whole ~38GB dataset once (the fast loader in luminex_nnue_train_v3.load_vram,
    identical to v2's), then trains GPU-fast with embedding_bag + bf16,
  - bakes in Net A (L1=256 L2=32 L3=64, QAT, SWA, cosine LR) as DEFAULTS — env-overridable but
    no CLI args needed,
  - quantizes the result to int8 in-place, then uploads BOTH the float and int8 nets via c2net.
  - time-budget (~3.7h) guarantees it never exceeds the 4h session; saves a net every epoch.

RUN (no arguments):
    python c500_train_v3.py

Files required on /tmp/code (upload alongside this script):
    luminex_nnue_train_v3.py   (the trainer)
    quantize_int8.py           (the int8 converter, unchanged from v2)
    feat_w.npy feat_b.npy feat_s.npy feat_t.npy  (the existing 280M featurized cache)
"""
import os, sys, subprocess
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
sys.path.insert(0, "/tmp/code")
os.environ["NNUE_NPY_DIR"] = "/tmp/code"   # read featurized data directly (persistent)

from c2net.context import prepare, upload_output
ctx = prepare()
out_dir = ctx.output_path

import torch
print(f"torch {torch.__version__}, cuda: {torch.cuda.is_available()}", flush=True)

if not os.path.exists("/tmp/code/feat_w.npy"):
    print("ERROR: no featurized cache at /tmp/code — run c500_featurize.py first.", flush=True)
    sys.exit(1)

import luminex_nnue_train_v3 as T

# ---- Net A defaults (env-overridable; no CLI args) ----
L1      = int(os.environ.get("NNUE_L1", "256"))
L2      = int(os.environ.get("NNUE_L2", "32"))
L3      = int(os.environ.get("NNUE_L3", "64"))
EPOCHS  = int(os.environ.get("NNUE_EPOCHS", "30"))
BS      = int(os.environ.get("NNUE_BS", "32768"))
LR      = float(os.environ.get("NNUE_LR", "1.2e-3"))
BUDGET  = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.7 * 3600))))   # stop before 4h session limit
QATWARM = float(os.environ.get("NNUE_QAT_WARMUP", "0.25"))
out_file   = os.path.join(out_dir, "luminex_v3.nnue")
i8_file    = os.path.join(out_dir, "luminex_v3_i8.nnue")

print(f"\nTraining v3 (Net A) from /tmp/code: L1={L1} L2={L2} L3={L3} QAT-warmup={QATWARM} "
      f"epochs={EPOCHS} bs={BS} lr={LR} bf16=ON budget={BUDGET}s (~{BUDGET/3600:.1f}h)", flush=True)

T.train(
    data=None, out=out_file,                # data=None: uses the .npy VRAM cache, not a text file
    epochs=EPOCHS, batch_size=BS, lr=LR, device="cuda",
    L1=L1, L2=L2, L3=L3, swa=True, qat_warmup_frac=QATWARM,
    time_budget_sec=BUDGET,
    ckpt_path="/tmp/code/v3_ckpt.pt",       # crash-safety only (saved each epoch, resumable)
    use_amp=True,                           # bf16 — max FLOPS on C500 (QAT fake-quant stays fp32)
)

# ---- quantize float -> int8 in-place (unchanged quantize_int8.py from v2) ----
print(f"\nQuantizing {out_file} -> int8...", flush=True)
subprocess.run([sys.executable, "/tmp/code/quantize_int8.py", out_file, i8_file], check=True)

print(f"\nUploading nets from {out_dir}...", flush=True)
upload_output()
print("DONE! — luminex_v3_i8.nnue is the net to load on hyperai.", flush=True)
