#!/usr/bin/env python3
"""C500: train v4 NNUE for RAW STRENGTH (L1=1024, float) — surpass v2, approach 3000 Elo.
ONE session, NO command-line arguments.

v4-raw = v2's architecture with 2x the feature-transformer capacity (L1 512->1024, SF's size),
trained FLOAT (no int8-FT — that loses raw strength). The bigger L1 is the single biggest
raw-strength (equal-depth eval quality) lever. L2/L3 kept at v2 sizes to avoid overfitting
on the 280M dataset (the real ceiling for 3000 is more DATA, not just bigger L1).

Uses luminex_nnue_train_v3.py (the FLOAT trainer that LEARNS — int16 FT QAT + SCReLU on a
float accumulator + int8 L2/L3/out QAT). Just L1=1024.

Engine: needs NNUE_L1_MAX=1024 (bumped in nnue.h) to load this net.

RUN (no arguments):
    python c500_train_v4.py

Files required on /tmp/code:
    luminex_nnue_train_v3.py   (the float trainer — this wraps it)
    quantize_int8.py           (unchanged int8 converter)
    feat_w.npy feat_b.npy feat_s.npy feat_t.npy  (existing 280M cache)
"""
import os, sys, subprocess
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
sys.path.insert(0, "/tmp/code")
os.environ["NNUE_NPY_DIR"] = "/tmp/code"

from c2net.context import prepare, upload_output
ctx = prepare()
out_dir = ctx.output_path

import torch
print(f"torch {torch.__version__}, cuda: {torch.cuda.is_available()}", flush=True)

if not os.path.exists("/tmp/code/feat_w.npy"):
    print("ERROR: no featurized cache at /tmp/code — run c500_featurize.py first.", flush=True)
    sys.exit(1)

import luminex_nnue_train_v3 as T

# ---- v4-raw defaults (float, L1=1024; env-overridable; no CLI args) ----
L1      = int(os.environ.get("NNUE_L1", "1024"))   # 2x v2's capacity -> raw-strength lever
L2      = int(os.environ.get("NNUE_L2", "16"))     # v2's size (avoid overfit on 280M)
L3      = int(os.environ.get("NNUE_L3", "32"))     # v2's size
EPOCHS  = int(os.environ.get("NNUE_EPOCHS", "20"))
BS      = int(os.environ.get("NNUE_BS", "32768"))
LR      = float(os.environ.get("NNUE_LR", "1.2e-3"))
BUDGET  = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.7 * 3600))))
QATWARM = float(os.environ.get("NNUE_QAT_WARMUP", "0.25"))
out_file = os.path.join(out_dir, "luminex_v4_raw.nnue")
i8_file  = os.path.join(out_dir, "luminex_v4_raw_i8.nnue")

print(f"\nTraining v4-raw (FLOAT L1=1024) from /tmp/code: L1={L1} L2={L2} L3={L3} "
      f"QAT-warmup={QATWARM} epochs={EPOCHS} bs={BS} lr={LR} bf16=ON budget={BUDGET}s "
      f"(~{BUDGET/3600:.1f}h)", flush=True)
print("v4-raw = 2x L1 capacity (1024), float FT (no precision loss). Raw-strength push.", flush=True)

T.train(
    data=None, out=out_file,                # data=None: .npy VRAM cache
    epochs=EPOCHS, batch_size=BS, lr=LR, device="cuda",
    L1=L1, L2=L2, L3=L3, swa=True, qat_warmup_frac=QATWARM,
    time_budget_sec=BUDGET,
    ckpt_path="/tmp/code/v4_raw_ckpt.pt",
    use_amp=True,
)

print(f"\nQuantizing {out_file} -> int8...", flush=True)
subprocess.run([sys.executable, "/tmp/code/quantize_int8.py", out_file, i8_file], check=True)

print(f"\nUploading nets from {out_dir}...", flush=True)
upload_output()
print("DONE! — luminex_v4_raw_i8.nnue (L1=1024, float-trained). Load on the NNUE_L1_MAX=1024 engine.", flush=True)
