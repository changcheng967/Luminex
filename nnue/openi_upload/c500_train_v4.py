#!/usr/bin/env python3
"""C500: train v4 NNUE (L1=512 + QAT + L3=64) from the featurized cache on /tmp/code.
ONE session, NO command-line arguments.

v4 = "v2's capacity done right":
  - L1=512  (KEEP v2's FT capacity — do NOT shrink, unlike v3's L1=256 which only matched v2)
  - L2=16   (v2's value -> identical L2-dot cost -> same NPS as v2, NOT slower)
  - L3=64   (2x v2's L3, nearly free: L3-dot = 64x16 = tiny)
  - QAT     (train under simulated int8/int16 quantization -> recovers the precision v2 lost
             to post-hoc quantization. This is the real strength gain over v2.)
Expected: ~+15-25 Elo over v2 at the SAME speed (~990K NPS). Runnable on the EXISTING engine
(zero C++ changes — L1/L2/L3 read from the net header, dot_i8_vnni_small handles L3=64).

This is the twin of c500_train_v3.py with v4 (L1=512) defaults. Same .npy VRAM loader,
same c2net, same time-budget guard, in-place quantize + upload.

RUN (no arguments):
    python c500_train_v4.py

Files required on /tmp/code (upload alongside this script):
    luminex_nnue_train_v3.py   (the trainer — reused as-is)
    quantize_int8.py           (the int8 converter, unchanged)
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

# ---- v4 defaults (env-overridable; no CLI args) ----
# L1=512 keeps v2's capacity; L2=16 keeps v2's L2-dot cost (same NPS); L3=64 is cheap capacity.
L1      = int(os.environ.get("NNUE_L1", "512"))
L2      = int(os.environ.get("NNUE_L2", "16"))
L3      = int(os.environ.get("NNUE_L3", "64"))
EPOCHS  = int(os.environ.get("NNUE_EPOCHS", "20"))   # L1=512 is ~2x slower/epoch than v3's 256
BS      = int(os.environ.get("NNUE_BS", "32768"))
LR      = float(os.environ.get("NNUE_LR", "1.2e-3"))
BUDGET  = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.7 * 3600))))
QATWARM = float(os.environ.get("NNUE_QAT_WARMUP", "0.25"))
out_file = os.path.join(out_dir, "luminex_v4.nnue")
i8_file  = os.path.join(out_dir, "luminex_v4_i8.nnue")

print(f"\nTraining v4 from /tmp/code: L1={L1} L2={L2} L3={L3} QAT-warmup={QATWARM} "
      f"epochs={EPOCHS} bs={BS} lr={LR} bf16=ON budget={BUDGET}s (~{BUDGET/3600:.1f}h)", flush=True)
print("v4 = v2 capacity (L1=512) + QAT precision + L3=64. Same NPS as v2, ~+20 Elo.", flush=True)

T.train(
    data=None, out=out_file,                # data=None: uses the .npy VRAM cache, not a text file
    epochs=EPOCHS, batch_size=BS, lr=LR, device="cuda",
    L1=L1, L2=L2, L3=L3, swa=True, qat_warmup_frac=QATWARM,
    time_budget_sec=BUDGET,
    ckpt_path="/tmp/code/v4_ckpt.pt",
    use_amp=True,                           # bf16 matmuls (QAT fake-quant stays fp32)
)

print(f"\nQuantizing {out_file} -> int8...", flush=True)
subprocess.run([sys.executable, "/tmp/code/quantize_int8.py", out_file, i8_file], check=True)

print(f"\nUploading nets from {out_dir}...", flush=True)
upload_output()
print("DONE! — luminex_v4_i8.nnue is the net to load on hyperai.", flush=True)
