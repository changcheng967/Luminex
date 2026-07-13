#!/usr/bin/env python3
"""C500: train v4 = v2's architecture (L1=512) with VALIDATION-loss tracking.
ONE session, NO command-line arguments.

DISCIPLINE: do NOT increase L1 (capacity) or data until the train-vs-val loss curves prove
which is the bottleneck. This run gives you those curves:
  - val >> train & rising  -> OVERFITTING  -> data is the limit  -> then collect more positions
  - train plateaus high    -> UNDERFITTING -> capacity is the limit -> then try L1=1024
  - both converge & close  -> at the sweet spot -> better training (more epochs/LR tuning)

Same arch as v2 (L1=512 L2=16 L3=32), FLOAT (learns like v2/v3), QAT L2/L3/out, SWA, but
MORE EPOCHS (30) + per-epoch val loss so you can read the bottleneck off the log. The goal
is to first see if v2's arch is even converged on 280M — if not, more epochs alone may
surpass v2; if it plateaus, the curves say what to add next.

Engine: NNUE_L1_MAX=1024 (bumped) loads this L1=512 net fine (and any future L1=1024).

RUN (no arguments):
    python c500_train_v4.py
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

# ---- v4 = v2's arch (L1=512), val-loss-tracked, more epochs. Env-overridable; no CLI args. ----
L1      = int(os.environ.get("NNUE_L1", "512"))   # v2's size — NOT increased (await curve evidence)
L2      = int(os.environ.get("NNUE_L2", "16"))
L3      = int(os.environ.get("NNUE_L3", "32"))
EPOCHS  = int(os.environ.get("NNUE_EPOCHS", "30"))   # more than v2 -> see full convergence
BS      = int(os.environ.get("NNUE_BS", "32768"))
LR      = float(os.environ.get("NNUE_LR", "1.2e-3"))
BUDGET  = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.7 * 3600))))
QATWARM = float(os.environ.get("NNUE_QAT_WARMUP", "0.25"))
out_file = os.path.join(out_dir, "luminex_v4.nnue")
i8_file  = os.path.join(out_dir, "luminex_v4_i8.nnue")

print(f"\nTraining v4 (v2 arch L1=512, val-tracked) from /tmp/code: L1={L1} L2={L2} L3={L3} "
      f"epochs={EPOCHS} bs={BS} lr={LR} bf16=ON budget={BUDGET}s (~{BUDGET/3600:.1f}h)", flush=True)
print("v4 = v2 arch + val-loss curves. Read the log: val>>train => need data; train flat-high => need L1.", flush=True)

T.train(
    data=None, out=out_file,
    epochs=EPOCHS, batch_size=BS, lr=LR, device="cuda",
    L1=L1, L2=L2, L3=L3, swa=True, qat_warmup_frac=QATWARM,
    time_budget_sec=BUDGET,
    ckpt_path="/tmp/code/v4_ckpt.pt",
    use_amp=True,
)

print(f"\nQuantizing {out_file} -> int8...", flush=True)
subprocess.run([sys.executable, "/tmp/code/quantize_int8.py", out_file, i8_file], check=True)

print(f"\nUploading nets from {out_dir}...", flush=True)
upload_output()
print("DONE! — luminex_v4_i8.nnue (L1=512). READ the VAL loss lines to decide next step.", flush=True)
