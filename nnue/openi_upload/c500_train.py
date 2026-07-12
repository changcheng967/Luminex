#!/usr/bin/env python3
"""C500: train v2 NNUE from the featurized cache on /tmp/code — ONE session.

Reads the .npy straight from /tmp/code (persistent JuiceFS, like v1 did). A time-budget
(~3.7h) guarantees it never exceeds the 4h session: it trains as many epochs as fit and
saves the net at the end either way. bf16 (NNUE_AMP=1) + bigger batch optional for speed.
"""
import os, sys
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

import luminex_nnue_train as T

L1     = int(os.environ.get("NNUE_L1", "512"))
EPOCHS = int(os.environ.get("NNUE_EPOCHS", "15"))
BS     = int(os.environ.get("NNUE_BS", "32768"))
BUDGET = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.7 * 3600))))   # stop before 4h session limit
AMP    = os.environ.get("NNUE_AMP", "0") == "1"
out_file = os.path.join(out_dir, "luminex_v2.nnue")

print(f"\nTraining v2 from /tmp/code: L1={L1} epochs={EPOCHS} bs={BS} amp={AMP} "
      f"budget={BUDGET}s (~{BUDGET/3600:.1f}h, one session)", flush=True)

T.train(
    data=None, out=out_file,
    epochs=EPOCHS, batch_size=BS, lr=1e-3, device="cuda",
    L1=L1, lr_schedule=True, swa=True,
    time_budget_sec=BUDGET,
    ckpt_path="/tmp/code/v2_ckpt.pt",   # crash-safety only (saved each epoch)
    use_amp=AMP,
)

print(f"\nUploading {out_file}...", flush=True)
upload_output()
print("DONE!", flush=True)
