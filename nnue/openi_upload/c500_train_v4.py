#!/usr/bin/env python3
"""C500: train v4 NNUE (int16 ACCUMULATOR + int8 FT, L1=512) — the SF architecture.
ONE session, NO command-line arguments. This is the speed-architecture retrain.

v4 = int8 FT weights + int16 accumulator + integer ClippedReLU(0,127) activation.
Breaks the float-accumulator floors: update 724->~380 cyc (pure int16 add), SCReLU
320->~160 cyc (integer, no float chain) -> ~1.35M NPS single-thread (+36% over v2).
+ working MT -> 2M+ aggregate -> the +280 equal-depth edge over HCE materializes at bullet.

NOTE: v4 needs the int16-accumulator ENGINE PATH (LNI9 loader) — NOT yet built. Train the
net now (this script); the engine work is separate. The net saves as LNI9 (int8 FT + int8
L2/L3/out) ready for the v4 engine loader.

RUN (no arguments):
    python c500_train_v4.py

Files required on /tmp/code:
    luminex_nnue_train_v4.py   (the int16-acc trainer — this wraps it)
    feat_w.npy feat_b.npy feat_s.npy feat_t.npy  (the existing 280M featurized cache)
"""
import os, sys
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

import luminex_nnue_train_v4 as T

# ---- v4 defaults (env-overridable; no CLI args) ----
L1      = int(os.environ.get("NNUE_L1", "512"))
L2      = int(os.environ.get("NNUE_L2", "16"))
L3      = int(os.environ.get("NNUE_L3", "32"))
EPOCHS  = int(os.environ.get("NNUE_EPOCHS", "20"))
BS      = int(os.environ.get("NNUE_BS", "32768"))
LR      = float(os.environ.get("NNUE_LR", "1.2e-3"))
BUDGET  = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.7 * 3600))))
QATWARM = float(os.environ.get("NNUE_QAT_WARMUP", "0.25"))
out_file = os.path.join(out_dir, "luminex_v4_int16.nnue")

print(f"\nTraining v4 (int16 ACCUMULATOR) from /tmp/code: L1={L1} L2={L2} L3={L3} "
      f"QAT-warmup={QATWARM} epochs={EPOCHS} bs={BS} lr={LR} bf16=ON budget={BUDGET}s "
      f"(~{BUDGET/3600:.1f}h)", flush=True)
print("v4 = int8 FT + int16 acc + scaled SCReLU. Breaks the float-acc update floor -> ~1.14M NPS.", flush=True)

T.train(
    data=None, out=out_file,
    epochs=EPOCHS, batch_size=BS, lr=LR, device="cuda",
    L1=L1, L2=L2, L3=L3, swa=True, qat_warmup_frac=QATWARM,
    time_budget_sec=BUDGET,
    ckpt_path="/tmp/code/v4_int16_ckpt.pt",
    use_amp=True,
)

print(f"\nUploading {out_file} (LNI9, int16-acc)...", flush=True)
upload_output()
print("DONE! — luminex_v4_int16.nnue is the int16-acc net (needs the v4 engine loader).", flush=True)
