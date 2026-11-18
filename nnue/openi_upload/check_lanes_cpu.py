"""check_fused.py's relation-lane case (lanes_check) on N positions of any LCR1 file, on the CPU through Triton's
interpreter or on a GPU: the fused kernels with extra lane sums (ft_fused.py: _ft_fwd and _act_bwd with T > 0)
against the unfused join's ops: h, the G into the table's kernel and the lane sums' gradient bit for bit (T = 128 and
256 in float, 256 under QAT). On the CPU the interpreter computes each kernel's dots in numpy, so it checks what the
kernels add and in which order, not the GPU's tensor-core arithmetic; on a GPU it checks both. Exit 0 = pass.
usage: python check_lanes_cpu.py RECORDS.lcr [N] [cpu|cuda]   (defaults 256, cpu)"""
import os
import sys

n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
if device == "cpu":
    os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["NNUE_DEVICE"] = device
os.environ.update(NNUE_BUCKETS="8", NNUE_L1="768", NNUE_BS=str(n), NNUE_LOSS_POWER="3.0",
                  NNUE_QUIET_LOSS="1,0.3,0", NNUE_TARGET_CLIP="1500", NNUE_COMPILE="0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import check_fused as CF  # noqa: E402

recs = np.fromfile(sys.argv[1], dtype=np.uint8, count=n * 32).reshape(-1, 32)
if recs.shape[0] < n:
    sys.exit(f"{sys.argv[1]} holds {recs.shape[0]} records, not {n}")
CF.RT_BATCH = lambda RT: torch.from_numpy(recs.copy()).to(device)
if os.environ.get("TRITON_INTERPRET") == "1":
    # The interpreter rounds fp32 -> bf16 by truncation (3.8.0: 4,096 of 4,096 values), not to nearest even: the
    # reference rounds the same way where the kernels round (h, and the G the table's kernel reads).
    CF.TO_BF16 = lambda t: (t.float().view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
ok = True
for qat, T in ((False, 128), (False, 256), (True, 256)):
    ok = CF.lanes_check(None, "", qat, T, device=device) and ok
print("LANES_PASS" if ok else "LANES_FAIL", flush=True)
sys.exit(0 if ok else 1)
