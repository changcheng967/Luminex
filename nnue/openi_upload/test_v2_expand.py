"""Function-preservation smoke test for the v2 warm-start expansion.
Three arms, each warm-started from the same float LNN1 net:
  1. buckets (NNUE_BUCKETS=4)  — must match baseline EXACTLY (copies)
  2. pairwise (NNUE_PAIRWISE=1) — must match within O(delta^2) drift
  3. pairwise + J32 (NNUE_L2=32) — small bounded drift (noise 0.01)
Run on a machine with torch:  python test_v2_expand.py <float.nnue>
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luminex_nnue_train as L

net_path = sys.argv[1]
sd = L.LNNUE.load_lnn1_state(net_path)
print(f"loaded {net_path}: ft {tuple(sd['ft.weight'].shape)}, l2 {tuple(sd['l2.weight'].shape)}")

torch.manual_seed(7)
B, F = 512, 32
w_idx = torch.randint(0, L.NUM_INPUTS, (B, F))
b_idx = torch.randint(0, L.NUM_INPUTS, (B, F))
stm = torch.randint(0, 2, (B,)).float()

def build(env):
    for k, v in env.items(): os.environ[k] = v
    m = L.LNNUE(L1=768)
    L.LNNUE.v2_expand(m, sd)
    m.eval()
    return m

with torch.no_grad():
    base = build({'NNUE_PAIRWISE': '0', 'NNUE_BUCKETS': '1', 'NNUE_L2': '16'})
    ref = base(w_idx, b_idx, stm)

    for name, env, tol in [
        ("buckets=4",  {'NNUE_PAIRWISE': '0', 'NNUE_BUCKETS': '4', 'NNUE_L2': '16'}, 1e-4),
        ("pairwise",   {'NNUE_PAIRWISE': '1', 'NNUE_BUCKETS': '1', 'NNUE_L2': '16'}, 2.0),
        ("pw+J32",     {'NNUE_PAIRWISE': '1', 'NNUE_BUCKETS': '1', 'NNUE_L2': '32'}, 15.0),
        ("J32 only",   {'NNUE_PAIRWISE': '0', 'NNUE_BUCKETS': '1', 'NNUE_L2': '32'}, 15.0),
    ]:
        m = build(env)
        out = m(w_idx, b_idx, stm)
        d = (out - ref).abs().max().item()
        status = "PASS" if d <= tol else "FAIL"
        print(f"  {name:12s} max|diff| = {d:.6f} cp  (tol {tol})  {status}")
        assert d <= tol, f"{name} broke function preservation: {d}"
print("ALL_V2_ARMS_PASS")
