"""Widening test for a bucketed float net (LNB1): NNUE_L2 16 -> 32 must widen every output
bucket's tail, keep the old units' weights exactly, and leave the output within the noise
v2_expand adds (0.01 on the new rows and columns), like test_v2_expand.py's J32 arm.
Run on a machine with torch:  python test_bucket_widen.py <float LNB1 .nnue>
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luminex_nnue_train as L

sd = L.LNNUE.load_lnn1_state(sys.argv[1])
nb = sum(1 for k in sd if k.startswith("bl2.") and k.endswith(".weight"))
old_j = sd["bl2.0.weight"].shape[0]
print(f"loaded {sys.argv[1]}: {nb} buckets, L2 {old_j}, l2 {tuple(sd['bl2.0.weight'].shape)}")
assert nb > 1, "this test needs a bucketed (LNB1) net"

torch.manual_seed(7)
B, F = 512, 32
w_idx = torch.randint(0, L.NUM_INPUTS, (B, F))
b_idx = torch.randint(0, L.NUM_INPUTS, (B, F))
stm = torch.randint(0, 2, (B,)).float()


def build(l2):
    os.environ.update({"NNUE_PAIRWISE": "0", "NNUE_BUCKETS": str(nb), "NNUE_L2": str(l2)})
    m = L.LNNUE(L1=768)
    L.LNNUE.v2_expand(m, sd)
    m.eval()
    return m


with torch.no_grad():
    base = build(old_j)
    ref = base(w_idx, b_idx, stm)
    wide = build(2 * old_j)
    for b in range(nb):
        w2, w3 = wide.bl2[b].weight, wide.bl3[b].weight
        assert tuple(w2.shape) == (2 * old_j, sd[f"bl2.{b}.weight"].shape[1]), (b, tuple(w2.shape))
        assert tuple(w3.shape) == (sd[f"bl3.{b}.weight"].shape[0], 2 * old_j), (b, tuple(w3.shape))
        assert torch.equal(w2[:old_j], sd[f"bl2.{b}.weight"]), f"bucket {b}: old L2 rows changed"
        assert torch.equal(w3[:, :old_j], sd[f"bl3.{b}.weight"]), f"bucket {b}: old L3 columns changed"
        assert w2[old_j:].abs().max() > 0 and w3[:, old_j:].abs().max() > 0, f"bucket {b}: new units are all zero"
    d = (wide(w_idx, b_idx, stm) - ref).abs().max().item()
    print(f"  {nb} bucket tails widened {old_j} -> {2 * old_j}, old units unchanged; output max|diff| = {d:.4f} cp")
    assert d <= 15.0, f"widening moved the output by {d} cp"
print("BUCKET_WIDEN_PASS")
