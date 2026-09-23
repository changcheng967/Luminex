"""E5: Net2Net widening unit test — verify function preservation AND gradient wake-up.
Usage: python test_net2net.py (requires torch + luminex_nnue_train on path)"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from luminex_nnue_train import LNNUE, NUM_INPUTS

torch.manual_seed(42)
OLD, NEW, BS, MAXP = 512, 768, 256, 32

# Build a trained-looking 512 model (random weights + bias — enough for gradient test)
old_model = LNNUE(L1=OLD)
with torch.no_grad():
    old_model.ft.weight.normal_(0, 0.2)
    old_model.ft_bias.normal_(0, 0.1)
old_model.L1 = OLD
old_sd = {k: v.clone() for k, v in old_model.state_dict().items()}

# Build the 768 model and widen
new_model = LNNUE(L1=NEW)
new_model.L1 = NEW
LNNUE.net2net_widen(new_model, old_sd, OLD)

# Test 1: Function preservation — same features → same output (within tolerance)
w_idx = torch.randint(0, NUM_INPUTS, (BS, MAXP))
b_idx = torch.randint(0, NUM_INPUTS, (BS, MAXP))
stm = torch.randint(0, 2, (BS,)).float()

with torch.no_grad():
    old_out = old_model(w_idx, b_idx, stm)
    new_out = new_model(w_idx, b_idx, stm)
diff = (old_out - new_out).abs().max().item()
print(f"T1 function preservation: max|old-new| = {diff:.6f}")
assert diff < 1.0, f"FAIL: function not preserved (diff={diff})"
print(f"  PASS (tolerance 1.0cp)")

# Test 2: L2 new-column gradients (THE wake-up signal — activations are
# nonzero via EPS bias, so L2 cols get grads even though they're zero)
new_model.zero_grad()
out = new_model(w_idx, b_idx, stm)
loss = out.abs().mean()
loss.backward()

l2_grad = new_model.l2.weight.grad  # [L2, 2*768]
new_cols = torch.cat([l2_grad[:, OLD:NEW], l2_grad[:, NEW+OLD:]], dim=1)
l2_nonzero = (new_cols.abs() > 1e-12).sum().item()
print(f"T2 L2 new-column gradients: {l2_nonzero}/{new_cols.numel()} nonzero")
assert l2_nonzero > 0, "FAIL: L2 new columns dead — EPS bias not producing nonzero activations"
print(f"  PASS (L2 columns waking up)")

# Test 3: Full chain unlock — one optimizer step moves L2 off zero, then FT grads flow
opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
for step in range(3):
    opt.zero_grad()
    out = new_model(w_idx, b_idx, stm)
    loss = (out - torch.randn_like(out)).abs().mean()
    loss.backward()
    opt.step()
# Now check FT new-dim gradients
new_model.zero_grad()
out = new_model(w_idx, b_idx, stm)
loss = out.abs().mean()
loss.backward()
ft_grad = new_model.ft.weight.grad
new_dim_grad = ft_grad[:, OLD:]
nonzero = (new_dim_grad.abs() > 1e-12).sum().item()
total = new_dim_grad.numel()
print(f"T3 FT new-dim gradients after 3 steps: {nonzero}/{total} nonzero ({100*nonzero/total:.1f}%)")
assert nonzero > 0, "FAIL: FT new dims still dead after L2 moved — chain not unlocking"
print(f"  PASS (full chain unlocked)")

print("\n=== ALL NET2NET TESTS PASS ===")
