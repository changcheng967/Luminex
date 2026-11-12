"""Checks NNUE_FT_DENSE=3 (ft_fused.py) against the dense path NNUE_FT_DENSE=2. Exit 0 = pass.

usage: python check_fused.py DATA_DIR [CKPT.pt]
    One batch of 131,072 positions from the first t91 part, through the checkpoint (or random
    weights), eager, under FT_DENSE=2 and =3: the loss and the gradient of every parameter.
    Passes when the losses agree within 2e-3 and every gradient within 2e-2 (relative norm).
    The fused path rounds the weight gradient's input to bf16 where the dense path uses TF32.
usage: python check_fused.py --fp8 MODE DATA_DIR [CKPT.pt]
    The same batch through NNUE_FT_DENSE=3 in bf16 and with NNUE_FT_FP8=MODE (1 = FP8 weight
    gradient, 2 = FP8 forward too): the loss, and each gradient's relative error and cosine
    similarity. Passes when the losses agree within 1e-2 and every cosine is above 0.99; the
    paired runs (--compare) decide whether training is the same.
usage: python check_fused.py --compare ARM2.log ARM3.log
    Two paired short runs (same seed, data and schedule; FT_DENSE 2 vs 3). Passes when every
    [HEALTH] and [DISJOINT] loss they share (after step 0) agrees within 1% and the fused run
    finished."""
import glob
import importlib
import os
import re
import sys


def compare(a, b):
    def parse(p):
        h, d, done, sp = {}, {}, False, None
        for line in open(p, errors="replace"):
            m = re.match(r"\[HEALTH step (\d+)\] val loss=([0-9.]+)", line)
            if m:
                h[int(m.group(1))] = float(m.group(2))
            m = re.match(r"\[DISJOINT step (\d+)\] loss=([0-9.]+)", line)
            if m:
                d[int(m.group(1))] = float(m.group(2))
            m = re.search(r"DONE .*\(([0-9.]+) steps/s\)", line)
            if m:
                done, sp = True, float(m.group(1))
        return h, d, done, sp
    (h2, d2, _, s2), (h3, d3, ok3, s3) = parse(a), parse(b)
    worst, n = 0.0, 0
    for x2, x3, name in ((h2, h3, "held-out"), (d2, d3, "January")):
        for s in sorted(set(x2) & set(x3)):
            if s == 0:
                continue
            r = (x3[s] - x2[s]) / x2[s]
            worst, n = max(worst, abs(r)), n + 1
            print(f"  {name} step {s}: dense {x2[s]:.6f}, fused {x3[s]:.6f} ({r:+.2%})", flush=True)
    print(f"  speed: dense {s2} steps/s, fused {s3} steps/s", flush=True)
    ok = ok3 and n >= 2 and worst < 0.01
    print(("FUSED_PASS" if ok else "FUSED_FAIL") + f": worst difference {worst:.2%} over {n} readings; "
          f"fused run finished: {ok3}", flush=True)
    return ok


def gradients(data, ckpt, fp8=0):
    import numpy as np
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.update(NNUE_BUCKETS="8", NNUE_L1="768", NNUE_BS="131072", NNUE_LOSS_POWER="3.0",
                      NNUE_QUIET_LOSS="1,0.3,0", NNUE_TARGET_CLIP="1500", NNUE_QAT="0", NNUE_COMPILE="0")
    part = sorted(glob.glob(data + "/t91_*.lcr"))[0]
    rec = torch.from_numpy(np.fromfile(part, dtype=np.uint8, count=2_000_000 * 32).reshape(-1, 32)).to("cuda")
    g = torch.Generator(device="cuda").manual_seed(3)
    batch = rec[torch.randint(0, rec.shape[0], (131072,), device="cuda", generator=g)]
    out = {}
    modes = ((("2", "0"), ("3", "0")) if not fp8 else (("3", "0"), ("3", str(fp8))))
    for mode, f8 in modes:
        os.environ["NNUE_FT_DENSE"], os.environ["NNUE_FT_FP8"] = mode, f8
        import resident_train
        RT = importlib.reload(resident_train)
        torch.manual_seed(1)
        model = RT.LNNUE(L1=768).cuda()
        if ckpt:
            ck = torch.load(ckpt, map_location="cuda", weights_only=False)
            model.load_state_dict(ck["model"], strict=False)
            with torch.no_grad():
                model.ft.weight[RT.NUM_INPUTS].zero_()
        loss = RT.fwd_loss(model, batch)
        loss.backward()
        out[(mode, f8)] = (float(loss), {k: p.grad.detach().float().clone() for k, p in model.named_parameters()
                                         if p.grad is not None})
        del model
        torch.cuda.empty_cache()
    (l2, g2), (l3, g3) = out[modes[0]], out[modes[1]]
    a, b = ("dense", "fused") if not fp8 else ("fused bf16", f"fused FP8 mode {fp8}")
    dl = abs(l3 - l2) / max(abs(l2), 1e-30)
    print(f"  loss: {a} {l2:.6f}, {b} {l3:.6f} (relative difference {dl:.1e})", flush=True)
    worst, worst_cos, ok = 0.0, 1.0, dl < (1e-2 if fp8 else 2e-3) and set(g2) == set(g3)
    for k in sorted(g2):
        if k not in g3:
            print(f"  {k}: no gradient under FT_DENSE=3", flush=True)
            continue
        e = float((g3[k] - g2[k]).norm() / g2[k].norm().clamp_min(1e-30))
        c = float((g3[k] * g2[k]).sum() / (g3[k].norm() * g2[k].norm()).clamp_min(1e-30))
        worst, worst_cos = max(worst, e), min(worst_cos, c)
        if k.startswith("ft") or e > 2e-3:
            print(f"  {k}: gradient relative error {e:.1e}, cosine {c:.6f}", flush=True)
    ok = ok and (worst_cos > 0.99 if fp8 else worst < 2e-2)
    print(("GRAD_PASS" if ok else "GRAD_FAIL") + f": loss {dl:.1e}, worst gradient {worst:.1e}, "
          f"lowest cosine {worst_cos:.6f}", flush=True)
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    args, fp8 = sys.argv[1:], 0
    if args[0] == "--fp8":
        fp8, args = int(args[1]), args[2:]
    try:
        sys.exit(0 if gradients(args[0], args[1] if len(args) > 1 else "", fp8) else 1)
    except Exception as ex:   # any failure here only means the run trains on the dense path
        print(f"GRAD_FAIL: {ex.__class__.__name__}: {ex}", flush=True)
        sys.exit(1)
