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
usage: python check_fused.py --qat DATA_DIR [CKPT.pt]
    The same batch under NNUE_QAT=1: first the feature layer alone, the fused kernels' exact
    forward (each rounded weight as two bf16 parts) against the dense path's fp32 bmm, which
    must agree bit for bit; then the loss and gradients of FT_DENSE=3 against =2, as above.
usage: python check_fused.py --act DATA_DIR [CKPT.pt]
    NNUE_FT_ACT=1 (bias and SCReLU in the kernels) against =0 on the same batch, in float and
    under NNUE_QAT=1, eager. Bit for bit: the activations h against forward()'s ops (rounded to
    bf16 as autocast rounds them; fp32 under QAT), and for a given dL/dh, from one consumer and
    from two (the draw head's case), the gradient G that goes into the table's kernel. The
    table's and the bias's gradients within 1e-5 (relative norm: the table's kernel adds with
    atomics, the bias sums in another order); then fwd_loss end to end: the loss bit for bit
    and every gradient within 1e-5. The other modes run with NNUE_FT_ACT as set (default 1),
    except --qat's feature-layer check, which needs the sums (=0).
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


def one_batch(data, qat):
    """131,072 positions of the first t91 part, the environment of the checks."""
    import numpy as np
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.update(NNUE_BUCKETS="8", NNUE_L1="768", NNUE_BS="131072", NNUE_LOSS_POWER="3.0",
                      NNUE_QUIET_LOSS="1,0.3,0", NNUE_TARGET_CLIP="1500", NNUE_QAT="1" if qat else "0",
                      NNUE_COMPILE="0")
    part = sorted(glob.glob(data + "/t91_*.lcr"))[0]
    rec = torch.from_numpy(np.fromfile(part, dtype=np.uint8, count=2_000_000 * 32).reshape(-1, 32)).to("cuda")
    g = torch.Generator(device="cuda").manual_seed(3)
    return rec[torch.randint(0, rec.shape[0], (131072,), device="cuda", generator=g)]


def load_model(RT, ckpt):
    import torch
    torch.manual_seed(1)
    model = RT.LNNUE(L1=768).cuda()
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"], strict=False)
        with torch.no_grad():
            model.ft.weight[RT.NUM_INPUTS].zero_()
    return model


def gradients(data, ckpt, fp8=0, qat=False):
    import torch
    batch = one_batch(data, qat)
    act = os.environ.get("NNUE_FT_ACT", "1")
    out = {}
    exact = True
    if qat:   # the feature layer alone: fused (split bf16) against the dense fp32 bmm, bit for bit
        os.environ["NNUE_FT_DENSE"], os.environ["NNUE_FT_FP8"], os.environ["NNUE_FT_ACT"] = "3", "0", "0"
        import resident_train
        RT = importlib.reload(resident_train)
        torch.manual_seed(1)
        model = RT.LNNUE(L1=768).cuda()
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"], strict=False)
        with torch.no_grad():
            wf, bf = RT.expand(batch)[:2]
            wq = RT.qat_ft(model.ft.weight)
            hi, lo = RT.FF.split_bf16(wq[:RT.NUM_INPUTS].float())
            split_ok = torch.equal(hi.float() + lo.float(), wq[:RT.NUM_INPUTS].float())
            fw, fb = RT.ft_fused(wf, bf, model.ft.weight)
            dw, db = RT.ft_dense(wf, wq), RT.ft_dense(bf, wq)
            same = torch.equal(fw, dw) and torch.equal(fb, db)
            worst = max(float((fw - dw).abs().max()), float((fb - db).abs().max()))
        exact = split_ok and same and fw.dtype == torch.float32
        print(f"  QAT feature layer: hi + lo == the rounded table: {split_ok}; fused == dense fp32 bit for bit: "
              f"{same} (largest difference {worst:.3e}, fused dtype {fw.dtype}; "
              f"|table| <= {float(wq.abs().max()):.4f})", flush=True)
        del model
        torch.cuda.empty_cache()
    modes = ((("2", "0"), ("3", "0")) if not fp8 else (("3", "0"), ("3", str(fp8))))
    for mode, f8 in modes:
        os.environ["NNUE_FT_DENSE"], os.environ["NNUE_FT_FP8"], os.environ["NNUE_FT_ACT"] = mode, f8, act
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
    if qat:
        a, b = a + " QAT", b + " QAT"
    dl = abs(l3 - l2) / max(abs(l2), 1e-30)
    print(f"  loss: {a} {l2:.6f}, {b} {l3:.6f} (relative difference {dl:.1e})", flush=True)
    worst, worst_cos, ok = 0.0, 1.0, exact and dl < (1e-2 if fp8 else 2e-3) and set(g2) == set(g3)
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


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def act_check(data, ckpt, qat):
    """--act for one mode (float, or QAT): True if everything matches (module docstring)."""
    import torch
    batch = one_batch(data, qat)
    tag = "QAT" if qat else "float"
    os.environ["NNUE_FT_DENSE"], os.environ["NNUE_FT_FP8"], os.environ["NNUE_FT_ACT"] = "3", "0", "1"
    import resident_train
    RT = importlib.reload(resident_train)
    model = load_model(RT, ckpt)
    wf, bf, stm = RT.expand(batch)[:3]
    n, L1 = wf.shape[0], model.ft.weight.shape[1]
    seen = []   # the G each path hands the table's kernel, in sorted order
    real = RT.ft_weight_grad

    def spy(planes_s, rb, g, nrows):
        seen.append(g.to(torch.bfloat16).clone())
        return real(planes_s, rb, g, nrows)

    RT.ft_weight_grad = spy

    def leaves():
        w = model.ft.weight.detach().clone().requires_grad_()
        b = model.ft_bias.detach().clone().requires_grad_()
        return w, b, (RT.qat_ft(w) if qat else w), (RT.qat_bias(b) if qat else b)

    def before(W, B):   # NNUE_FT_ACT=0: FusedFT's sums in side-row order, then forward()'s ops
        aw, ab = RT.FusedFT.apply(wf, bf, W)
        xw, xb = (aw + B, ab + B) if qat else (aw.float() + B, ab.float() + B)
        s = stm.view(-1, 1) > 0.5
        h = torch.cat([torch.where(s, xw, xb), torch.where(s, xb, xw)], 1)
        return torch.clamp(h, 0.0, 1.0).square()

    ok = True
    with torch.no_grad():
        W, B = leaves()[2:]
        ref = before(W, B)
        ref = ref if qat else ref.to(torch.bfloat16)   # what the next matmul reads (autocast's cast)
        hv, h2 = RT.FusedAct.apply(wf, bf, W, B, stm)
        same = torch.equal(hv, ref) and hv.dtype == ref.dtype and hv.data_ptr() == h2.data_ptr()
        x = torch.cat(RT.FusedFT.apply(wf, bf, W)).float() + B
        inside = float(((x > 0) & (x < 1)).float().mean())
        print(f"  {tag}: h == forward()'s ops bit for bit: {same} ({hv.dtype}, largest difference "
              f"{float((hv.float() - ref.float()).abs().max()):.3e}; {inside:.1%} of the sums inside (0, 1), "
              f"{float((x == 0).float().mean()) + float((x == 1).float().mean()):.2e} exactly 0 or 1)", flush=True)
        ok = ok and same
        del x
    gen = torch.Generator(device="cuda").manual_seed(5)
    gdt = torch.float32 if qat else torch.bfloat16
    g1 = (torch.randn(n, 2 * L1, device="cuda", generator=gen) * 1e-3).to(gdt)
    g2 = (torch.randn(n, 2 * L1, device="cuda", generator=gen) * 1e-4).to(gdt)
    for two in (False, True):
        seen.clear()
        res = []
        for new in (False, True):
            w, b, W, B = leaves()
            if new:
                hv, h2 = RT.FusedAct.apply(wf, bf, W, B, stm)
                torch.autograd.backward([hv, h2] if two else [hv], [g1, g2] if two else [g1])
            else:
                h = before(W, B)
                if qat:   # two fp32 consumers: autograd adds their gradients in fp32
                    torch.autograd.backward([h], [g1 + g2 if two else g1])
                else:     # two consumers, each through its own autocast cast
                    outs = [h.to(torch.bfloat16)] + ([h.to(torch.bfloat16)] if two else [])
                    torch.autograd.backward(outs, [g1, g2] if two else [g1])
            res.append((w.grad, b.grad))
        g_same = len(seen) == 2 and torch.equal(seen[0], seen[1])
        ew, eb = rel(res[1][0], res[0][0]), rel(res[1][1], res[0][1])
        what = "two consumers (g1 + g2)" if two else "one consumer"
        print(f"  {tag}, {what}: G into the table's kernel bit for bit: {g_same} "
              f"({0 if not g_same else int(seen[0].ne(0).sum()):,} nonzero); table gradient {ew:.1e}, "
              f"bias gradient {eb:.1e}", flush=True)
        ok = ok and g_same and ew < 1e-5 and eb < 1e-5
    RT.ft_weight_grad = real
    out = []
    for act in ("0", "1"):   # fwd_loss end to end
        RT.FT_ACT = act == "1"
        model.zero_grad(set_to_none=True)
        loss = RT.fwd_loss(model, batch)
        loss.backward()
        out.append((loss.detach().clone(), {k: p.grad.detach().float().clone() for k, p in model.named_parameters()
                                            if p.grad is not None}))
    (l0, gr0), (l1, gr1) = out
    worst = max(rel(gr1[k], gr0[k]) for k in gr0) if set(gr0) == set(gr1) else float("inf")
    exact_tail = [k for k in gr0 if not k.startswith("ft") and k in gr1 and torch.equal(gr0[k], gr1[k])]
    print(f"  {tag}, fwd_loss: loss {float(l0):.8f} vs {float(l1):.8f}, bit for bit: {torch.equal(l0, l1)}; "
          f"worst gradient {worst:.1e}; the tail's gradients bit for bit: {len(exact_tail)} of "
          f"{sum(1 for k in gr0 if not k.startswith('ft'))}; table {rel(gr1['ft.weight'], gr0['ft.weight']):.1e}, "
          f"bias {rel(gr1['ft_bias'], gr0['ft_bias']):.1e}", flush=True)
    ok = ok and torch.equal(l0, l1) and worst < 1e-5
    del model
    torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    args, fp8, qat = sys.argv[1:], 0, False
    if args[0] == "--act":
        try:
            ok = all([act_check(args[1], args[2] if len(args) > 2 else "", q) for q in (False, True)])
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"ACT_FAIL: {ex.__class__.__name__}: {ex}", flush=True)
            sys.exit(1)
        print("ACT_PASS" if ok else "ACT_FAIL", flush=True)
        sys.exit(0 if ok else 1)
    if args[0] == "--fp8":
        fp8, args = int(args[1]), args[2:]
    elif args[0] == "--qat":
        qat, args = True, args[1:]
    try:
        sys.exit(0 if gradients(args[0], args[1] if len(args) > 1 else "", fp8, qat) else 1)
    except Exception as ex:   # any failure here only means the run trains on the dense path
        print(f"GRAD_FAIL: {ex.__class__.__name__}: {ex}", flush=True)
        sys.exit(1)
