def _fused_act_test():
    """Box test of PR #88, NNUE_FT_ACT: the feature layer's bias, SCReLU and side-to-move write in the Triton
    kernels. About 8 minutes, no training run: writes /marimo/fatest (delete it afterwards) and the log
    /marimo/fatest.log; /marimo/data and /marimo/out are only read. Refuses while a training run is active or the
    GPU holds more than 2 GB.
      1. check_fused.py --act on the first t91 part: h and the gradient into the table's kernel bit for bit
         against forward()'s elementwise ops (float and QAT, one consumer and two), then fwd_loss end to end.
      2. The float phase as the runs train it (L1 768, NNUE_FT_DENSE=3, the draw head, compiled), from the float
         checkpoint, with NNUE_FT_ACT=0 and then =1: the trainer's own profile, 100 steps after 300 of warm-up,
         on 150M positions all on the card.
    Each step runs in its own process group, killed at 15 minutes or when the cell stops, so a hung step can
    neither hold the cell nor keep the GPU from the next run.
    The last line is FUSED_ACT_OK or FUSED_ACT_NOT_OK, with both step times."""
    import glob, hashlib, os, re, signal, subprocess, sys, textwrap, threading, time, urllib.request

    # The cells' draw head (v5r2.py's PATCH_DRAW, verbatim), applied to this branch's trainer for the float runs.
    PATCH_DRAW = textwrap.dedent('''
        import sys
        src, dst = sys.argv[1], sys.argv[2]
        rt = open(src, encoding="utf-8").read()


        def patch(old, new, what):
            global rt
            if rt.count(old) != 1:
                sys.exit(f"draw patch: {what} appears {rt.count(old)} times")
            rt = rt.replace(old, new)


        patch("def forward(model, wf, bf, stm, cnt, acc=None):",
              "def forward(model, wf, bf, stm, cnt, acc=None, want_h=False):", "forward()")
        patch("    return F.linear(tail_act(z), Wo, bo).gather(1, bucket).view(n).float() * 300.0\\n",
              "    _o = F.linear(tail_act(z), Wo, bo).gather(1, bucket).view(n).float() * 300.0\\n"
              "    return (_o, h) if want_h else _o\\n", "the value's return")
        patch('def fwd_loss(model, rec):\\n'
              '    wf, bf, stm, tgt, flag, cnt = expand(rec)\\n'
              '    acc = ft_fused(wf, bf, model.ft.weight) if FT_DENSE == 3 else None\\n'
              '    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\\n'
              '        pred = forward(model, wf, bf, stm, cnt, acc)\\n'
              '    return batch_loss(pred.float(), tgt.clamp(-TARGET_CLIP, TARGET_CLIP), flag)\\n',
              'DRAW_W = float(os.environ.get("NNUE_DRAW_W", "0"))\\n'
              '\\n'
              '\\n'
              'def draw_logit(model, h, pred, cnt):\\n'
              '    """The draw head: one logit per output bucket from the activations and the value\\'s size\\n'
              '    (detached: the draw loss shapes the accumulator, not the value\\'s output directly)."""\\n'
              '    n = h.shape[0]\\n'
              '    bucket = ((cnt - 1) * model.num_buckets // 32).clamp(0, model.num_buckets - 1).view(n, 1)\\n'
              '    z = model.dhead(h).float().gather(1, bucket).view(n)\\n'
              '    return z + model.dhead_p[bucket.view(n)] * (pred.detach().float().abs() / 400.0)\\n'
              '\\n'
              '\\n'
              'def fwd_loss(model, rec):\\n'
              '    wf, bf, stm, tgt, flag, cnt = expand(rec)\\n'
              '    acc = ft_fused(wf, bf, model.ft.weight) if FT_DENSE == 3 else None\\n'
              '    dh = getattr(model, "dhead", None)\\n'
              '    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\\n'
              '        if dh is None:\\n'
              '            pred = forward(model, wf, bf, stm, cnt, acc)\\n'
              '        else:\\n'
              '            pred, h = forward(model, wf, bf, stm, cnt, acc, want_h=True)\\n'
              '            dlogit = draw_logit(model, h, pred, cnt)\\n'
              '    vloss = batch_loss(pred.float(), tgt.clamp(-TARGET_CLIP, TARGET_CLIP), flag)\\n'
              '    if dh is None:\\n'
              '        return vloss\\n'
              '    fl = rec[:, 26].to(torch.int64)\\n'
              '    known = ((fl >> 3) & 1).float()\\n'
              '    drawn = (((fl >> 4) & 3) == 1).float()\\n'
              '    bce = F.binary_cross_entropy_with_logits(dlogit, drawn, reduction="none")\\n'
              '    return vloss + DRAW_W * (bce * known).sum() / known.sum().clamp(min=1.0)\\n', "fwd_loss()")
        patch("    model = LNNUE(L1=L1).to(dev)\\n",
              "    model = LNNUE(L1=L1).to(dev)\\n"
              "    if DRAW_W > 0:\\n"
              "        model.dhead = torch.nn.Linear(2 * L1, model.num_buckets).to(dev)\\n"
              "        torch.nn.init.zeros_(model.dhead.weight)\\n"
              "        torch.nn.init.zeros_(model.dhead.bias)\\n"
              "        model.dhead_p = torch.nn.Parameter(torch.zeros(model.num_buckets, device=dev))\\n"
              '        print(f"[draw] draw head on, loss weight {DRAW_W:g}", flush=True)\\n', "model construction")
        patch("    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\\n",
              '    if hasattr(model, "dhead"):   # its own group, the last: no weight decay, the tail learning rate\\n'
              '        groups.append({"params": list(model.dhead.parameters()) + [model.dhead_p], "weight_decay": 0.0,\\n'
              '                       "lr": LR * TAIL_LR_MULT})\\n'
              "    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\\n", "the optimizer line")
        open(dst, "w", encoding="utf-8").write(rt)
        print(f"  draw head: patched trainer {dst}")
    ''')

    COMMIT = "e6da953cd6375ddf672c924e300182b2a14c4d37"   # branch fused-act (PR #88, on PR #86's cb44cca)
    PINS = {"resident_train.py": "62b59768bda410ccc7bbb05463471c139898d13a9de6d1f7437c931782c96539",
            "ft_fused.py": "dccb4415a925f4c0625e484cf8b68e1b92f232e923aac3399a2217fe0622ebef",
            "check_fused.py": "a918ecd79de2f99cf488a8df5771aaf320e004f6c40ee451108cd0fb826238a7",
            "luminex_nnue_train.py": "ea5b94f2026c82b7a111d54d75fcf87fc30a8986290332b3dd446bb64f1222a8",
            "feeder.cpp": "e01bbbf99c8bd712b3fe28cdb90287e91f310b3eba8ebd922ece92aabfed8779"}
    DATA, TEST = "/marimo/data", "/marimo/fatest"
    TOOLS, OUT = TEST + "/tools", TEST + "/out"
    FLOAT_CK = "/marimo/out/luminex_gen768s_02.pt"   # the float phase's checkpoint, with the draw head
    t0 = time.time()
    if subprocess.run("pgrep -f '[r]esident_train'", shell=True, capture_output=True).stdout.strip():
        print("a training run is active; not starting"); return
    try:
        used = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=30).stdout.split()[0])
    except Exception as e:
        print(f"nvidia-smi failed ({e.__class__.__name__}); not starting"); return
    if used > 2048:
        print(f"the GPU holds {used:.0f} MiB; not starting"); return
    parts = sorted(glob.glob(DATA + "/t91_*.lcr"))
    if not parts:
        print(f"FAIL: no {DATA}/t91_*.lcr"); return
    ck = FLOAT_CK if os.path.exists(FLOAT_CK) else ""
    print(f"data: {len(parts)} LCR1 parts, the first {os.path.basename(parts[0])}; start: {ck or 'random weights'}", flush=True)
    os.makedirs(TOOLS, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else ""

    gh = f"https://raw.githubusercontent.com/changcheng967/Luminex/{COMMIT}/nnue/openi_upload/"
    for name, want in PINS.items():
        path = f"{TOOLS}/{name}"
        for url in (gh + name, "https://gh-proxy.com/" + gh + name):
            if sha(path) == want:
                break
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    open(path, "wb").write(r.read())
            except Exception as e:
                print(f"  {name}: {url[:60]}... failed ({e.__class__.__name__})", flush=True)
        if sha(path) != want:
            print(f"FAIL: {name} at {COMMIT[:7]} does not match sha256 {want}"); return
    print(f"code: {len(PINS)} files at {COMMIT[:7]}, sha256 ok", flush=True)
    open(f"{TOOLS}/patch_draw.py", "w", encoding="utf-8").write(PATCH_DRAW)
    r = subprocess.run([sys.executable, f"{TOOLS}/patch_draw.py", f"{TOOLS}/resident_train.py",
                        f"{TOOLS}/resident_train_dh.py"], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip(), flush=True)
    if r.returncode != 0:
        print("FAIL: the draw patch"); return

    def stream(title, argv, env, keys, logname, limit=900):
        """Runs one step in its own process group, killed at `limit` seconds or when the cell stops, so a
        hung step can neither hold the cell nor leave a process on the GPU."""
        print(f"\n== {title}", flush=True)
        ta, lines = time.time(), []
        with open(f"{OUT}/{logname}", "w") as lg:
            pr = subprocess.Popen(argv, env=env, cwd=TOOLS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, errors="replace", start_new_session=True)

            def kill():
                try:
                    os.killpg(pr.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            timer = threading.Timer(limit, kill)
            timer.start()
            try:
                for line in pr.stdout:
                    lg.write(line)
                    lg.flush()
                    lines.append(line.rstrip())
                    m = re.search(r" step (\d+) loss=", line)
                    if any(k in line for k in keys) or (m and int(m.group(1)) % 100 == 0):
                        print("   " + line.rstrip()[:220], flush=True)
                rc = pr.wait()
            finally:
                timer.cancel()
                if pr.poll() is None:   # the cell was stopped
                    kill()
                    pr.wait()
        print(f"   exit {rc} in {time.time() - ta:.0f} s"
              + (f" (killed at the {limit} s limit)" if rc == -signal.SIGKILL else ""), flush=True)
        if rc != 0:
            print("\n".join("   " + x[:220] for x in lines[-25:]), flush=True)
        return rc, lines

    env0 = {k: v for k, v in os.environ.items() if not k.startswith("NNUE_")}
    env0["PYTHONUNBUFFERED"] = "1"
    rc, lines = stream("1/2: check_fused.py --act (bit for bit, float and QAT)",
                       [sys.executable, "-u", f"{TOOLS}/check_fused.py", "--act", DATA] + ([ck] if ck else []), env0,
                       ("bit for bit", "fwd_loss", "ACT_", "Error", "FAIL"), "check_act.log")
    act_pass = rc == 0 and any(x.startswith("ACT_PASS") for x in lines)
    base = dict(env0, NNUE_DATA=DATA + "/t91_*.lcr", NNUE_MAX_POS="150000000", NNUE_VRAM_POS="150000000",
                NNUE_CONTINUE="0", NNUE_FEED_THREADS="8", NNUE_BUCKETS="8", NNUE_TAIL_LR_MULT="4", NNUE_L1="768",
                NNUE_BS="131072", NNUE_T_MAX_STEPS="100000", NNUE_BUDGET_SEC="0", NNUE_LOSS_POWER="3.0",
                NNUE_QUIET_LOSS="1,0.3,0", NNUE_GRAD_CLIP="1.0", NNUE_TARGET_CLIP="1500", NNUE_VAL_FRAC="0.002",
                NNUE_EVAL_EVERY="1000000", NNUE_SAVE_EVERY="1000000", NNUE_KEEP_EVERY="0", NNUE_COMPILE="1",
                NNUE_SEED="6", NNUE_OUT_DIR=OUT, NNUE_PROFILE="100", NNUE_PROFILE_WARMUP="300", NNUE_LR="5e-5",
                NNUE_QAT="0", NNUE_FT_DENSE="3", NNUE_DRAW_W="1e-3")
    if ck:
        base["NNUE_RESUME"] = ck
    res = {}
    for act in ("0", "1"):
        rc, lines = stream(f"2/2: float phase, NNUE_FT_ACT={act}",
                           [sys.executable, "-u", f"{TOOLS}/resident_train_dh.py"],
                           dict(base, NNUE_FT_ACT=act, NNUE_OUT_NAME=f"fa_float{act}.nnue"),
                           ("[profile]", "[data]", "[RESUME]", "[draw]", "[HEALTH", "DONE (", "FAIL", "Error", "Traceback"),
                           f"fa_float{act}.log")
        wall = gpu = None
        losses = {}
        for x in lines:
            m = re.search(r"\[profile\] \d+ steps of .*: ([0-9.]+) ms a step \(wall\), GPU kernels ([0-9.]+) ms", x)
            if m:
                wall, gpu = float(m.group(1)), float(m.group(2))
            m = re.search(r" step (\d+) loss=([0-9.]+)", x)
            if m:
                losses[int(m.group(1))] = float(m.group(2))
        res[act] = (rc, wall, gpu, losses)
    (rc0, w0, g0, l0), (rc1, w1, g1, l1) = res["0"], res["1"]
    common = sorted(set(l0) & set(l1))
    worst = max((abs(l0[s] - l1[s]) for s in common), default=float("nan"))
    print(f"\nlosses, ACT=0 vs ACT=1: {len(common)} logged steps, largest difference {worst:.5f} (the table "
          f"kernel's atomics differ run to run, ~1e-7 a step)", flush=True)
    ok = act_pass and rc0 == 0 and rc1 == 0 and w0 and w1 and w1 < w0 and len(common) > 0 and worst < 2e-4
    times = (f"float step {w0} ms -> {w1} ms wall, GPU {g0} -> {g1} ms"
             + (f", {w0 / w1 - 1:+.0%} steps/s" if w0 and w1 else ""))
    print(f"\n{'FUSED_ACT_OK' if ok else 'FUSED_ACT_NOT_OK'}: {times}; check_fused --act "
          f"{'passed' if act_pass else 'FAILED'} | {time.time() - t0:.0f} s (delete {TEST} when copied)", flush=True)


import sys as _sys


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for t in self.streams:
            t.write(x)
            t.flush()

    def flush(self):
        for t in self.streams:
            t.flush()


with open("/marimo/fatest.log", "w") as _log:
    _stdout = _sys.stdout
    _sys.stdout = _Tee(_stdout, _log)
    try:
        _fused_act_test()
    except BaseException:
        import traceback as _tb
        print("FAIL:", _tb.format_exc(), flush=True)
    finally:
        _sys.stdout = _stdout
