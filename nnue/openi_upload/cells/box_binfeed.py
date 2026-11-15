def _binfeed_retest():
    """Re-test of PR #85's binpack feeder after the 2026-11-15 box run that stopped for good at step 650, and of
    PR #86's QAT. About 12 minutes, nothing kept: writes /marimo/bfre (delete it afterwards) and the log
    /marimo/bfre.log; /marimo/data, /marimo/ckpt and /marimo/bftest are only read. Refuses while a training run is
    active or the GPU holds more than 2 GB.
      1. The stream alone, driven as the trainer drives it at the GPU's pace: 16 decode threads, a 72-chunk ring, a
         reservoir of a whole pass, then a 131,072-record pull every 12 ms for 4 passes of the Aug 2026 BT4 month.
         First cb44cca's binfeed.cpp (the code that stopped), then 786821a's (every wait re-checks every 20 ms;
         late = waits a timer ended with the condition true). A stall (30 s without records) is reported.
      2. The float phase as run 3 would train it on binpacks (L1 768, NNUE_FT_DENSE=3, the draw head, compiled):
         3,000 steps (2.8 passes of the month; the stream reaches its 4th pass), then 500 QAT steps in the same
         process (NNUE_QAT_AFTER, PR #86), with NNUE_STALL_SEC=60 so a stall is reported within a minute.
      3. The same float phase on LCR1 records (/marimo/data, 150M positions on the card), 600 steps: the baseline.
      4. check_fused.py --qat with step 2's float checkpoint: the fused exact QAT forward against the dense one.
    The last lines: RUN3_BINPACKS_OK or RUN3_BINPACKS_NOT_OK (both phases finished, no stall, float steps/s within
    3% of the LCR baseline's), and QAT86_OK or QAT86_NOT_OK."""
    import glob, hashlib, json, os, re, statistics, subprocess, sys, textwrap, time, urllib.parse, urllib.request

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

    # The stream driven alone (step 1): argv = code dir, binpack, decode threads, passes, seconds between pulls.
    STRESS = '''
    import os, sys, threading, time
    CODE, FILE, T, PASSES, SLEEP = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    os.environ.update(NNUE_DEVICE="cpu", NNUE_READ_THREADS="8", NNUE_VAL_FRAC="0.002", NNUE_BINPACK_CACHE="1")
    sys.path.insert(0, CODE)
    import numpy as np
    import resident_train as RT
    thr = min(int(RT.VAL_FRAC * 2 ** 32), 2 ** 32 - 1)
    bp = RT.Binpacks([FILE], thr, True)
    n = bp.select(0)
    bs = 131072
    S = n // bs * bs
    bp.start(6, T, 4 * T + 8)
    st = {"got": 0, "t": time.time()}
    def state():
        return f"; stream {bp.state()}" if hasattr(bp, "state") else ""
    def watchdog():
        while True:
            time.sleep(2)
            if time.time() - st["t"] > 30:
                print(f"STREAM_STALL: no records for {time.time() - st['t']:.0f} s at {st['got']:,} positions "
                      f"({st['got'] / n:.3f} passes){state()}", flush=True)
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()
    buf = np.empty((1 << 22, 32), np.uint8)
    for o in range(0, S, 1 << 22):
        k = min(1 << 22, S - o)
        bp.next_into(buf.ctypes.data, k)
        st["got"] += k
        st["t"] = time.time()
    t1, pulls = time.time(), 0
    while st["got"] < PASSES * n:
        bp.next_into(buf.ctypes.data, bs)
        pulls += 1
        if SLEEP:
            time.sleep(SLEEP)
        st["got"] += bs
        st["t"] = time.time()
    print(f"STREAM_OK: {st['got'] / n:.2f} passes, {pulls} pulls of {bs:,} in {time.time() - t1:.0f} s{state()}", flush=True)
    os._exit(0)
    '''

    NEW = "b725c583be96d945bc745043916408cc82b84170"   # branch binpack-qat: PR #86 with #85's fix 786821a merged
    OLD = "cb44ccae505a80b5789ef2d9785f0a301d25e3ad"   # the code that stopped at step 650
    PINS_NEW = {"resident_train.py": "05fa55a00b2455639aadfc73553fb49422e4ffbd7eb5518afe3c2bca3a395b92",
                "binfeed.cpp": "844a1814c2322474c2af4204bf16a444dcc448c37a80b9ea765a2ed6783c1b77",
                "ft_fused.py": "6db6cfc39c388b01b42fd9fe1f0d82e806e7964a9d2d2c327ca7de08f418968c",
                "check_fused.py": "a0e5d71f0eb52e8435be08e309886a648721554a0809360645af1670a256d837",
                "luminex_nnue_train.py": "ea5b94f2026c82b7a111d54d75fcf87fc30a8986290332b3dd446bb64f1222a8",
                "feeder.cpp": "e01bbbf99c8bd712b3fe28cdb90287e91f310b3eba8ebd922ece92aabfed8779",
                "pfetch.py": "cf24acc2e7482c8afb2862528e2aed6a1b7e4ce8e0e5f7e244e3e98e117a0ccb"}
    PINS_OLD = {"resident_train.py": "978617664877793aab2d6b4885396172c2e9234c48f5c4bfbfeb691c804d316a",
                "binfeed.cpp": "0a0432fe17b2b7a1f9042d7490c1f450245851da31820cc85f65391bde62d850",
                "luminex_nnue_train.py": "ea5b94f2026c82b7a111d54d75fcf87fc30a8986290332b3dd446bb64f1222a8"}
    NP_COMMIT = "ed84cc1fd2255faa0f33dcfa7839c9b6b4bb49b8"   # nnue-pytorch: the binpack reader
    MONTH = "test91-2026-08-aug-150gb6p.v6.relabel-BT4-tf13tune-histfix.binpack"
    HOME, DATA, TEST = "/marimo", "/marimo/data", "/marimo/bfre"
    START = HOME + "/ckpt/luminex_gen768s_01.nnue"   # the start the stopped run used
    LCR_REF = 67.7   # the current trainer's float steps/s at L1 768 with the draw head (the parent's figure)
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
        print(f"FAIL: no {DATA}/t91_*.lcr for the baseline and check_fused"); return
    for d in ("new", "old", "data", "out"):
        os.makedirs(f"{TEST}/{d}", exist_ok=True)

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()

    def fetch(commit, pins, dst):
        gh = f"https://raw.githubusercontent.com/changcheng967/Luminex/{commit}/nnue/openi_upload/"
        for name, want in pins.items():
            path = f"{dst}/{name}"
            for url in (gh + name, "https://gh-proxy.com/" + gh + name):
                if os.path.exists(path) and sha(path) == want:
                    break
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        open(path, "wb").write(r.read())
                except Exception as e:
                    print(f"  {name}: {url[:70]}... failed ({e.__class__.__name__})", flush=True)
            if not (os.path.exists(path) and sha(path) == want):
                raise SystemExit(f"FAIL: {name} at {commit[:7]} does not match sha256 {want}")
        print(f"code: {len(pins)} files at {commit[:7]} -> {dst}, sha256 ok", flush=True)

    fetch(NEW, PINS_NEW, f"{TEST}/new")
    fetch(OLD, PINS_OLD, f"{TEST}/old")
    np_dir = next((d for d in (f"{HOME}/tools/nnue-pytorch", f"{HOME}/bftest/tools/nnue-pytorch", f"{TEST}/np")
                   if os.path.exists(f"{d}/data_loader/cpp/lib/binpack.h")), None)
    if np_dir is None:
        np_dir = f"{TEST}/np"
        os.makedirs(np_dir, exist_ok=True)
        subprocess.run(f"curl -fsSL --retry 5 https://codeload.github.com/official-stockfish/nnue-pytorch/tar.gz/{NP_COMMIT}"
                       f" | tar xz -C {np_dir} --strip-components=1", shell=True, check=True)
    print(f"nnue-pytorch: {np_dir}", flush=True)
    q = urllib.parse.urlencode([("paths", "binpacks/v6-bt4/" + MONTH), ("expand", "true")]).encode()
    req = urllib.request.Request("https://huggingface.co/api/datasets/xushawn/t91-lgs/paths-info/main", data=q)
    info = json.load(urllib.request.urlopen(req, timeout=60))[0]
    size, oid = info["size"], info["lfs"]["oid"]
    month = next((p for p in (f"{HOME}/bftest/data/{MONTH}", f"{DATA}/{MONTH}", f"{TEST}/data/{MONTH}")
                  if os.path.exists(p) and os.path.getsize(p) == size and sha(p) == oid), None)
    if month is None:
        month = f"{TEST}/data/{MONTH}"
        subprocess.run([sys.executable, f"{TEST}/new/pfetch.py", "https://huggingface.co/datasets/xushawn/t91-lgs/resolve/"
                        f"main/binpacks/v6-bt4/{MONTH}", month, str(size), "8"], check=True)
        if sha(month) != oid:
            print(f"FAIL: {MONTH} sha256 differs from Hugging Face's"); return
    print(f"month: {month} ({size:,} bytes, sha256 ok)", flush=True)
    open(f"{TEST}/new/patch_draw.py", "w", encoding="utf-8").write(PATCH_DRAW)
    r = subprocess.run([sys.executable, f"{TEST}/new/patch_draw.py", f"{TEST}/new/resident_train.py",
                        f"{TEST}/new/resident_train_dh.py"], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip(), flush=True)
    if r.returncode:
        print("FAIL: the draw patch"); return
    open(f"{TEST}/stress.py", "w").write(textwrap.dedent(STRESS))

    env0 = {k: v for k, v in os.environ.items() if not k.startswith("NNUE_")}
    env0.update(PYTHONUNBUFFERED="1", NNUE_PYTORCH=np_dir)

    def stream(title, argv, env, keys, logname, limit):
        print(f"\n== {title}", flush=True)
        ta, lines = time.time(), []
        with open(f"{TEST}/out/{logname}", "w") as lg:
            pr = subprocess.Popen(argv, env=env, cwd=TEST, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  bufsize=1, errors="replace", start_new_session=True)
            timer = __import__("threading").Timer(limit, lambda: os.killpg(pr.pid, 9))
            timer.start()
            for line in pr.stdout:
                lg.write(line)
                lg.flush()
                lines.append(line.rstrip())
                m = re.search(r" step (\d+) loss=", line)
                if any(k in line for k in keys) or (m and int(m.group(1)) % 500 == 0):
                    print("   " + line.rstrip()[:240], flush=True)
            rc = pr.wait()
            timer.cancel()
        print(f"   exit {rc} in {time.time() - ta:.0f} s{' (killed at the time limit)' if rc == -9 else ''}", flush=True)
        if rc != 0:
            print("\n".join("   " + x[:240] for x in lines[-30:]), flush=True)
        return rc, lines

    res = {}
    for tag, code in (("old", f"{TEST}/old"), ("new", f"{TEST}/new")):
        rc, lines = stream(f"1/4: the stream alone, {tag} binfeed.cpp ({OLD[:7] if tag == 'old' else '786821a'}), "
                           "16 threads, a pull every 12 ms, 4 passes",
                           [sys.executable, "-u", f"{TEST}/stress.py", code, month, "16", "4", "0.012"], env0,
                           ("STREAM_", "Error", "FAIL", "[data]"), f"stress_{tag}.log", 600)
        res[tag] = next((x for x in lines if x.startswith("STREAM_")), f"no result (exit {rc})")
    keys = ("[data]", "[feed]", "[draw]", "[HEALTH", "== QAT PHASE", "DONE (", "[STALL]", "FAIL", "Error", "Traceback")
    base = dict(env0, NNUE_BUCKETS="8", NNUE_TAIL_LR_MULT="4", NNUE_L1="768", NNUE_BS="131072", NNUE_LOSS_POWER="3.0",
                NNUE_QUIET_LOSS="1,0.3,0", NNUE_GRAD_CLIP="1.0", NNUE_TARGET_CLIP="1500", NNUE_VAL_FRAC="0.002",
                NNUE_COMPILE="1", NNUE_FT_DENSE="3", NNUE_QAT="0", NNUE_DRAW_W="1e-3", NNUE_LR="5e-5", NNUE_SEED="6",
                NNUE_FEED_THREADS="8", NNUE_KEEP_EVERY="0", NNUE_OUT_DIR=f"{TEST}/out", NNUE_STALL_SEC="60")
    if os.path.exists(START):
        base["NNUE_RESUME"] = START
    rcA, linesA = stream("2/4: float phase on binpacks (the draw head), 3,000 steps, then 500 QAT steps in-process",
                         [sys.executable, "-u", f"{TEST}/new/resident_train_dh.py"],
                         dict(base, NNUE_BINPACKS=month, NNUE_T_MAX_STEPS="3000", NNUE_EVAL_EVERY="3000",
                              NNUE_SAVE_EVERY="3000", NNUE_QAT_AFTER="500", NNUE_QAT_LR="2e-5", NNUE_QAT_EVERY="500",
                              NNUE_OUT_NAME="bfre.nnue"), keys, "binpacks.log", 1500)
    rcB, linesB = stream("3/4: the same float phase on LCR1 records (150M on the card), 600 steps: the baseline",
                         [sys.executable, "-u", f"{TEST}/new/resident_train_dh.py"],
                         dict(base, NNUE_DATA=DATA + "/t91_*.lcr", NNUE_MAX_POS="150000000", NNUE_VRAM_POS="150000000",
                              NNUE_T_MAX_STEPS="600", NNUE_EVAL_EVERY="600", NNUE_SAVE_EVERY="600",
                              NNUE_OUT_NAME="bfre_lcr.nnue"), keys, "lcr.log", 900)
    ck = f"{TEST}/out/bfre.pt"
    rcC, linesC = stream("4/4: check_fused.py --qat (the fused exact QAT forward against the dense one)",
                         [sys.executable, "-u", f"{TEST}/new/check_fused.py", "--qat", DATA] + ([ck] if os.path.exists(ck) else []),
                         env0, ("QAT feature layer", "loss:", "GRAD_", "Error", "FAIL"), "check_qat.log", 900)

    def rates(lines):
        """Median steps/s of each phase's 50-step readings after its first 300 steps."""
        out, phase, first = {}, "float", None
        for x in lines:
            if "== QAT PHASE" in x:
                phase, first = "qat", None
            m = re.search(r" step (\d+) loss=.* \| ([0-9.]+) steps/s", x)
            if m:
                s = int(m.group(1))
                first = s if first is None else first
                if s >= first + 300:
                    out.setdefault(phase, []).append(float(m.group(2)))
        return {k: statistics.median(v) for k, v in out.items() if v}

    ra, rb = rates(linesA), rates(linesB)
    stalls = [x for x in linesA + linesB if x.startswith("[STALL]")]
    doneA = sum(1 for x in linesA if x.startswith("DONE ("))
    qat_ran = any("== QAT PHASE" in x for x in linesA)
    exact = next((x.strip() for x in linesC if "QAT feature layer" in x), "no QAT feature-layer line")
    print("\n== SUMMARY", flush=True)
    print(f"stream alone, old: {res['old']}", flush=True)
    print(f"stream alone, new: {res['new']}", flush=True)
    print(f"float steps/s, binpacks: {ra.get('float', 'n/a')}; LCR1 baseline: {rb.get('float', 'n/a')} (the parent's "
          f"{LCR_REF}); QAT steps/s in-process on NNUE_FT_DENSE=3: {ra.get('qat', 'n/a')}", flush=True)
    print(f"stall reports: {len(stalls)}" + "".join("\n   " + x[:240] for x in stalls[:40]), flush=True)
    print(f"check_fused --qat: {exact}; exit {rcC}", flush=True)
    fa, fb = ra.get("float"), rb.get("float")
    ok3 = rcA == 0 and doneA == 2 and not stalls and fa is not None and fb is not None and fa >= 0.97 * fb
    print(f"{'RUN3_BINPACKS_OK' if ok3 else 'RUN3_BINPACKS_NOT_OK'}: float {fa} steps/s on binpacks against {fb} on "
          f"LCR1 records; {doneA} of 2 phases finished; {len(stalls)} stall report line(s)", flush=True)
    ok6 = rcA == 0 and qat_ran and doneA == 2 and rcC == 0
    print(f"{'QAT86_OK' if ok6 else 'QAT86_NOT_OK'}: in-process QAT {'ran' if qat_ran else 'did not run'} at "
          f"{ra.get('qat', 'n/a')} steps/s; check_fused --qat exit {rcC} | {time.time() - t0:.0f} s "
          f"(delete {TEST} when copied)", flush=True)
    return True



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


with open("/marimo/bfre.log", "w") as _log:
    _stdout = _sys.stdout
    _sys.stdout = _Tee(_stdout, _log)
    _done = False
    try:
        _done = _binfeed_retest()
    except BaseException:
        import traceback as _tb
        print("FAIL:", _tb.format_exc(), flush=True)
    finally:
        if not _done:   # every exit, a refusal or a failure too, ends on the verdict a chain waits for
            print("RUN3_BINPACKS_NOT_OK: the test did not run to its end (the lines above say why)", flush=True)
            print("QAT86_NOT_OK: the test did not run to its end (the lines above say why)", flush=True)
        _sys.stdout = _stdout
