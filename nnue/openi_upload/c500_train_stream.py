#!/usr/bin/env python3
"""C500 PER-FRAME trainer (VRAM-resident, OOM-safe). Featurize one frame in <=CAP-position
loads straight to VRAM -> train -> free -> next load (big frames auto-split into parts).
bf16 + grad-clip (v6 fix). Clean per-frame output. No whole-frame VRAM spike (cat frees
each list immediately; PYTORCH_CUDA_ALLOC_CONF=expandable_segments kills fragmentation).

RUN (no args):
  NNUE_L1=512 NNUE_EPOCHS=1 NNUE_BS=131072 NNUE_LR=1e-3 NNUE_FEAT_THREADS=14 NNUE_GRAD_CLIP=1.0
  NNUE_VRAM_POS=180000000   # max positions per VRAM load (big frames split into parts)
"""
import os, sys, subprocess, time, glob, copy
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # anti-fragmentation
import numpy as np, torch
try:
    from c2net.context import prepare, upload_output
    ctx = prepare(); out_dir = ctx.output_path
except Exception:   # non-OpenI environment (smoke test / local box)
    import tempfile
    class _Ctx: output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    ctx = _Ctx(); out_dir = ctx.output_path
    os.makedirs(out_dir, exist_ok=True)
    upload_output = lambda *a, **k: None

SEARCH_ROOTS = ["/tmp/code", "/tmp/dataset", "/tmp/frames", "/tmp"]
def _walk():
    for root in SEARCH_ROOTS:
        if os.path.isdir(root):
            for dp, dn, fn in os.walk(root, followlinks=True):
                yield dp, fn
def _find_file(name):
    for dp, fn in _walk():
        if name in fn: return os.path.join(dp, name)
    return None

FRAMES_DIR = os.environ.get("NNUE_FRAMES_DIR", "/tmp/code/frames")
FRAMES = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.zst")) + glob.glob(os.path.join(FRAMES_DIR, "frame_*.xz")))
if not FRAMES:
    for dp, _ in _walk():
        FRAMES += sorted(glob.glob(os.path.join(dp, "frame_*.zst")) + glob.glob(os.path.join(dp, "frame_*.xz")))
    FRAMES = sorted(set(FRAMES))
if not FRAMES:
    tar = _find_file("gamepack.tar")
    if tar:
        os.makedirs(FRAMES_DIR, exist_ok=True)
        print(f"  extracting {tar} -> {FRAMES_DIR}", flush=True)
        subprocess.run(f"tar xf '{tar}' -C '{FRAMES_DIR}'", shell=True)
        FRAMES = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.zst")) + glob.glob(os.path.join(FRAMES_DIR, "frame_*.xz")))
assert FRAMES, "no frame_*.zst/.xz found and no gamepack.tar"
if any(f.endswith(".zst") for f in FRAMES) and not os.path.exists("/usr/bin/zstd"):
    subprocess.run("apt-get install -y zstd >/dev/null 2>&1 || pip install -q zstandard", shell=True)

train_mod = _find_file("luminex_nnue_train.py")
CODE_DIR = os.environ.get("CODE_DIR") or (os.path.dirname(train_mod) if train_mod else "/tmp/code")
sys.path.insert(0, CODE_DIR)
FEAT = os.environ.get("FEAT") or _find_file("luminex-featurize") or os.path.join(CODE_DIR, "luminex-featurize")
assert os.path.exists(FEAT), f"featurizer missing: {FEAT}"
from luminex_nnue_train import LNNUE, save_nnue

L1   = int(os.environ.get("NNUE_L1", "512"))
BS   = int(os.environ.get("NNUE_BS", "131072"))
LR   = float(os.environ.get("NNUE_LR", "1e-3"))
NTH  = int(os.environ.get("NNUE_FEAT_THREADS", "14"))
BUDGET = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.5*3600))))
BUF  = int(os.environ.get("NNUE_BUF", "2000000"))
CAP  = int(os.environ.get("NNUE_VRAM_POS", "180000000"))   # max pos per VRAM load (big frames split)
_gc = float(os.environ.get("NNUE_GRAD_CLIP", "1.0"))   # v6 fix (prevents gradient spikes)
_wc = float(os.environ.get("NNUE_WCLAMP", "0"))          # OFF (fallback only; root cause = tail wd + amsgrad)
# Auto-convergence: pick the largest data subset that converges (loss plateaus +
# cosine LR reaches zero) inside the time budget. No manual epoch count needed.
_CONV_PATIENCE = int(os.environ.get("NNUE_CONV_PATIENCE", "1200"))  # steps without improvement to declare convergence
_CONV_MIN_EPOCHS = int(os.environ.get("NNUE_CONV_MIN_EPOCHS", "1")) # minimum passes before early-stop is armed
# Single-pass mode: disable convergence detector entirely — it keeps killing
# training prematurely because per-part loss varies (harder positions = higher loss)
if _CONV_TARGET_PASSES <= 1:
    _CONV_PATIENCE = 999999999  # effectively disabled
_CONV_TARGET_PASSES = float(os.environ.get("NNUE_CONV_PASSES", "6")) # expected passes for subset sizing
_CAL_STEPS = 60   # calibration steps to measure throughput
_FEAT_CACHE = os.environ.get("NNUE_FEAT_CACHE", "1") != "0"  # cache featurized frames across epochs
REC  = 136; SCALE = 400.0
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("NNUE_OUT_NAME", "luminex_v6.nnue"); OUT_BASE = OUT[:-5] if OUT.endswith(".nnue") else OUT
total_bytes = sum(os.path.getsize(f) for f in FRAMES)
print(f"per-frame train (OOM-safe): {len(FRAMES)} frames, {total_bytes/1e9:.2f}GB, L1={L1} bs={BS} cap={CAP:,} feat_threads={NTH} grad-clip={_gc} device={device}", flush=True)

model = LNNUE(L1=L1).to(device)
# Multi-block resume: upload the previous block's luminex_v8.pt next to the code;
# weights + global step are restored (optimizer state rebuilds in ~1K steps).
_resume = os.environ.get("NNUE_RESUME") or os.path.join(os.path.dirname(__file__) or ".", OUT_BASE + ".pt")
_opt_state = None  # deferred: optimizer doesn't exist yet at resume time
if os.path.exists(_resume) and os.environ.get("NNUE_RESUME", "1") != "0":
    try:
        _ck = torch.load(_resume, map_location=device, weights_only=False)
        model.load_state_dict(_ck["model"]); gstep = _ck.get("gstep", 0)
        _opt_state = _ck.get("opt")  # Adam/AdamW moments — the key to cross-block continuity
        if _opt_state: print(f"[RESUME] + optimizer state ({len(_opt_state['state'])} params) — no warm-up loss", flush=True)
        print(f"[RESUME] loaded {_resume} at gstep={gstep}", flush=True)
    except Exception as _e:
        print(f"[RESUME] FAILED ({_e}) - training from scratch", flush=True)
model.probe_ft(device)   # EmbeddingBag FT
print(f"  [LNNUE] ft_mode={model.ft_mode} (compile OFF)", flush=True)
# Phase 0 root-cause L2 fix: decay ONLY the tail (where L2/SCReLU feedback grows weights),
# NOT the FT (protects rare king/piece/square buckets from uniform-decay undertraining, #45).
# + AMSGrad (bounds effective LR per-param, prevents any single weight running away).
_ft_params  = [p for p in model.parameters() if p.numel() > 100000]   # FT embedding (12.6M elements)
_tail_params = [p for p in model.parameters() if p.numel() <= 100000]  # L2/L3/out (<10K each)
_tail_wd = float(os.environ.get("NNUE_TAIL_WD", "1e-2"))
opt = torch.optim.AdamW([
    {"params": _ft_params,  "weight_decay": 0.0},        # NO decay on FT (rare-bucket protection)
    {"params": _tail_params, "weight_decay": _tail_wd},  # tail decay (fixes L2 SCReLU feedback)
], lr=LR, amsgrad=True)
if _opt_state:
    try:
        opt.load_state_dict(_opt_state)
        print("  [opt] AdamW moments restored — zero warm-up penalty", flush=True)
    except Exception as _e:
        print(f"  [opt] state restore FAILED ({_e}) — rebuilding (1K-step warm-up)", flush=True)
print(f"  [opt] FT params={sum(p.numel() for p in _ft_params):,} (wd=0) | tail params={sum(p.numel() for p in _tail_params):,} (wd={_tail_wd}) | amsgrad=True", flush=True)

# ---- AUTO-CONVERGENCE SIZING ------------------------------------------------
# Calibrate throughput, then pick the largest data subset that can be trained
# to convergence (loss plateaus + cosine LR reaches zero) inside the budget.
_OVERHEAD_SEC = 300   # startup, model init, first featurize warm-up
_SPS_FALLBACK = 4.0   # conservative steps/sec if calibration unavailable (resume blocks skip calib)

_sps = _SPS_FALLBACK
if gstep == 0:  # fresh run: calibrate with a tiny forward+backward
    import time as _t
    _cal_t0 = _t.time()
    _dummy_w = torch.randint(0, 24576, (BS, 32), device=device, dtype=torch.long)
    _dummy_b = torch.randint(0, 24576, (BS, 32), device=device, dtype=torch.long)
    _dummy_s = torch.ones(BS, device=device)
    _dummy_t = torch.zeros(BS, device=device)
    for _ in range(_CAL_STEPS):
        opt.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _p = model(_dummy_w, _dummy_b, _dummy_s)
            _l = ((_p - _dummy_t).abs().mean())
        _l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), _gc)
        opt.step()
    _sps = _CAL_STEPS / (_t.time() - _cal_t0)
    print(f"  [calibrate] {_sps:.1f} steps/s (BS={BS})", flush=True)
    model = LNNUE(L1=L1).to(device)  # reset — calibration dirtied the weights
    # re-init optimizer (fresh model params)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.numel() > 100000], "weight_decay": 0.0},
        {"params": [p for p in model.parameters() if p.numel() <= 100000], "weight_decay": _tail_wd},
    ], lr=LR, amsgrad=True)

_avail_sec = max(600, BUDGET - _OVERHEAD_SEC)
_total_steps = int(_avail_sec * _sps)
_total_visits = _total_steps * BS
_subset_pos = int(_total_visits / _CONV_TARGET_PASSES)
# select frames until subset is filled
_subset_frames = []; _subset_bytes = 0
for f in FRAMES:
    fb = os.path.getsize(f)
    if _subset_bytes + fb / 1.05 > _subset_pos:
        break
    _subset_frames.append(f); _subset_bytes += int(fb / 1.05)
# FRAME SKIP must be applied BEFORE subset selection, not after
_FRAMES_SKIP = int(os.environ.get("NNUE_FRAME_SKIP", "0"))  # skip first N frames (fresh-data mode)
if _FRAMES_SKIP > 0 and _FRAMES_SKIP < len(FRAMES):
    FRAMES = FRAMES[_FRAMES_SKIP:]
    # re-run subset selection on the remaining (fresh) frames
    _subset_frames = []; _subset_bytes = 0
    for f in FRAMES:
        fb = os.path.getsize(f)
        if _subset_bytes + fb / 1.05 > _subset_pos:
            break
        _subset_frames.append(f); _subset_bytes += int(fb / 1.05)
FRAMES = _subset_frames if _subset_frames else FRAMES[:1]  # at least 1 frame
est_pos = sum(os.path.getsize(f) for f in FRAMES) // 1.05
EPOCHS = max(_CONV_MIN_EPOCHS, int(_total_visits / max(1, est_pos)))
T_MAX = min(_total_steps, EPOCHS * est_pos // BS)
# LR schedule: exponential decay (gamma=0.992/epoch, SF's proven schedule)
# NOT cosine — cosine anneals to zero too fast; exponential preserves signal longer
_LR_GAMMA = float(os.environ.get("NNUE_LR_GAMMA", "0.992"))
sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=_LR_GAMMA)
print(f"  [sched] exponential gamma={_LR_GAMMA} (initial lr={LR})", flush=True)

# SWA: average weights from the last 25% of training (free quality boost)
_SWA_START = int(os.environ.get("NNUE_SWA_START", "0"))  # 0=disabled; set to epoch number to enable
_swa_model = None; _swa_n = 0
if _SWA_START > 0:
    _swa_model = copy.deepcopy(model)
    for p in _swa_model.parameters(): p.data.zero_()
    print(f"  [swa] enabled from epoch {_SWA_START}", flush=True)

def _swa_update():
    global _swa_n
    if _swa_model is None: return
    _swa_n += 1
    with torch.no_grad():
        for swa_p, model_p in zip(_swa_model.parameters(), model.parameters()):
            swa_p.data += (model_p.data - swa_p.data) / _swa_n
print(f"  [auto-conv] subset={len(FRAMES)} frames (~{est_pos/1e9:.1f}B pos) | epochs={EPOCHS} | "
      f"T_max={T_MAX} steps | est total visits={EPOCHS*est_pos/1e9:.1f}B | budget {_avail_sec}s @ {_sps:.1f} st/s", flush=True)

# convergence state
_best_loss = float("inf"); _stale_steps = 0

def _budget_hit(gstep, total_pos):
    print(f">>> BUDGET hit at step {gstep} - saving & stopping", flush=True)
    save_nnue(model, os.path.join(out_dir, OUT))
    upload_output(); print(f"DONE (budget) - {OUT}: {gstep} steps, {total_pos:,} pos", flush=True)
    sys.exit(0)

gstep = globals().get('gstep', 0); t0 = time.time(); total_pos = 0   # keeps resumed gstep
for epoch in range(EPOCHS):
    for fi, frame_path in enumerate(FRAMES):
        _stale_steps = 0; _best_loss = float("inf")   # reset BOTH per frame: new frames start
        # with higher loss (unknown patterns); stale must mean "flat on THIS frame", not
        # "not beating a global best set on a different, easier frame"
        if BUDGET and time.time() - t0 >= BUDGET:
            save_nnue(model, os.path.join(out_dir, OUT)); break
        # FEATURIZE CACHE: first epoch featurizes and caches to /tmp; later
        # epochs read the cache (saves ~70s/frame — 25%+ of multi-epoch budget).
        _fc = f"/tmp/featcache_{os.path.basename(frame_path)}.raw"
        if _FEAT_CACHE and epoch == 0:
            dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
            cmd = f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {NTH} | tee {_fc}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0)
        elif _FEAT_CACHE and os.path.exists(_fc) and os.path.getsize(_fc) > REC * 1000:
            print(f"  [cache] reading {os.path.basename(_fc)} ({os.path.getsize(_fc)/1e6:.0f}MB)", flush=True)
            p = subprocess.Popen(["cat", _fc], stdout=subprocess.PIPE, bufsize=0)
        else:
            dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
            cmd = f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {NTH}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0)
        part = 0
        _health = None   # set by the parts loop; stays None if the frame yields nothing
        while True:   # process frame in <=CAP-position VRAM loads (big frames -> multiple parts)
            ws, bs, ss, ts = [], [], [], []
            got = 0
            while got < CAP:
                data = bytearray()
                while len(data) < BUF * REC:
                    d = p.stdout.read(min(1 << 20, BUF * REC - len(data)))
                    if not d: break
                    data += d
                m = len(data) // REC
                if m == 0: break
                a = np.frombuffer(bytes(data[:m * REC]), dtype=np.uint8).reshape(m, REC)
                ws.append(torch.from_numpy(a[:, :64].copy().view(np.int16).reshape(m, 32)).to(device))
                bs.append(torch.from_numpy(a[:, 64:128].copy().view(np.int16).reshape(m, 32)).to(device))
                ss.append(torch.from_numpy(a[:, 128:132].copy().view(np.float32).reshape(m)).to(device))
                ts.append(torch.from_numpy(a[:, 132:136].copy().view(np.float32).reshape(m)).to(device))
                got += m
            if got == 0: break
            part += 1
            if part == 1: print(f"[featurizing+training frame {fi+1}/{len(FRAMES)} ...]", flush=True)
            # cat each, FREE its list immediately -> avoids the 2x peak that OOM'd before
            w = torch.cat(ws); del ws
            b = torch.cat(bs); del bs
            s = torch.cat(ss); del ss
            t = torch.cat(ts); del ts
            N = got
            print(f"  [frame {fi+1} part {part}: {N:,} pos -> train]", flush=True)
            perm = torch.randperm(N, device=device)
            _POWER = float(os.environ.get("NNUE_LOSS_POWER", "2.6"))  # 0 = old sigmoid-MSE
            _FEN_SKIP = float(os.environ.get("NNUE_FEN_SKIP", "0.3"))  # skip prob for noisy positions
            for i in range(0, N, BS):
                idx = perm[i:i + BS]
                wi = w[idx].long(); bi = b[idx].long(); si = s[idx]; ti = t[idx]
                # Smart FEN skip: drop positions where |target| is extreme (>800cp)
                # — these are usually won/lost positions where eval adds noise
                if _FEN_SKIP > 0:
                    _keep = ti.abs() < 800.0
                    if _keep.sum() < BS // 4: continue  # skip batch if too few survive
                    wi = wi[_keep]; bi = bi[_keep]; si = si[_keep]; ti = ti[_keep]
                    if len(ti) < BS // 8: continue
                opt.zero_grad()
                if os.environ.get("NNUE_AUTOCAST", "1") != "0":
                    ctx_ac = torch.autocast(device_type=device, dtype=torch.bfloat16)
                else:
                    ctx_ac = torch.autocast(device_type=device, enabled=False)   # fp32 (no bf16 drift)
                with ctx_ac:
                    pred = model(wi, bi, si)
                    if _POWER > 0:
                        # Power-2.6 loss in SIGMOID space (SF's proven formula):
                        # diff = |σ(pred/SCALE) - σ(target/SCALE)|, loss = diff^2.6
                        # NOT raw cp space — raw cp gradients are ~6500x larger and
                        # get annihilated by gradient clipping (the bug that caused
                        # loss to flatline at ~230K with zero effective learning)
                        _sdiff = (torch.sigmoid(pred / SCALE) - torch.sigmoid(ti / SCALE)).abs()
                        loss = (_sdiff ** _POWER).mean()
                    else:
                        loss = ((torch.sigmoid(pred / SCALE) - torch.sigmoid(ti / SCALE)) ** 2).mean()
                loss.backward()
                if _gc > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _gc)   # v6 fix
                opt.step()
                if _wc > 0:   # hard weight clamp: stops bf16-driven L2 drift (|W| can't exceed _wc)
                    with torch.no_grad():
                        for p in model.parameters():
                            p.clamp_(-_wc, _wc)
                sched.step(); gstep += 1
                if _swa_model is not None and epoch >= _SWA_START:
                    _swa_update()
                # convergence tracking: best loss + stale counter
                _cur = loss.item()
                if _cur < _best_loss - 1e-6:
                    _best_loss = _cur; _stale_steps = 0
                else:
                    _stale_steps += 1
                if gstep % 50 == 0:
                    try: _l2 = f" L2|max|={float(model.l2.weight.detach().abs().max().item()):.2f}"
                    except Exception: _l2 = ""
                    _conv = f" conv_stale={_stale_steps}" if _stale_steps > _CONV_PATIENCE // 4 else ""
                    dt = time.time() - t0
                    print(f"  e{epoch} f{fi+1}p{part} step {gstep} loss={_cur:.5f}{_l2}{_conv} | {total_pos/1e6:.0f}M+{i/1e6:.0f}M | {gstep/max(dt,1):.1f} steps/s", flush=True)
                # AUTO-CONVERGENCE early stop: after minimum passes, if loss has been flat
                # for _CONV_PATIENCE steps, the model has converged — save and exit
                if (epoch >= _CONV_MIN_EPOCHS - 1 and _stale_steps >= _CONV_PATIENCE):
                    del w, b, s, t, perm; torch.cuda.empty_cache()
                    try: p.stdout.close(); p.terminate()
                    except Exception: pass
                    print(f">>> CONVERGED at step {gstep} (loss flat {_CONV_PATIENCE} steps, "
                          f"best={_best_loss:.5f}) — saving & stopping", flush=True)
                    save_nnue(model, os.path.join(out_dir, OUT))
                    torch.save({"model": model.state_dict(), "gstep": gstep, "opt": opt.state_dict()},
                               os.path.join(out_dir, OUT_BASE + ".pt"))
                    upload_output()
                    print(f"DONE (converged) - {OUT}: {gstep} steps, {total_pos:,} pos", flush=True)
                    sys.exit(0)
                if BUDGET and time.time() - t0 >= BUDGET:
                    del w, b, s, t, perm; torch.cuda.empty_cache()
                    try: p.stdout.close(); p.terminate()
                    except Exception: pass
                    _budget_hit(gstep, total_pos)
            total_pos += N
            # v8 health line: computed BEFORE the tensors are freed below.
            # Catches failure-archive signatures loss alone missed (v7-run4
            # weight-growth saturation, eval-scale compression, silent stall).
            _health = None
            try:
                _m = min(200000, int(w.shape[0]))
                if _m > 0:
                    with torch.no_grad():
                        _pv = model(w[:_m].long(), b[:_m].long(), s[:_m]).float()
                        _tv = t[:_m].float()
                        _mae = (_pv - _tv).abs().mean().item()
                        _ps, _ts = _pv.std().item(), _tv.std().item()
                        _wn = " ".join(f"{_nm}={float(getattr(model, _nm).weight.detach().norm().item()):.0f}"
                                       for _nm in ("ft", "emb", "l1", "l2", "out") if hasattr(model, _nm))
                    _health = f"MAE={_mae:.1f}cp predSTD={_ps:.0f} tgtSTD={_ts:.0f} | {_wn}"
            except Exception as _e:
                _health = f"unavailable ({_e})"
            del w, b, s, t, perm; torch.cuda.empty_cache()
        try: p.stdout.close(); p.wait()
        except Exception: pass
        if _health:
            print(f"  [HEALTH f{fi+1}] {_health}", flush=True)
        else:
            # A frame that yields ZERO positions means the featurizer crashed
            # (e.g. binary/glibc mismatch) — an empty epoch must never "succeed".
            raise SystemExit(f"FATAL: frame {fi+1} produced 0 positions — featurizer "
                             f"broken? Check GLIBC/exec-bit on {FEAT}.")
        print(f"  [frame {fi+1} done: cum {total_pos:,} ({total_pos/1e9:.2f}B), {time.time()-t0:.0f}s]", flush=True)
        save_nnue(model, os.path.join(out_dir, OUT))   # incremental save after each frame
        torch.save({"model": model.state_dict(), "gstep": gstep, "opt": opt.state_dict()}, os.path.join(out_dir, OUT_BASE + ".pt"))
    else:
        continue
    break

# SWA swap: if SWA was active, use the averaged weights for the final save
if _swa_model is not None and _swa_n > 0:
    print(f"  [swa] swapping in averaged weights ({_swa_n} updates)", flush=True)
    model.load_state_dict(_swa_model.state_dict())

save_nnue(model, os.path.join(out_dir, OUT))
print(f"DONE - {OUT}: {gstep} steps, {total_pos:,} pos in {time.time()-t0:.0f}s. uploading...", flush=True)
upload_output()
print(f"Quantize locally: python quantize_i8.py {OUT} {OUT_BASE}_i8.nnue", flush=True)
