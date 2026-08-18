#!/usr/bin/env python3
"""C500 PER-FRAME trainer (VRAM-resident, OOM-safe). Featurize one frame in <=CAP-position
loads straight to VRAM -> train -> free -> next load (big frames auto-split into parts).
bf16 + grad-clip (v6 fix). Clean per-frame output. No whole-frame VRAM spike (cat frees
each list immediately; PYTORCH_CUDA_ALLOC_CONF=expandable_segments kills fragmentation).

RUN (no args):
  NNUE_L1=512 NNUE_EPOCHS=1 NNUE_BS=131072 NNUE_LR=1e-3 NNUE_FEAT_THREADS=14 NNUE_GRAD_CLIP=1.0
  NNUE_VRAM_POS=180000000   # max positions per VRAM load (big frames split into parts)
"""
import os, sys, subprocess, time, glob
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # anti-fragmentation
import numpy as np, torch
from c2net.context import prepare, upload_output
ctx = prepare(); out_dir = ctx.output_path

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
EPOCHS = int(os.environ.get("NNUE_EPOCHS", "1"))
BS   = int(os.environ.get("NNUE_BS", "131072"))
LR   = float(os.environ.get("NNUE_LR", "1e-3"))
NTH  = int(os.environ.get("NNUE_FEAT_THREADS", "14"))
BUDGET = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.5*3600))))
BUF  = int(os.environ.get("NNUE_BUF", "2000000"))
CAP  = int(os.environ.get("NNUE_VRAM_POS", "180000000"))   # max pos per VRAM load (big frames split)
_gc = float(os.environ.get("NNUE_GRAD_CLIP", "1.0"))   # v6 fix (prevents gradient spikes)
_wc = float(os.environ.get("NNUE_WCLAMP", "0"))          # OFF (fallback only; root cause = tail wd + amsgrad)
REC  = 136; SCALE = 400.0
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("NNUE_OUT_NAME", "luminex_v6.nnue"); OUT_BASE = OUT[:-5] if OUT.endswith(".nnue") else OUT
total_bytes = sum(os.path.getsize(f) for f in FRAMES)
print(f"per-frame train (OOM-safe): {len(FRAMES)} frames, {total_bytes/1e9:.2f}GB, L1={L1} bs={BS} cap={CAP:,} feat_threads={NTH} grad-clip={_gc} device={device}", flush=True)

model = LNNUE(L1=L1).to(device)
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
print(f"  [opt] FT params={sum(p.numel() for p in _ft_params):,} (wd=0) | tail params={sum(p.numel() for p in _tail_params):,} (wd={_tail_wd}) | amsgrad=True", flush=True)
est_pos = int(total_bytes / 1.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS * est_pos // BS))
print(f"  cosine T_max={max(1, EPOCHS * est_pos // BS)} steps", flush=True)

def _budget_hit(gstep, total_pos):
    print(f">>> BUDGET hit at step {gstep} - saving & stopping", flush=True)
    save_nnue(model, os.path.join(out_dir, OUT))
    upload_output(); print(f"DONE (budget) - {OUT}: {gstep} steps, {total_pos:,} pos", flush=True)
    sys.exit(0)

gstep = 0; t0 = time.time(); total_pos = 0
for epoch in range(EPOCHS):
    for fi, frame_path in enumerate(FRAMES):
        if BUDGET and time.time() - t0 >= BUDGET:
            save_nnue(model, os.path.join(out_dir, OUT)); break
        dec = "zstd -dc" if frame_path.endswith(".zst") else "xz -dc"
        cmd = f"{dec} {frame_path} | {FEAT} --stream --input /dev/stdin --threads {NTH}"
        p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0)
        part = 0
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
            for i in range(0, N, BS):
                idx = perm[i:i + BS]
                wi = w[idx].long(); bi = b[idx].long(); si = s[idx]; ti = t[idx]
                opt.zero_grad()
                if os.environ.get("NNUE_AUTOCAST", "1") != "0":
                    ctx_ac = torch.autocast(device_type=device, dtype=torch.bfloat16)
                else:
                    ctx_ac = torch.autocast(device_type=device, enabled=False)   # fp32 (no bf16 drift)
                with ctx_ac:
                    pred = model(wi, bi, si)
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
                if gstep % 50 == 0:
                    try: _l2 = f" L2|max|={float(model.l2.weight.detach().abs().max().item()):.2f}"
                    except Exception: _l2 = ""
                    dt = time.time() - t0
                    print(f"  e{epoch} f{fi+1}p{part} step {gstep} loss={loss.item():.5f}{_l2} | {total_pos/1e6:.0f}M+{i/1e6:.0f}M | {gstep/max(dt,1):.1f} steps/s", flush=True)
                if BUDGET and time.time() - t0 >= BUDGET:
                    del w, b, s, t, perm; torch.cuda.empty_cache()
                    try: p.stdout.close(); p.terminate()
                    except Exception: pass
                    _budget_hit(gstep, total_pos)
            total_pos += N
            del w, b, s, t, perm; torch.cuda.empty_cache()
        try: p.stdout.close(); p.wait()
        except Exception: pass
        print(f"  [frame {fi+1} done: cum {total_pos:,} ({total_pos/1e9:.2f}B), {time.time()-t0:.0f}s]", flush=True)
        save_nnue(model, os.path.join(out_dir, OUT))   # incremental save after each frame
    else:
        continue
    break

save_nnue(model, os.path.join(out_dir, OUT))
print(f"DONE - {OUT}: {gstep} steps, {total_pos:,} pos in {time.time()-t0:.0f}s. uploading...", flush=True)
upload_output()
print(f"Quantize locally: python quantize_i8.py {OUT} {OUT_BASE}_i8.nnue", flush=True)
