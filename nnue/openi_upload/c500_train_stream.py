#!/usr/bin/env python3
"""C500 STREAMING trainer: train on the full Game-Pack .xz (billions of positions) WITHOUT
materializing features on disk/VRAM. luminex-featurize --stream reads the .xz and pipes
136-byte feature records straight here; a BUF-position shuffle buffer lives in CPU RAM and
batches are shuttled to the GPU. C500 disk footprint = just the .xz + binaries (~3-4 GB).

The featurizer (~2-3M pos/s on 8 cores) outruns training (~900K), so pipe backpressure
throttles featurize to the train rate — no multi-GB staging anywhere.

RUN (no args), env-tunable:
  NNUE_XZ=/tmp/dataset/<gamepack.xz>   # the uploaded Game-Pack dataset
  FEAT=/tmp/code/luminex-featurize
  NNUE_L1=512 NNUE_EPOCHS=1 NNUE_BS=32768 NNUE_LR=1e-3 NNUE_BUF=2000000 NNUE_FEAT_THREADS=8
"""
import os, sys, subprocess, time
os.environ.setdefault("PYTORCH_DEFAULT_NCHW", "1")
import numpy as np, torch
from c2net.context import prepare, upload_output
ctx = prepare(); out_dir = ctx.output_path

# Locate dataset + code AFTER prepare() mounted them. The env vars are hints; we
# auto-find under /tmp/dataset so the exact OpenI mount folder name doesn't matter.
def _find(name, root="/tmp/dataset"):
    # os.walk(followlinks=True): OpenI mounts datasets via symlinks, and glob's
    # ** pattern does NOT descend into symlinked directories.
    for dp, dn, fn in os.walk(root, followlinks=True):
        if name in fn:
            return os.path.join(dp, name)
    return None
XZ = os.environ.get("NNUE_XZ", "")
if not XZ or not os.path.exists(XZ):
    XZ = _find("gamepack.xz") or "/tmp/dataset/gamepack.xz"
CODE_DIR = os.environ.get("CODE_DIR", "")
if not CODE_DIR or not os.path.exists(os.path.join(CODE_DIR, "luminex_nnue_train.py")):
    _p = _find("luminex_nnue_train.py")
    CODE_DIR = os.path.dirname(_p) if _p else "/tmp/code"
sys.path.insert(0, CODE_DIR)
FEAT = os.environ.get("FEAT", "")
if not FEAT or not os.path.exists(FEAT):
    FEAT = _find("luminex-featurize") or os.path.join(CODE_DIR, "luminex-featurize")
print((f"XZ={XZ} ({os.path.getsize(XZ)/1e9:.2f} GB)" if os.path.exists(XZ)
       else f"!! XZ={XZ} NOT FOUND — is the dataset selected in the task?"), flush=True)
print(f"CODE_DIR={CODE_DIR}  FEAT={FEAT} (exists={os.path.exists(FEAT)})", flush=True)

from luminex_nnue_train import LNNUE, NUM_INPUTS, MAX_PIECES, save_nnue
L1   = int(os.environ.get("NNUE_L1", "512"))
EPOCHS = int(os.environ.get("NNUE_EPOCHS", "1"))
BS   = int(os.environ.get("NNUE_BS", "32768"))
LR   = float(os.environ.get("NNUE_LR", "1e-3"))
BUF  = int(os.environ.get("NNUE_BUF", "2000000"))   # CPU-RAM shuffle buffer (positions)
NTH  = int(os.environ.get("NNUE_FEAT_THREADS", "8"))
BUDGET = int(os.environ.get("NNUE_BUDGET_SEC", str(int(3.5*3600))))
REC  = 136
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"stream-train: {XZ} | L1={L1} epochs={EPOCHS} bs={BS} buf={BUF} feat_threads={NTH} device={device}", flush=True)
SCALE = 400.0
import glob
FRAME_GLOB = os.environ.get("NNUE_FRAMES", "")
FRAMES = sorted(glob.glob(FRAME_GLOB)) if FRAME_GLOB else []
if not FRAMES:
    FRAMES = [XZ]   # single-pack fallback
total_bytes = sum(os.path.getsize(f) for f in FRAMES)
est_pos = int(total_bytes / 1.076)
print(f"training on {len(FRAMES)} frame(s), {total_bytes/1e9:.2f} GB total -> ~{est_pos/1e9:.2f}B positions", flush=True)

# ---- streaming chunk source: xz -dc | featurize --stream ----
# Multi-frame: loop over frame_*.xz (multi-stage encode produces separate frames, no merge).
class Streamer:
    def __init__(self, xz_path):
        cmd = f"xz -dc {xz_path} | {FEAT} --stream --input /dev/stdin --threads {NTH}"
        self.p = subprocess.Popen(["bash","-c",cmd], stdout=subprocess.PIPE, bufsize=0)
        self.n = 0
    def chunk(self):
        want = BUF * REC
        data = bytearray()
        while len(data) < want:
            d = self.p.stdout.read(min(1<<20, want - len(data)))
            if not d: break
            data += d
        m = len(data) // REC
        if m == 0: return None
        a = np.frombuffer(bytes(data[:m*REC]), dtype=np.uint8).reshape(m, REC)
        w = a[:, :64].copy().view(np.int16).reshape(m, 32)
        b = a[:, 64:128].copy().view(np.int16).reshape(m, 32)
        s = a[:, 128:132].copy().view(np.float32).reshape(m)
        t = a[:, 132:136].copy().view(np.float32).reshape(m)
        idx = np.random.permutation(m)
        self.n += m
        return w[idx], b[idx], s[idx], t[idx], m

def train():
    model = LNNUE(L1=L1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps_per_epoch = max(1, est_pos // BS)
    # Anneal the cosine LR over the BUDGET (not the full 15B epoch), so the LR
    # fully decays by the time the budget stops us mid-epoch. Assumes
    # ~NNUE_TARGET_POSPS throughput; default is conservative (low) so the LR
    # anneals on time or slightly early (extra steps then run at min_lr = a fine
    # low-LR polish). Tune NNUE_TARGET_POSPS up after you see real throughput.
    if BUDGET:
        target_posps = float(os.environ.get("NNUE_TARGET_POSPS", "800000"))
        anneal_steps = max(1, int(BUDGET * target_posps / BS))
        print(f"  cosine anneals over budget: T_max={anneal_steps} steps (~{target_posps:.0f} pos/s assumed)", flush=True)
    else:
        anneal_steps = EPOCHS * steps_per_epoch
        print(f"  cosine T_max={anneal_steps} steps (full epoch)", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=anneal_steps)
    ckpt = "/tmp/stream_ckpt.pt"
    gstep = 0; t0 = time.time()
    for epoch in range(EPOCHS):
        if BUDGET and (time.time()-t0) >= BUDGET:
            print(f"  budget hit before epoch {epoch} — pausing", flush=True); break
        ep_pos = 0; wt = time.time(); ep_n = 0
        for fi, frame_path in enumerate(FRAMES):
            st = Streamer(frame_path)
            while True:
                c = st.chunk()
                if c is None: break
                w,b,s,t,m = c
                for i in range(0, m, BS):
                    lo, hi = i, min(i+BS, m)
                    wi = torch.from_numpy(w[lo:hi]).to(device).long()
                    bi = torch.from_numpy(b[lo:hi]).to(device).long()
                    si = torch.from_numpy(s[lo:hi]).to(device)
                    ti = torch.from_numpy(t[lo:hi]).to(device)
                    opt.zero_grad()
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        pred = model(wi, bi, si)
                        loss = ((torch.sigmoid(pred/SCALE) - torch.sigmoid(ti/SCALE))**2).mean()
                    loss.backward(); opt.step(); sched.step(); gstep += 1
                    ep_pos += (hi-lo)
                    if gstep % 25 == 0:
                        dt = time.time()-wt
                        print(f"  e{epoch} f{fi+1}/{len(FRAMES)} step {gstep} loss={loss.item():.5f} | {ep_pos/1e6:.0f}M pos ({ep_pos/max(dt,1)/1e6:.2f}M/s)", flush=True)
                    if BUDGET and time.time()-t0 >= BUDGET:
                        print(f"  >>> BUDGET hit at step {gstep} ({ep_pos/1e6:.0f}M pos) — saving final net & stopping", flush=True)
                        save_nnue(model, os.path.join(out_dir, "luminex_v2.nnue"))
                        torch.save({"epoch":epoch,"model":model.state_dict(),"opt":opt.state_dict(),"gstep":gstep}, ckpt)
                        try: st.p.stdout.close(); st.p.terminate()
                        except Exception: pass
                        return
            try: st.p.stdout.close(); st.p.wait()
            except: pass
            ep_n += st.n
            print(f"  [epoch {epoch} frame {fi+1}/{len(FRAMES)}: {st.n:,} pos]", flush=True)
        print(f"  [epoch {epoch} done: {ep_n:,} positions across {len(FRAMES)} frames]", flush=True)
        save_nnue(model, os.path.join(out_dir, f"luminex_v2_ep{epoch}.nnue"))
        torch.save({"epoch":epoch+1,"model":model.state_dict(),"opt":opt.state_dict(),"gstep":gstep}, ckpt)
    save_nnue(model, os.path.join(out_dir, "luminex_v2.nnue"))

train()

# Upload the FLOAT net (LNN1 format). int8 (LNI8) quantization is done OFFLINE
# on the CPU with quantize_i8.py — it's pure numpy (no GPU), faster locally, and
# frees every second of the GPU session for training.
print("uploading float net (luminex_v2.nnue)...", flush=True)
upload_output()
print("DONE — luminex_v2.nnue (float). Quantize to LNI8 locally: "
      "python quantize_i8.py luminex_v2.nnue luminex_v6_i8.nnue", flush=True)
