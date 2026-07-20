#!/usr/bin/env python3
"""Optimized multi-stage encode (NO merge) — maximize positions + speed on a 20 GB disk.

Innovations (tested on the SSH):
  * parallel download (6 concurrent ~17 MB/s; saturates the box's bandwidth)
  * multi-thread encode (8 cores) + xz -T0 -6 (0.77 B/pos; -9 gave no improvement)
  * frame.xz at 0.77 B/pos => up to ~26 B positions could fit 20 GB (download/shutdown limits first)
  * ATOMIC + VERIFIED frames: encode|xz -> frame_K.xz.tmp; xz -t integrity check; rename only if OK.
    A shutdown mid-encode leaves a .tmp that's discarded on next start -> no corrupt/wasted data.
  * incremental per-date enumeration (no giant upfront listing that trips the HF 429 limit)

Run:  nohup python3 multistage_encode.py > /hyperai/home/ms.out 2>&1 &
"""
import os, sys, subprocess as sp, time, glob, json, shutil, urllib.request, urllib.error
import concurrent.futures, threading

HOME      = "/hyperai/home"
PGN_DIR   = HOME + "/pgns"
FRAME_DIR = HOME + "/frames"
ENCODE    = HOME + "/luminex-encode"
os.makedirs(PGN_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

API   = "https://huggingface.co/api/datasets/official-stockfish/fishtest_pgns/tree/main"
BASE  = "https://huggingface.co/datasets/official-stockfish/fishtest_pgns"
DL_CONCURRENCY = 6        # parallel downloads (saturates ~17 MB/s)
BATCH_GB       = 1.5      # PGN GB per encode batch (~200M positions)
MIN_FREE_GB    = 3.0      # stop when free disk < this (room for one batch + encode)
QUOTA_GB       = 20.0

hdr = {}
tok_path = HOME + "/huggingface/token"
if os.path.exists(tok_path):
    hdr["Authorization"] = "Bearer " + open(tok_path).read().strip()

logf = open(HOME + "/multistage.log", "a")
def P(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

def free_gb():
    total = 0
    for root, _, files in os.walk(HOME):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    for f in glob.glob("/tmp/_enc_*"):
        try: total += os.path.getsize(f)
        except OSError: pass
    return QUOTA_GB - total / 1e9

def hf_get(path, retries=6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{API}/{path}?recursive=true")
            for k, v in hdr.items(): req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=40) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1: time.sleep(15 * (attempt + 1)); continue
            return []
        except Exception:
            if attempt < retries - 1: time.sleep(5); continue
            return []

def download_one(fp):
    dst = PGN_DIR + "/" + fp.replace("/", "_")
    url = f"{BASE}/resolve/main/{fp}?download=true"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url)
            for k, v in hdr.items(): req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as o:
                shutil.copyfileobj(r, o)
            return dst
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3: time.sleep(15 * (attempt + 1)); continue
            P(f"  dl fail {fp}: HTTP {e.code}"); return None
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            P(f"  dl fail {fp}: {e}"); return None

# ---- recover any .tmp left by a previous interrupted run (verify + keep or discard) ----
for tmp in glob.glob(FRAME_DIR + "/frame_*.xz.tmp"):
    if sp.run(["xz", "-t", tmp]).returncode == 0:
        os.rename(tmp, tmp[:-4]); P(f"recovered {os.path.basename(tmp[:-4])}")
    else:
        os.remove(tmp); P(f"discarded incomplete {os.path.basename(tmp)}")

P(f"=== multistage start {time.strftime('%H:%M:%S')} | auth={'yes' if hdr else 'NO'} | dl_x{DL_CONCURRENCY} ===")

# incremental file iterator
dates = [e["path"] for e in hf_get("") if e.get("type") == "directory"]
P(f"{len(dates)} dates available")
date_idx, date_files = 0, []
flock = threading.Lock()
def next_files(n):
    global date_idx, date_files
    with flock:
        out = []
        while len(out) < n:
            if date_files:
                out.append(date_files.pop(0))
            elif date_idx < len(dates):
                d = dates[date_idx]; date_idx += 1
                date_files = [e["path"] for e in hf_get(d) if e["path"].endswith(".pgn.gz")]
            else:
                break
        return out

frame_idx = len(glob.glob(FRAME_DIR + "/frame_*.xz"))
total_pos, files_done = 0, 0
while True:
    if free_gb() < MIN_FREE_GB:
        P(f"disk low ({free_gb():.1f} GB < {MIN_FREE_GB}) -> stopping"); break
    for f in glob.glob(PGN_DIR + "/*"): os.remove(f)
    # gather a batch: download in chunks of DL_CONCURRENCY, stop at BATCH_GB or low disk
    dsts, bgb = [], 0.0
    while bgb < BATCH_GB:
        if free_gb() < MIN_FREE_GB: break
        chunk = next_files(DL_CONCURRENCY)
        if not chunk: break
        with concurrent.futures.ThreadPoolExecutor(DL_CONCURRENCY) as ex:
            got = list(ex.map(download_one, chunk))
        for d in got:
            if d:
                dsts.append(d); bgb += os.path.getsize(d) / 1e9; files_done += 1
    if not dsts:
        P("no more files -> stopping"); break
    P(f"batch {frame_idx}: {len(dsts)} files ({files_done} total), {bgb:.2f} GB PGN, free={free_gb():.1f}")
    fl = PGN_DIR + "/_list.txt"
    open(fl, "w").write("\n".join(dsts) + "\n")
    frame_final = f"{FRAME_DIR}/frame_{frame_idx:04d}.xz"
    frame_tmp   = frame_final + ".tmp"
    # encode | xz -> .tmp
    enc = sp.Popen([ENCODE, "--threads", "8", "--filelist", fl], stdout=sp.PIPE, stderr=sp.PIPE)
    xz  = sp.Popen(["xz", "-T0", "-6"], stdin=enc.stdout, stdout=open(frame_tmp, "wb"))
    enc.stdout.close()
    enc_err = enc.stderr.read().decode(errors="replace")
    rc, enc_rc = xz.wait(), enc.wait()
    for line in enc_err.split("\n"):
        if "pos=" in line:
            try: total_pos += int(line.split("pos=")[1].split()[0])
            except (ValueError, IndexError): pass
    if rc != 0 or enc_rc != 0 or not os.path.exists(frame_tmp):
        P(f"  ENCODE FAILED (xz={rc} enc={enc_rc}): {enc_err.strip()[-300:]}")
        try: os.remove(frame_tmp)
        except OSError: pass
    elif sp.run(["xz", "-t", frame_tmp]).returncode != 0:
        P(f"  VERIFY FAILED (xz -t) -> discarding {frame_tmp}"); os.remove(frame_tmp)
    else:
        os.rename(frame_tmp, frame_final)
        frame_idx += 1
        P(f"  OK: {enc_err.strip()[-160:]}")
        for b in [l for l in enc_err.split("\n") if "BADSAN" in l][:5]:
            P(f"  {b.strip()[:140]}")
    for f in glob.glob(PGN_DIR + "/*"): os.remove(f)
    P(f"  total ~{total_pos/1e9:.2f}B pos, {frame_idx} verified frames, free={free_gb():.1f} GB")

P(f"DONE: {frame_idx} verified frames, ~{total_pos/1e9:.2f}B positions in {FRAME_DIR}")
