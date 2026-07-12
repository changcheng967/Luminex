#!/usr/bin/env python3
"""C500 featurize — fixed for the real environment.

Environment (discovered via df/mount/cgroup):
  /tmp/code    = JuiceFS (slow network FS), 50G, ~13G free  ← DON'T write big files here
  /            = overlay (fast local disk), 1.7T free       ← write .npy HERE
  memory.max   = 40 GB (cgroup)                              ← real RAM limit (host shows 323G)

Design:
  - Writes .npy to DATA_DIR on the overlay (fast, 1.7T), not /tmp/code (JuiceFS).
  - Reads the cgroup memory.max (40G), NOT /proc/meminfo (which shows the host's 323G).
  - Creates the Pool BEFORE opening the memmap → workers don't inherit the 41G mapping.
  - Flush + posix_fadvise(DONTNEED) every 1M rows → page cache stays ~150MB, never OOMs.
  - Fast parser, auto-validated vs python-chess (2000 pos) before the run.
  - Resumable: checkpoint every 1M rows; re-run skips to where it stopped.
"""
import os, sys, time, glob, shutil, subprocess, struct
from multiprocessing import Pool
import numpy as np

try:
    import chess
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'chess'])
    import chess

# .npy + meta go to /tmp/code (PERSISTENT across restarts). It's JuiceFS (slower than the
# overlay, but the only path that survives a restart). Override with NNUE_DATA_DIR if needed.
DATA_DIR = os.environ.get("NNUE_DATA_DIR", "/tmp/code")

KingBuckets = [-1,-1,-1,-1,31,30,29,28, -1,-1,-1,-1,27,26,25,24,
    -1,-1,-1,-1,23,22,21,20, -1,-1,-1,-1,19,18,17,16,
    -1,-1,-1,-1,15,14,13,12, -1,-1,-1,-1,11,10,9,8,
    -1,-1,-1,-1,7,6,5,4, -1,-1,-1,-1,3,2,1,0]
NUM_INPUTS = 768 * 32
MAX_PIECES = 32
BYTES_PER_POS = MAX_PIECES * 2 * 2 + 8   # 136

# (piece_color, piece_type): color 1=white 0=black; type 1..6 — SAME numbers python-chess uses.
PIECE_INFO = {
    'P':(1,1),'N':(1,2),'B':(1,3),'R':(1,4),'Q':(1,5),'K':(1,6),
    'p':(0,1),'n':(0,2),'b':(0,3),'r':(0,4),'q':(0,5),'k':(0,6)}

def _orient(is_white_pov, sq, ksq):
    kfile = ksq & 7
    return ((7 if kfile < 4 else 0) ^ (0 if is_white_pov else 56) ^ sq)

def _halfka_idx(is_white_pov, king_sq, sq, piece_color, piece_type):
    p_idx = (piece_type - 1) * 2 + (1 if piece_color != is_white_pov else 0)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 768

def featurize_position_fast(fen_eval):
    fen, ev = fen_eval
    try:
        parts = fen.split()
        placement = parts[0]
        stm_white = (len(parts) < 2 or parts[1] == 'w')
    except Exception:
        return None
    wk = bk = -1; pieces = []; rank, file = 7, 0
    for ch in placement:
        if ch == '/': rank -= 1; file = 0
        elif ch.isdigit(): file += int(ch)
        else:
            info = PIECE_INFO.get(ch)
            if info:
                pc, pt = info; sq = rank * 8 + file
                if pt == 6:
                    if pc == 1: wk = sq
                    else: bk = sq
                pieces.append((sq, pc, pt))
            file += 1
    if wk < 0 or bk < 0 or len(pieces) > MAX_PIECES: return None
    w, b = [], []
    for sq, pc, pt in pieces:
        w.append(_halfka_idx(True,  wk, sq, pc, pt))
        b.append(_halfka_idx(False, bk, sq, pc, pt))
    return w, b, (1.0 if stm_white else 0.0), ev

def featurize_position_chess(fen_eval):
    fen, ev = fen_eval
    try: board = chess.Board(fen)
    except Exception: return None
    wk = board.king(chess.WHITE); bk = board.king(chess.BLACK)
    if wk is None or bk is None: return None
    w, b = [], []
    for sq, p in board.piece_map().items():
        pc = 1 if p.color == chess.WHITE else 0; pt = int(p.piece_type)
        w.append(_halfka_idx(True,  wk, sq, pc, pt))
        b.append(_halfka_idx(False, bk, sq, pc, pt))
    if len(w) > MAX_PIECES: return None
    return w, b, (1.0 if board.turn == chess.WHITE else 0.0), ev

def validate(gz_path, n=2000):
    print(f"Validating fast parser vs python-chess on {n} positions...", flush=True)
    checked = mismatches = 0
    for fen_eval in fen_eval_stream_gz(gz_path, limit=n):
        ref = featurize_position_chess(fen_eval)
        if ref is None: continue
        fast = featurize_position_fast(fen_eval)
        if fast is None or tuple(sorted(fast[0])) != tuple(sorted(ref[0])) \
                       or tuple(sorted(fast[1])) != tuple(sorted(ref[1])) \
                       or fast[2] != ref[2]:
            mismatches += 1
            if mismatches <= 3: print(f"  MISMATCH: {fen_eval[0]}", flush=True)
        checked += 1
    print(f"  checked {checked} | mismatches {mismatches}", flush=True)
    if mismatches:
        print("VALIDATION FAILED. Aborting.", flush=True); sys.exit(1)
    print("VALIDATION PASSED — running at full speed.", flush=True)

def fen_eval_stream_gz(gz_path, limit=None, skip=0):
    proc = subprocess.Popen(['gunzip', '-c', gz_path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = proc.stdout; yielded = 0; nskip = 0; tail = b''
    try:
        while True:
            chunk = out.read(16 * 1024 * 1024)
            data = tail + chunk
            if not chunk: lines = data.split(b'\n'); tail = b''
            else: lines = data.split(b'\n'); tail = lines.pop()
            for raw in lines:
                if b'|' not in raw: continue
                fen, _, ev = raw.strip().rpartition(b'|')
                try: ev = int(ev)
                except ValueError: continue
                if nskip < skip: nskip += 1; continue
                yield (fen.decode(), ev); yielded += 1
                if limit and yielded >= limit: return
            if not chunk: break
    finally:
        out.close(); proc.wait()

def cgroup_mem_limit():
    for p in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        try:
            with open(p) as f: v = int(f.read().strip())
            if 0 < v < (1 << 50): return v
        except (FileNotFoundError, ValueError): pass
    return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    N_WORKERS = int(os.environ.get("N_WORKERS", "10"))
    USE_FAST = os.environ.get("FEATURIZE_PARSER", "fast") == "fast"
    featurize = featurize_position_fast if USE_FAST else featurize_position_chess
    CKPT = os.path.join(DATA_DIR, "featurize_ckpt.txt")

    gz_path = None
    for d in [os.environ.get("C2NET_DATASET", "/tmp/dataset"), "/tmp/dataset", "/tmp/code", "/tmp"]:
        for name in ["sf_dataset_big.txt.gz", "sf_dataset.txt.gz"]:
            m = glob.glob(os.path.join(d, "**", name), recursive=True)
            if m: gz_path = m[0]; break
        if gz_path: break
    assert gz_path, "sf_dataset_big.txt.gz not found"
    print(f"Data: {gz_path}", flush=True)
    print(f"Parser: {'fast (validated)' if USE_FAST else 'python-chess'} | DATA_DIR: {DATA_DIR}", flush=True)
    cg = cgroup_mem_limit()
    print(f"cgroup memory.max: {cg/1e9:.0f} GB" if cg else "cgroup limit: not found (using disk cap)", flush=True)

    # resume check
    idx_start = 0; skipped_start = 0; resume = False; N = None
    if os.path.exists(CKPT) and os.path.exists(os.path.join(DATA_DIR, "feat_w.npy")):
        try:
            with open(CKPT) as f:
                idx_start = int(f.readline()); skipped_start = int(f.readline()); N = int(f.readline())
            existing = np.load(os.path.join(DATA_DIR, "feat_w.npy"), mmap_mode='r')
            if existing.shape == (N, MAX_PIECES) and idx_start > 0:
                resume = True
                print(f"RESUME from {idx_start:,} (of {N:,}).", flush=True)
            else:
                print("Checkpoint stale — starting fresh.", flush=True)
                resume = False; idx_start = skipped_start = 0; N = None
        except Exception as e:
            print(f"Checkpoint read failed ({e}) — fresh.", flush=True)
            resume = False; idx_start = skipped_start = 0; N = None

    if not resume:
        if USE_FAST: validate(gz_path, n=2000)
        du = shutil.disk_usage(DATA_DIR)
        disk_cap = int(du.free * 0.85 / BYTES_PER_POS)
        N = min(disk_cap, 300_000_000)
        env_cap = os.environ.get("NNUE_MAX_POS")
        if env_cap: N = min(N, int(env_cap))
        print(f"disk {du.free/1e9:.0f} GB free on {DATA_DIR} | target {N:,} positions "
              f"(~{N*BYTES_PER_POS/1e9:.0f} GB; RAM kept low via fadvise)", flush=True)
        for stale in ["feat_w.npy", "feat_b.npy", "feat_s.npy", "feat_t.npy"]:
            p = os.path.join(DATA_DIR, stale)
            if os.path.exists(p): os.remove(p)
        mode = 'w+'
    else:
        mode = 'r+'

    # Create the Pool FIRST (clean parent) — workers must not inherit the 41G memmap mapping.
    print(f"Starting Pool ({N_WORKERS} workers) BEFORE opening memmaps...", flush=True)
    pool = Pool(N_WORKERS)

    # Open memmaps on the overlay (main process only).
    print(f"Opening arrays (mode={mode}, {N:,} rows) on overlay...", flush=True)
    w_idx  = np.lib.format.open_memmap(f"{DATA_DIR}/feat_w.npy", mode=mode, dtype=np.int16,   shape=(N, MAX_PIECES))
    b_idx  = np.lib.format.open_memmap(f"{DATA_DIR}/feat_b.npy", mode=mode, dtype=np.int16,   shape=(N, MAX_PIECES))
    stm    = np.lib.format.open_memmap(f"{DATA_DIR}/feat_s.npy", mode=mode, dtype=np.float32, shape=(N,))
    target = np.lib.format.open_memmap(f"{DATA_DIR}/feat_t.npy", mode=mode, dtype=np.float32, shape=(N,))

    fadvise_fds = []
    for n in ["feat_w.npy", "feat_b.npy", "feat_s.npy", "feat_t.npy"]:
        try: fadvise_fds.append(os.open(os.path.join(DATA_DIR, n), os.O_RDONLY))
        except OSError: pass
    def drop_cache():
        for fd in fadvise_fds:
            try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError): pass

    print(f"Featurizing (skip={idx_start:,}, batched writes of 50K, flush+fadvise every 1M)...", flush=True)
    t0 = time.time(); idx = idx_start; skipped = skipped_start
    remaining = N - idx_start
    # Buffer rows in RAM, write big batches → few JuiceFS FUSE calls (was 4838 pos/s
    # writing one row at a time; batching cuts calls from 300M to ~6000).
    BUF = 50000
    buf_w = np.full((BUF, MAX_PIECES), NUM_INPUTS, dtype=np.int16)
    buf_b = np.full((BUF, MAX_PIECES), NUM_INPUTS, dtype=np.int16)
    buf_stm = np.zeros(BUF, dtype=np.float32)
    buf_tgt = np.zeros(BUF, dtype=np.float32)
    buf_n = 0
    next_ckpt = ((idx_start // 1_000_000) + 1) * 1_000_000
    try:
        def flush_buffer():
            nonlocal idx, buf_n
            if buf_n == 0: return
            cnt = min(buf_n, N - idx)
            if cnt <= 0: return
            w_idx[idx:idx+cnt] = buf_w[:cnt]      # ONE big write per array (not per row)
            b_idx[idx:idx+cnt] = buf_b[:cnt]
            stm[idx:idx+cnt] = buf_stm[:cnt]
            target[idx:idx+cnt] = buf_tgt[:cnt]
            idx += cnt
            buf_n = 0
            buf_w[:] = NUM_INPUTS                  # reset padding for reuse (avoid stale features)
            buf_b[:] = NUM_INPUTS
        for result in pool.imap(featurize, fen_eval_stream_gz(gz_path, limit=remaining, skip=idx_start), chunksize=4096):
            if result is None: skipped += 1; continue
            if idx + buf_n >= N: break
            w, b, s, e = result
            buf_w[buf_n, :len(w)] = w
            buf_b[buf_n, :len(b)] = b
            buf_stm[buf_n] = s
            buf_tgt[buf_n] = e
            buf_n += 1
            if buf_n == BUF:
                flush_buffer()
                if idx >= next_ckpt:
                    w_idx.flush(); b_idx.flush(); stm.flush(); target.flush()
                    drop_cache()
                    with open(CKPT, "w") as f: f.write(f"{idx}\n{skipped}\n{N}\n")
                    el = time.time() - t0; rate = (idx - idx_start) / el
                    print(f"  {idx:,}/{N:,} ({rate:.0f} pos/s, ETA {(N-idx)/rate:.0f}s, skipped {skipped}) [ckpt]", flush=True)
                    next_ckpt = ((idx // 1_000_000) + 1) * 1_000_000
        flush_buffer()   # write any remainder
    finally:
        pool.close(); pool.join()

    w_idx.flush(); b_idx.flush(); stm.flush(); target.flush(); drop_cache()
    for fd in fadvise_fds:
        try: os.close(fd)
        except OSError: pass
    del w_idx, b_idx, stm, target
    with open(os.path.join(DATA_DIR, "featurized_meta.txt"), "w") as f:
        f.write(f"{idx}\n{N}\n{skipped}\n")
    if os.path.exists(CKPT): os.remove(CKPT)
    el = time.time() - t0
    print(f"\nDONE: {idx:,} valid, {skipped} skipped, {el:.0f}s ({(idx-idx_start)/el:.0f} pos/s)", flush=True)
    print(f"  {DATA_DIR}/feat_w.npy + feat_b.npy + feat_s.npy + feat_t.npy", flush=True)
    print(f"  Train with: NNUE_NPY_DIR={DATA_DIR} python c500_train.py", flush=True)

if __name__ == '__main__':
    main()
