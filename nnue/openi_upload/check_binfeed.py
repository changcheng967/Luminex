"""Checks binfeed.cpp (resident_train.py's NNUE_BINPACKS) against binpack2lcr on one binpack and
measures how fast it decodes. Exit 0 = every check passed (last line BINFEED_PASS).

usage: python check_binfeed.py FILE.binpack [--work DIR] [--threads 1,2,4,8] [--keep] [--speed]
       (NNUE_PYTORCH = the nnue-pytorch checkout both are built against; NNUE_DEVICE=cpu
       runs the reservoir check without a GPU)

1. Exactness: binpack2lcr's records of the file (converted into --work) against everything the
   feeder yields from it: one pass of its stream plus the held-out set of its index pass. The
   same counts as binpack2lcr's DONE line (positions, marked skip, quiet/capture/check,
   won/drew/lost, search-tree samples with no result, chunks) and the same sha256 of the records
   sorted into byte order (so the same flags, result bits included); no held-out
   position in the stream and only held-out ones in the held-out set; and the run cells'
   material check (check_data.py) on both record sets.
2. The index cache: a second open reads <file>.bfx and gives the same counts and held-out set.
3. Determinism: two streams with one seed agree record for record; another seed orders the pass
   differently and still yields the same records.
4. The reservoir: BinpackFeeder with a small reservoir of S positions; the batches of each cycle
   of S / bs steps are exactly the stream's window of S positions that came in the cycle before.
5. Speed: the decode rate over the whole file at each thread count, and the stream's delivery
   rate in the trainer's 131,072-record pulls, against the ~9.8M positions/s the GPU takes at
   L1 768 (75 steps/s). --speed runs this part alone."""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

os.environ.setdefault("NNUE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import resident_train as RT  # noqa: E402

GPU_RATE = 75 * 131072   # positions/s the float phase takes at L1 768


def say(msg):
    print(msg, flush=True)


def sorted_sha(rec):
    """sha256 of the records sorted into byte order (sorted in place)."""
    RT.binfeed_lib().bf_sort(rec.ctypes.data, len(rec), os.cpu_count() or 8)
    return hashlib.sha256(memoryview(rec).cast("B")).hexdigest()


def cell_material_check(all_recs, block=2_000_000):
    """The run cells' check_data.py, line for line, on uint8 [n, 32] records (block by block:
    its int64 piece codes take 256 bytes a record)."""
    tally = {name: [0, 0] for name in ("piece", "rook", "queen")}
    for o in range(0, len(all_recs), block):
        recs = all_recs[o:o + block]
        cnt = recs[:, 27].astype(np.int64)
        nib = np.stack([recs[:, 8:24] & 15, recs[:, 8:24] >> 4], 2).reshape(-1, 32).astype(np.int64)
        valid = np.arange(32)[None, :] < cnt[:, None]
        val = np.array([1, 3, 3, 5, 9, 0] * 2)[nib] * np.where(nib < 6, 1, -1) * valid
        edge = val.sum(1)
        white = (recs[:, 26] & 1).astype(bool)
        tgt = recs[:, 24:26].copy().view(np.int16).ravel().astype(np.int64)
        wt = np.where(white, tgt, -tgt)
        flag = (recs[:, 26] >> 1) & 3
        for e, name in ((3, "piece"), (5, "rook"), (9, "queen")):
            m = (np.abs(edge) == e) & (flag == 0)
            v = wt[m] * np.sign(edge[m])
            tally[name][0] += int(m.sum())
            tally[name][1] += int((v >= 800).sum())
    return {k: (c, w / c if c else float("nan")) for k, (c, w) in tally.items()}


def fmt_shares(s):
    return ", ".join(f"{k} up {v:.1%} (of {c:,})" for k, (c, v) in s.items())


def binpack2lcr(path, work, threads):
    """binpack2lcr's records of the file (built next to this file if needed) and its DONE line."""
    exe = os.path.join(work, "binpack2lcr")
    inc = os.path.join(os.environ.get("NNUE_PYTORCH", os.path.join(HERE, "nnue-pytorch")), "data_loader", "cpp")
    if not os.path.exists(exe):
        subprocess.run(["g++", "-O2", "-march=native", "-std=c++20", "-pthread", "-I", inc, "-o", exe,
                        os.path.join(HERE, "binpack2lcr.cpp")], check=True)
    out = os.path.join(work, os.path.basename(path) + ".ref")
    log = out + ".log"
    have = sum(os.path.getsize(p) for p in glob.glob(out + ".*.lcr")) // RT.REC
    lines = open(log).read().strip().splitlines() if os.path.exists(log) else []
    if not lines or "DONE" not in lines[-1] or parse_done(lines[-1])["positions"] != have:   # convert again
        for p in glob.glob(out + ".*.lcr"):
            os.remove(p)
        t = time.time()
        with open(log, "w") as f:
            subprocess.run([exe, path, out, str(threads)], stderr=f, check=True)
        say(f"  binpack2lcr: converted in {time.time() - t:.1f} s ({threads} threads)")
    done = open(log).read().strip().splitlines()[-1]
    parts = sorted(glob.glob(out + ".*.lcr"))
    total = sum(os.path.getsize(p) for p in parts) // RT.REC
    rec = np.empty((total, RT.REC), dtype=np.uint8)
    o = 0
    for p in parts:
        a = np.fromfile(p, dtype=np.uint8).reshape(-1, RT.REC)
        rec[o:o + len(a)] = a
        o += len(a)
    return rec, done, parts + [log]


def parse_done(line):
    m = re.search(r"DONE (\d+) positions \(quiet (\d+), capture (\d+), check (\d+); side to move won (\d+), "
                  r"drew (\d+), lost (\d+)\) from (\d+) chunks.*; (\d+) marked skip", line)
    if not m:
        raise SystemExit(f"FAIL: cannot read binpack2lcr's DONE line: {line}")
    k = ("positions", "quiet", "capture", "check", "won", "drew", "lost", "chunks", "marked")
    d = dict(zip(k, map(int, m.groups())))
    t = re.search(r"; (\d+) search-tree samples \(ply 9999\) with no game result", line)   # only when any
    d["no_result"] = int(t.group(1)) if t else 0
    return d


def stream(bp, seed, count, threads, out=None):
    """The first `count` records of the stream with this seed (a fresh start)."""
    out = np.empty((count, RT.REC), dtype=np.uint8) if out is None else out
    bp.start(seed, threads, 4 * threads + 8)
    blk = 1 << 20
    for o in range(0, count, blk):
        k = min(blk, count - o)
        bp.next_into(out[o:o + k].ctypes.data, k)
    bp.lib.bf_stream_stop(bp.h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binpack")
    ap.add_argument("--work", default=None, help="directory for binpack2lcr's records [<binpack>.check]")
    ap.add_argument("--threads", default=None, help="thread counts to time [1,2,4,...,cores]")
    ap.add_argument("--keep", action="store_true", help="keep binpack2lcr's records and the index cache")
    ap.add_argument("--speed", action="store_true", help="only time the decoding and the stream")
    a = ap.parse_args()
    path = os.path.abspath(a.binpack)
    work = a.work or path + ".check"
    os.makedirs(work, exist_ok=True)
    cores = os.cpu_count() or 8
    tlist = [int(x) for x in a.threads.split(",")] if a.threads else sorted({1, 2, 4, cores // 2, cores} - {0})
    RT.DECODE_THREADS = cores
    thr = min(int(RT.VAL_FRAC * 2 ** 32), 2 ** 32 - 1)
    ok = True

    def check(cond, what):
        nonlocal ok
        ok &= bool(cond)
        say(f"  {'ok  ' if cond else 'FAIL'} {what}")

    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else ""
    say(f"== {os.path.basename(path)}: {os.path.getsize(path):,} bytes, sha256 {digest}; {cores} threads here; "
        f"device {RT.dev}")
    if a.speed:
        bp = RT.Binpacks([path], thr, False)
        speed(bp, bp.select(0), tlist, cores)
        bp.close()
        return
    say("== 1. exactness against binpack2lcr")
    ref, done, parts = binpack2lcr(path, work, cores)
    d = parse_done(done)
    say(f"  binpack2lcr: {done}")
    cache = path + ".bfx"
    if os.path.exists(cache):
        os.remove(cache)
    bp = RT.Binpacks([path], thr, False)
    n = bp.select(0)
    i = bp.info
    held = bp.val_records()
    check(n + len(held) == d["positions"] == len(ref),
          f"positions: feeder {n:,} training + {len(held):,} held out = {n + len(held):,}; "
          f"binpack2lcr {d['positions']:,} (records in its files {len(ref):,})")
    check(i["marked"] == d["marked"], f"marked skip (VALUE_NONE) left out: feeder {i['marked']:,}, binpack2lcr {d['marked']:,}")
    for k in ("quiet", "capture", "check", "won", "drew", "lost", "no_result", "chunks"):
        check(i[k] == d[k], f"{k}: feeder {i[k]:,}, binpack2lcr {d[k]:,}")
    t = time.time()
    rec = np.empty((n + len(held), RT.REC), dtype=np.uint8)
    stream(bp, 1, n, cores, rec[:n])
    t_pass = time.time() - t
    rec[n:] = held
    hs = rec[:n, 28:32].copy().view(np.uint32).ravel()
    hv = held[:, 28:32].copy().view(np.uint32).ravel()
    check(bool((hs >= thr).all()) and bool((hv < thr).all()),
          f"held out by hash (< {thr:,}): none in the stream's pass, all {len(held):,} in the held-out set")
    ms_ref, ms_new = cell_material_check(ref), cell_material_check(rec)
    check(ms_ref == ms_new and ms_new["queen"][1] >= 0.9,
          f"material check (quiet, labelled beyond +800): feeder {fmt_shares(ms_new)} | binpack2lcr {fmt_shares(ms_ref)}")
    t = time.time()
    h_ref, h_new = sorted_sha(ref), sorted_sha(rec)
    check(h_ref == h_new, f"sha256 of the sorted records: feeder {h_new}, binpack2lcr {h_ref} "
                          f"(one pass of the stream in {t_pass:.1f} s; sorted and hashed in {time.time() - t:.0f} s)")
    del ref

    say("== 2. index cache")
    t = time.time()
    bp1 = RT.Binpacks([path], thr, True, quiet=True)
    t1 = time.time() - t
    wrote = os.path.exists(cache)
    t = time.time()
    bp2 = RT.Binpacks([path], thr, True, quiet=True)
    t2 = time.time() - t
    c1, c2 = bp1.file_counts()[0], bp2.file_counts()[0]
    check(wrote and c2[3] == 1 and c1[:3] == c2[:3] == (n, len(held), i["marked"])
          and np.array_equal(bp2.val_records(), held),
          f"{os.path.basename(cache)} ({os.path.getsize(cache) if wrote else 0:,} bytes): written in {t1:.1f} s "
          f"(index pass), read in {t2:.1f} s, same counts and held-out set")
    bp1.close()
    bp2.close()
    if not a.keep and os.path.exists(cache):
        os.remove(cache)

    say("== 3. determinism")
    k = min(n, 3_000_000)
    s1, s2 = stream(bp, 1, k, cores), stream(bp, 1, k, max(1, cores // 2))
    check(np.array_equal(s1, s2), f"seed 1 twice ({cores} and {max(1, cores // 2)} threads): the first {k:,} records agree")
    o2 = stream(bp, 2, n, cores)
    check(not np.array_equal(o2[:k], s1), "seed 2 starts the pass with other records")
    h2 = sorted_sha(np.concatenate([o2, held]))
    check(h2 == h_ref, f"seed 2's pass holds the same records (sorted sha256 {h2})")
    del o2, s1, s2, rec

    say("== 4. the reservoir (BinpackFeeder)")
    bs = 65536
    RT.RESERVOIR, RT.BINPACK_CACHE = 4 * bs, False
    feeder = RT.BinpackFeeder([path], 3, bs)
    cyc, ncyc = feeder.cycle, 6
    got = [feeder.next()[0].cpu().numpy() for _ in range(cyc * ncyc)]
    ref_stream = stream(bp, 3, feeder.S * ncyc, cores)
    same = True
    for c in range(ncyc):
        drawn = np.concatenate(got[c * cyc:(c + 1) * cyc])
        window = ref_stream[c * feeder.S:(c + 1) * feeder.S].copy()
        same &= sorted_sha(drawn) == sorted_sha(window)
    check(same, f"S = {feeder.S:,} places, {cyc} batches of {bs:,} a cycle: each of {ncyc} cycles drew exactly "
                f"the stream's window of S positions before it")
    first = got[0]
    check(not np.array_equal(first, ref_stream[:bs]), "a batch is not the stream's order")
    del got, ref_stream

    speed(bp, n, tlist, cores)
    bp.close()
    if not a.keep:
        for p in parts:
            os.remove(p)
    say("BINFEED_PASS" if ok else "BINFEED_FAIL")
    sys.exit(0 if ok else 1)


def speed(bp, n, tlist, cores):
    say("== 5. speed")
    out = (bp.ct.c_int64 * 3)()
    for T in tlist:
        sec = bp.lib.bf_bench(bp.h, T, 0, out)
        e, us = out[0], out[1] + out[2]
        say(f"  decode, {T:2d} thread(s): {e:,} entries in {sec:.2f} s = {e / sec / 1e6:.2f}M/s "
            f"({e / sec / 1e6 / T:.2f}M a thread); usable {us / sec / 1e6:.2f}M/s ({us / sec / 1e6 / T:.2f}M a thread)")
    pull = 131072
    for T in sorted({max(1, cores // 4), max(1, cores // 2), cores}):
        buf = np.empty((pull, RT.REC), dtype=np.uint8)
        bp.start(5, T, 4 * T + 8)
        bp.next_into(buf.ctypes.data, pull)   # the decoders start
        cnt, t = min(n, 40_000_000), time.time()
        for _ in range(cnt // pull):
            bp.next_into(buf.ctypes.data, pull)
        sec = time.time() - t
        bp.lib.bf_stream_stop(bp.h)
        r = cnt // pull * pull / sec
        say(f"  stream, {T:2d} decode thread(s): {r / 1e6:.2f}M positions/s in pulls of {pull:,} "
            f"({r / GPU_RATE:.1f}x the GPU's {GPU_RATE / 1e6:.1f}M/s at 75 steps/s)")


if __name__ == "__main__":
    main()
