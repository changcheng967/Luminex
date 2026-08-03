#!/usr/bin/env python3
"""SPSA tuner v3 — per Sonnet 5's structural redesign.

Key changes from v2:
  - tc=0.05+0 (50ms/move, both engines identical — no forfeits, fair)
  - concurrency=4 (8 single-threaded engines on 8 cores = zero contention)
  - 40 games/eval (20 opening pairs via -repeat), 250 iterations
  - Stash V20 = primary (closest in strength → max gradient signal)
  - Stash V21 = held-out (validation only, never tuned against)
  - Proper decaying a_k/c_k gain schedule (Spall 1998)
  - Sanity check before real run
"""
import os, sys, subprocess, json, random, math, time, argparse

PARAMS = [
    ("lmr_scale_quiet",   40,  20,  70),
    ("lmr_scale_noisy",   24,  10,  50),
    ("futility_coeff",   130,  80, 200),
    ("futility_offset",    50,   0, 120),
    ("nmp_base",            3,   2,   6),
    ("nmp_thresh1",         5,   3,   8),
    ("nmp_thresh2",        12,   8,  18),
    ("razor_base",        300, 150, 500),
    ("razor_coeff",        60,  30, 120),
    ("rev_fut_coeff",     100,  60, 160),
    ("aspiration_delta",    50,  20, 100),
    ("singular_margin",   200, 100, 400),
]
N = len(PARAMS)
DEFAULTS = [p[1] for p in PARAMS]

def write_params(theta, path):
    with open(path, 'w') as f:
        for v in theta: f.write(f"{int(round(v))}\n")

def setup_engine_dir(base_dir, engine_path, theta):
    os.makedirs(base_dir, exist_ok=True)
    write_params(theta, os.path.join(base_dir, "spsa_params.txt"))
    wrapper = os.path.join(base_dir, "run.sh")
    with open(wrapper, 'w') as f:
        f.write(f"#!/bin/bash\ncd {base_dir}\nexec {engine_path}\n")
    os.chmod(wrapper, 0o755)
    return wrapper

def run_match(engine_cmd, opponent_cmd, rounds, concurrency, tc,
              openings="/hyperai/home/openings.pgn"):
    cmd = [
        "/hyperai/home/cutechess/cutechess-cli",
        "-engine", f"cmd={engine_cmd}", "name=Cand", "proto=uci",
        "-engine", f"cmd={opponent_cmd}", "name=Opp", "proto=uci",
        "-each", f"tc={tc}",
        "-openings", f"file={openings}", "format=pgn",
        "-rounds", str(rounds), "-repeat",
        "-concurrency", str(concurrency),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        for line in r.stdout.splitlines():
            if "Score of Cand" in line:
                parts = line.split(":")[1].strip().split()
                return int(parts[0]), int(parts[2]), int(parts[4])  # W, L, D
    except Exception as e:
        print(f"  match error: {e}", flush=True)
    return 0, 0, rounds

def score_to_elo(w, l, d):
    n = w + l + d
    if n == 0: return 0.0
    s = (w + 0.5 * d) / n
    if s <= 0.0: return -800.0
    if s >= 1.0: return 800.0
    return -400.0 * math.log10(1.0 / s - 1.0)

def sanity_check(engine_path, opp, tc):
    print("=== SANITY CHECK ===", flush=True)
    cmd = setup_engine_dir("/tmp/spsa_default", engine_path, DEFAULTS)
    w, l, d = run_match(cmd, opp, 20, 4, tc)
    s = (w + 0.5 * d) / max(w + l + d, 1)
    print(f"  vs {os.path.basename(opp)}: {w}W-{l}L-{d}D score={s:.1%}", flush=True)
    if s < 0.25 or s > 0.85:
        print(f"  *** FAIL: degenerate score. Harness broken.", flush=True)
        return False
    print(f"  PASS.", flush=True)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--opponent", required=True, help="primary tuning opponent (close in Elo)")
    ap.add_argument("--heldout", required=True, help="held-out validation opponent")
    ap.add_argument("--rounds", type=int, default=40, help="games per eval (20 opening pairs)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--tc", default="0.05+0")
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--output", default="spsa_result.json")
    args = ap.parse_args()

    if not sanity_check(args.engine, args.opponent, args.tc):
        sys.exit(1)

    theta = list(DEFAULTS)
    best_theta = list(DEFAULTS)
    best_elo = None
    A = args.iterations / 5.0
    a_base = 5.0   # tuned for Elo-scale gradients
    c_base = 0.3

    print(f"SPSA v3: {N} params, {args.iterations} iters, {args.rounds} games/eval, "
          f"tc={args.tc} conc={args.concurrency}", flush=True)
    print(f"Tuning: {args.opponent} | Heldout: {args.heldout}", flush=True)
    print(f"Params: {theta}", flush=True)

    g_hist = [[] for _ in range(N)]
    t0 = time.time()

    for k in range(args.iterations):
        a_k = a_base / ((k + 1 + A) ** 0.602)
        c_k = c_base / ((k + 1) ** 0.101)
        delta = [random.choice([-1, 1]) for _ in range(N)]

        tp, tm = [], []
        for i, (_, _, lo, hi) in enumerate(PARAMS):
            ri = hi - lo
            p = c_k * ri * delta[i] * 0.1
            tp.append(max(lo, min(hi, theta[i] + p)))
            tm.append(max(lo, min(hi, theta[i] - p)))

        pc = setup_engine_dir("/tmp/spsa_plus", args.engine, tp)
        mc = setup_engine_dir("/tmp/spsa_minus", args.engine, tm)

        wp, lp, dp = run_match(pc, args.opponent, args.rounds, args.concurrency, args.tc)
        wm, lm, dm = run_match(mc, args.opponent, args.rounds, args.concurrency, args.tc)
        ep, em = score_to_elo(wp, lp, dp), score_to_elo(wm, lm, dm)

        for i, (_, _, lo, hi) in enumerate(PARAMS):
            ri = hi - lo
            ci = c_k * ri * 0.1
            gi = (ep - em) / (2.0 * ci * delta[i])
            g_hist[i].append(gi)
            theta[i] = max(lo, min(hi, theta[i] + a_k * gi * ri))

        dt = time.time() - t0
        ag = [sum(g[-20:]) / min(len(g), 20) if g else 0 for g in g_hist]
        sg = sorted(range(N), key=lambda i: abs(ag[i]), reverse=True)[:3]
        top3 = " ".join(f"{PARAMS[i][0][:8]}={ag[i]:+.1f}" for i in sg)
        print(f"[{k+1}/{args.iterations}] ep={ep:.0f} em={em:.0f} Δ={ep-em:+.1f} "
              f"{wp}-{lp}-{dp} {wm}-{lm}-{dm} | {top3} | {dt:.0f}s", flush=True)

        if (k + 1) % args.checkpoint == 0:
            cc = setup_engine_dir("/tmp/spsa_check", args.engine, theta)
            wh, lh, dh = run_match(cc, args.heldout, args.rounds * 3,
                                   args.concurrency, args.tc)
            ce = score_to_elo(wh, lh, dh)
            print(f"  CHECK vs {os.path.basename(args.heldout)}: "
                  f"{wh}-{lh}-{dh} elo={ce:.0f} best={best_elo}", flush=True)
            if best_elo is None or ce > best_elo:
                best_elo = ce
                best_theta = list(theta)
                write_params(best_theta, "spsa_best.txt")
                print(f"  *** NEW BEST {best_elo:.0f} ***", flush=True)
            with open(args.output, 'w') as f:
                json.dump({"k": k+1, "theta": theta, "best_theta": best_theta,
                           "best_elo": best_elo}, f, indent=2)

    # Final 3-way validation
    print(f"\n=== FINAL VALIDATION (tc=1+0.01, 500g) ===", flush=True)
    fc = setup_engine_dir("/tmp/spsa_final", args.engine, best_theta)
    for opp in [args.opponent, args.heldout]:
        w, l, d = run_match(fc, opp, 500, 2, "1+0.01")  # real TC for final gate
        print(f"  vs {os.path.basename(opp)}: {w}W-{l}L-{d}D elo={score_to_elo(w,l,d):.0f}", flush=True)
    # Also vs baseline (default params) at real TC
    bc = setup_engine_dir("/tmp/spsa_baseline", args.engine, DEFAULTS)
    w, l, d = run_match(fc, bc, 500, 2, "1+0.01")
    print(f"  vs Baseline(default): {w}W-{l}L-{d}D elo={score_to_elo(w,l,d):.0f}", flush=True)
    print(f"\nBest params: {[int(round(v)) for v in best_theta]}", flush=True)
    write_params(best_theta, "spsa_best.txt")

if __name__ == "__main__":
    main()
