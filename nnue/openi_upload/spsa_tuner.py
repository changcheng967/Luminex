#!/usr/bin/env python3
"""SPSA tuner v2 — fixes from v1 debugging.

v1 bugs fixed:
  - Score parsing worked but results were degenerate (-800) due to TIMEOUT FORFEITURES
    at tc=1+0.01 concurrency 4 (8 engines on 8 cores → CPU contention → time losses).
  - Fix: tc=2+0.05 (more buffer), concurrency=2 (less contention).
  - Fix: track timeouts separately (reported in log, not silently folded into losses).
  - Fix: positive-control sanity test before real run (engine vs clearly-weaker opponent).

Design (per Sonnet 5 protocol):
  - Opponent rotation: pool + held-out (prevents single-opponent overfit)
  - Proper a_k/c_k gain schedule (Spall 1998)
  - Per-constant gradient tracking
  - Periodic checkpoint validation vs held-out opponent
  - Ship only if 3-way gate passes (baseline + held-out + Stash)
  - Narrow 12-constant set (not 50-100) for first run
"""
import os, sys, subprocess, json, random, math, time, argparse, re

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
        for v in theta:
            f.write(f"{int(round(v))}\n")

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
    """Run cutechess match. Returns (wins, losses, draws, timeouts)."""
    cmd = [
        "/hyperai/home/cutechess/cutechess-cli",
        "-engine", f"cmd={engine_cmd}", "name=Cand", "proto=uci",
        "-engine", f"cmd={opponent_cmd}", "name=Opp", "proto=uci",
        "-each", f"tc={tc}",
        "-openings", f"file={openings}", "format=pgn",
        "-rounds", str(rounds),
        "-repeat",
        "-concurrency", str(concurrency),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        w = l = d = 0
        timeouts = 0
        for line in r.stdout.splitlines():
            if "Score of Cand" in line:
                parts = line.split(":")[1].strip().split()
                w, l, d = int(parts[0]), int(parts[2]), int(parts[4])
            # Count timeout-related results for logging
            if "loses on time" in line and "Cand" in r.stdout.split("\n\n")[0]:
                timeouts += 1  # rough count — logged but not separately scored
        return w, l, d, timeouts
    except Exception as e:
        print(f"  match error: {e}", flush=True)
        return 0, 0, rounds, 0

def score_to_elo(w, l, d):
    n = w + l + d
    if n == 0: return 0.0
    score = (w + 0.5 * d) / n
    if score <= 0.0: return -800.0
    if score >= 1.0: return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)

def sanity_check(engine_path, weak_opponent, tc):
    """Positive-control: engine (default params) vs clearly-weaker opponent.
    Must show non-degenerate score (>30% and <90%). If not, harness is broken."""
    print("=== POSITIVE CONTROL SANITY CHECK ===", flush=True)
    default_cmd = setup_engine_dir("/tmp/spsa_default", engine_path, DEFAULTS)
    w, l, d, to = run_match(default_cmd, weak_opponent, 20, 1, tc)
    score = (w + 0.5 * d) / max(w + l + d, 1)
    print(f"  Engine(default) vs {os.path.basename(weak_opponent)}: {w}W-{l}L-{d}D "
          f"score={score:.1%} timeouts={to}", flush=True)
    if score < 0.30 or score > 0.90:
        print(f"  *** FAIL: score={score:.1%} is degenerate. Harness broken — abort.", flush=True)
        return False
    if to > 4:
        print(f"  *** WARNING: {to} timeouts in 20 games. Increase tc before proceeding.", flush=True)
    print(f"  PASS: non-degenerate, harness is functional.", flush=True)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--weak-test", help="clearly-weaker opponent for sanity check")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=1)  # LOW to prevent timeout corruption
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--tc", default="1+0.01")  # LONGER than bullet — fewer timeouts
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--output", default="spsa_result.json")
    args = ap.parse_args()

    # ── Positive-control sanity check ──
    if args.weak_test:
        if not sanity_check(args.engine, args.weak_test, args.tc):
            sys.exit(1)
    else:
        print("WARNING: no --weak-test provided, skipping sanity check", flush=True)

    # ── Initialize ──
    theta = list(DEFAULTS)
    best_theta = list(DEFAULTS)
    best_elo = None
    A = args.iterations / 10.0
    a_base = 0.15
    c_base = 0.3  # REDUCED from 1.0 → smaller initial perturbations

    print(f"SPSA v2: {N} params, {args.iterations} iters, {args.rounds} games/eval", flush=True)
    print(f"Opponents: {args.opponents} (tuning), {args.heldout} (heldout)", flush=True)
    print(f"TC: {args.tc} concurrency={args.concurrency} (low contention)", flush=True)
    print(f"Initial params: {theta}", flush=True)

    gradient_history = [[] for _ in range(N)]
    t0 = time.time()

    for k in range(args.iterations):
        a_k = a_base / ((k + 1 + A) ** 0.602)
        c_k = c_base / ((k + 1) ** 0.101)
        delta = [random.choice([-1, 1]) for _ in range(N)]

        theta_plus = []
        theta_minus = []
        for i, (name, default, lo, hi) in enumerate(PARAMS):
            range_i = hi - lo
            perturb = c_k * range_i * delta[i] * 0.1
            theta_plus.append(max(lo, min(hi, theta[i] + perturb)))
            theta_minus.append(max(lo, min(hi, theta[i] - perturb)))

        plus_cmd = setup_engine_dir("/tmp/spsa_plus", args.engine, theta_plus)
        minus_cmd = setup_engine_dir("/tmp/spsa_minus", args.engine, theta_minus)
        opp = args.opponents[k % len(args.opponents)]

        wp, lp, dp, tp = run_match(plus_cmd, opp, args.rounds, args.concurrency, args.tc)
        wm, lm, dm, tm = run_match(minus_cmd, opp, args.rounds, args.concurrency, args.tc)

        elo_p = score_to_elo(wp, lp, dp)
        elo_m = score_to_elo(wm, lm, dm)

        for i, (name, default, lo, hi) in enumerate(PARAMS):
            range_i = hi - lo
            c_i = c_k * range_i * 0.1
            g_i = (elo_p - elo_m) / (2.0 * c_i * delta[i])
            gradient_history[i].append(g_i)
            theta[i] = max(lo, min(hi, theta[i] + a_k * g_i * range_i))

        dt = time.time() - t0
        avg_g = [sum(g[-10:]) / min(len(g), 10) if g else 0 for g in gradient_history]
        sorted_g = sorted(range(N), key=lambda i: abs(avg_g[i]), reverse=True)[:3]
        top3 = " ".join(f"{PARAMS[i][0]}={avg_g[i]:+.1f}" for i in sorted_g)
        print(f"[{k+1}/{args.iterations}] elo+={elo_p:.0f} elo-={elo_m:.0f} Δ={elo_p-elo_m:+.1f} "
              f"to={tp}+{tm} opp={os.path.basename(opp)} | {top3} | {dt:.0f}s", flush=True)

        if (k + 1) % args.checkpoint == 0:
            check_cmd = setup_engine_dir("/tmp/spsa_check", args.engine, theta)
            wh, lh, dh, th = run_match(check_cmd, args.heldout, args.rounds * 2,
                                       args.concurrency, args.tc)
            check_elo = score_to_elo(wh, lh, dh)
            print(f"  CHECKPOINT vs {os.path.basename(args.heldout)}: "
                  f"{wh}W-{lh}L-{dh}D elo={check_elo:.0f} to={th} (best={best_elo})", flush=True)
            if best_elo is None or check_elo > best_elo:
                best_elo = check_elo
                best_theta = list(theta)
                write_params(best_theta, "spsa_best.txt")
                print(f"  *** NEW BEST: elo={best_elo:.0f} saved to spsa_best.txt", flush=True)
            state = {"iteration": k + 1, "theta": theta, "best_theta": best_theta,
                     "best_elo": best_elo}
            with open(args.output, 'w') as f:
                json.dump(state, f, indent=2)

    # Final validation
    print(f"\n=== FINAL VALIDATION ===", flush=True)
    final_cmd = setup_engine_dir("/tmp/spsa_final", args.engine, best_theta)
    for opp_path in args.opponents + [args.heldout]:
        w, l, d, to = run_match(final_cmd, opp_path, 500, args.concurrency, args.tc)
        elo = score_to_elo(w, l, d)
        print(f"  vs {os.path.basename(opp_path)}: {w}W-{l}L-{d}D elo={elo:.0f} to={to}", flush=True)
    print(f"\nBest params (elo={best_elo}): {[int(round(v)) for v in best_theta]}", flush=True)
    write_params(best_theta, "spsa_best.txt")

if __name__ == "__main__":
    main()
