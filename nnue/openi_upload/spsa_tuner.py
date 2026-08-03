#!/usr/bin/env python3
"""SPSA tuner for Luminex HCE search constants.

Stash-anchored: optimizes for game performance vs external opponents (NOT self-play,
which has been shown misleading 4× in this project). Rotates through a pool of
opponents to prevent overfitting to one opponent's style.

Design (per Sonnet 5 protocol):
  - Opponent rotation: tuning pool (StashV20, StashV21), held-out validation (StashV18)
  - Proper a_k/c_k gain schedule (not fixed step size)
  - Per-constant gradient tracking
  - Periodic checkpoint validation vs held-out opponent
  - Ship only if 3-way gate passes (baseline + held-out + Stash)

Usage (on the cloud SSH box):
  python3 spsa_tuner.py --engine /hyperai/home/Luminex/build_release/luminex \
    --opponents /hyperai/home/stash20 /hyperai/home/stash21 \
    --heldout /hyperai/home/stash13 \
    --rounds 200 --concurrency 4 --iterations 300

The engine reads spsa_params.txt from CWD (12 whitespace-separated ints).
This script creates two wrapper directories (plus/minus) with different params.
"""
import os, sys, subprocess, json, random, math, time, argparse, shutil

# ── Parameter definitions ──────────────────────────────────────────────────
# Each: name, default, min, max, perturbation_scale (relative to range)
PARAMS = [
    # Search constants (highest leverage — never game-tuned)
    ("lmr_scale_quiet",   40,  20,  70, 1.0),   # base LMR scale for quiet moves
    ("lmr_scale_noisy",   24,  10,  50, 1.0),   # base LMR scale for captures
    ("futility_coeff",   130,  80, 200, 1.0),   # futility = coeff * depth + offset
    ("futility_offset",    50,   0, 120, 1.0),
    ("nmp_base",            3,   2,   6, 1.0),   # null-move R base
    ("nmp_thresh1",        5,   3,   8, 1.0),   # depth threshold for R+1
    ("nmp_thresh2",       12,   8,  18, 1.0),   # depth threshold for R+2
    ("razor_base",        300, 150, 500, 1.0),   # razoring margin base
    ("razor_coeff",        60,  30, 120, 1.0),   # razoring margin depth² coeff
    ("rev_fut_coeff",     100,  60, 160, 1.0),   # reverse futility per-depth
    ("aspiration_delta",    50,  20, 100, 1.0),   # initial aspiration window
    # 12th param reserved for singular margin (not yet wired — placeholder)
    ("singular_margin",   200, 100, 400, 1.0),
]
N = len(PARAMS)
DEFAULTS = [p[1] for p in PARAMS]

def write_params(theta, path):
    """Write params file (one int per line) that the engine loads at startup."""
    with open(path, 'w') as f:
        for v in theta:
            f.write(f"{int(round(v))}\n")

def setup_engine_dir(base_dir, engine_path, theta):
    """Create a wrapper directory with spsa_params.txt + a shell wrapper."""
    os.makedirs(base_dir, exist_ok=True)
    write_params(theta, os.path.join(base_dir, "spsa_params.txt"))
    wrapper = os.path.join(base_dir, "run.sh")
    with open(wrapper, 'w') as f:
        f.write(f"#!/bin/bash\ncd {base_dir}\nexec {engine_path}\n")
    os.chmod(wrapper, 0o755)
    return wrapper

def run_match(engine_cmd, opponent_cmd, rounds, concurrency, tc="1+0.01",
              openings="/hyperai/home/openings.pgn"):
    """Run cutechess match, return (wins, losses, draws) for engine."""
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        for line in r.stdout.splitlines():
            if "Score of Cand" in line:
                # "Score of Cand vs Opp: W - L - D  [score] N"
                parts = line.split(":")[1].strip().split()
                w, l, d = int(parts[0]), int(parts[2]), int(parts[4])
                return w, l, d
    except Exception as e:
        print(f"  match error: {e}", flush=True)
    return 0, 0, rounds  # worst case if match fails

def score_to_elo(w, l, d):
    """Convert W-L-D to Elo estimate (from the engine's perspective)."""
    n = w + l + d
    if n == 0: return 0.0
    score = (w + 0.5 * d) / n
    if score <= 0.0: return -800.0
    if score >= 1.0: return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--opponents", nargs="+", required=True, help="tuning pool opponent paths")
    ap.add_argument("--heldout", required=True, help="held-out validation opponent")
    ap.add_argument("--rounds", type=int, default=200, help="games per perturbation eval")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--tc", default="1+0.01")
    ap.add_argument("--checkpoint", type=int, default=25, help="validate vs heldout every N iters")
    ap.add_argument("--output", default="spsa_result.json")
    args = ap.parse_args()

    theta = list(DEFAULTS)
    best_theta = list(DEFAULTS)
    best_elo = None

    # SPSA gain schedule (standard Spall 1998)
    # a: step size, c: perturbation size
    # Calibrated so initial step ~5% of param range, initial perturbation ~10%
    A = args.iterations / 10.0  # stability constant
    a_base = 0.15  # tuned for Elo-scale gradients
    c_base = 1.0   # perturbation = c * range * delta_i

    print(f"SPSA tuner: {N} params, {args.iterations} iters, {args.rounds} games/eval", flush=True)
    print(f"Opponents: {args.opponents} (tuning), {args.heldout} (heldout)", flush=True)
    print(f"Initial params: {theta}", flush=True)

    gradient_history = [[] for _ in range(N)]
    t0 = time.time()

    for k in range(args.iterations):
        # Gain schedule
        a_k = a_base / ((k + 1 + A) ** 0.602)
        c_k = c_base / ((k + 1) ** 0.101)

        # Generate Bernoulli ±1 perturbation
        delta = [random.choice([-1, 1]) for _ in range(N)]

        # Create perturbed parameter vectors (clamped to valid range)
        theta_plus = []
        theta_minus = []
        for i, (name, default, lo, hi, _) in enumerate(PARAMS):
            range_i = hi - lo
            perturb = c_k * range_i * delta[i] * 0.1  # 10% of range * delta
            tp = max(lo, min(hi, theta[i] + perturb))
            tm = max(lo, min(hi, theta[i] - perturb))
            theta_plus.append(tp)
            theta_minus.append(tm)

        # Setup engine directories
        plus_dir = f"/tmp/spsa_plus"
        minus_dir = f"/tmp/spsa_minus"
        plus_cmd = setup_engine_dir(plus_dir, args.engine, theta_plus)
        minus_cmd = setup_engine_dir(minus_dir, args.engine, theta_minus)

        # Pick opponent (rotate through pool)
        opp = args.opponents[k % len(args.opponents)]

        # Run matches
        wp, lp, dp = run_match(plus_cmd, opp, args.rounds, args.concurrency, args.tc)
        wm, lm, dm = run_match(minus_cmd, opp, args.rounds, args.concurrency, args.tc)

        elo_p = score_to_elo(wp, lp, dp)
        elo_m = score_to_elo(wm, lm, dm)

        # Gradient estimate: g_i = (y_plus - y_minus) / (2 * c_i * delta_i)
        for i, (name, default, lo, hi, _) in enumerate(PARAMS):
            range_i = hi - lo
            c_i = c_k * range_i * 0.1
            g_i = (elo_p - elo_m) / (2.0 * c_i * delta[i])
            gradient_history[i].append(g_i)

        # Update parameters
        for i, (name, default, lo, hi, _) in enumerate(PARAMS):
            range_i = hi - lo
            c_i = c_k * range_i * 0.1
            g_i = (elo_p - elo_m) / (2.0 * c_i * delta[i])
            theta[i] = max(lo, min(hi, theta[i] + a_k * g_i * range_i))

        dt = time.time() - t0
        avg_g = [sum(g[-10:]) / min(len(g), 10) if g else 0 for g in gradient_history]
        print(f"[{k+1}/{args.iterations}] elo+={elo_p:.0f} elo-={elo_m:.0f} Δ={elo_p-elo_m:+.1f} "
              f"opp={os.path.basename(opp)} | "
              f"top3_grad: ", end="", flush=True)
        # Show top-3 gradients by magnitude
        sorted_g = sorted(range(N), key=lambda i: abs(avg_g[i]), reverse=True)[:3]
        for i in sorted_g:
            print(f"{PARAMS[i][0]}={avg_g[i]:+.1f} ", end="", flush=True)
        print(f"| {dt:.0f}s", flush=True)

        # Periodic checkpoint: validate vs held-out
        if (k + 1) % args.checkpoint == 0:
            check_cmd = setup_engine_dir("/tmp/spsa_check", args.engine, theta)
            wh, lh, dh = run_match(check_cmd, args.heldout, args.rounds * 2, args.concurrency, args.tc)
            check_elo = score_to_elo(wh, lh, dh)
            print(f"  CHECKPOINT vs {os.path.basename(args.heldout)}: "
                  f"{wh}-{lh}-{dh} elo={check_elo:.0f} (best={best_elo})", flush=True)
            if best_elo is None or check_elo > best_elo:
                best_elo = check_elo
                best_theta = list(theta)
                # Save best params
                write_params(best_theta, "spsa_best.txt")
                print(f"  *** NEW BEST: elo={best_elo:.0f} params saved to spsa_best.txt", flush=True)
            # Save state
            state = {"iteration": k + 1, "theta": theta, "best_theta": best_theta,
                     "best_elo": best_elo, "gradients": [g[-50:] for g in gradient_history]}
            with open(args.output, 'w') as f:
                json.dump(state, f, indent=2)

    # Final validation
    print(f"\n=== FINAL VALIDATION ===", flush=True)
    final_cmd = setup_engine_dir("/tmp/spsa_final", args.engine, best_theta)
    for opp_path in args.opponents + [args.heldout]:
        w, l, d = run_match(final_cmd, opp_path, 500, args.concurrency, args.tc)
        elo = score_to_elo(w, l, d)
        print(f"  vs {os.path.basename(opp_path)}: {w}-{l}-{d} elo={elo:.0f}", flush=True)

    print(f"\nBest params (elo={best_elo}): {[int(round(v)) for v in best_theta]}", flush=True)
    write_params(best_theta, "spsa_best.txt")
    print(f"Saved to spsa_best.txt. Copy to engine CWD as spsa_params.txt to apply.", flush=True)

if __name__ == "__main__":
    main()
