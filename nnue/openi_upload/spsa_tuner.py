#!/usr/bin/env python3
"""SPSA tuner v4 — game-based optimization of 12 Luminex search constants.

FIXED from v3:
  - Normalized [0,1] optimization space (v3's `*ri` in the update made range
    cancel out of the gradient but not the step → identical absolute steps for
    range-50 and range-4 params, saturating small-range params instantly).
  - Per-param INTEGER perturbation with min ±1 (v3's 5%-of-range perturbation
    rounded to 0 for small-range params like nmp_base [2..6] → no gradient).
  - Step CLAMPING (noise guard): with ±57 Elo noise/eval, unguarded steps
    explode; clamp to max_step_frac decaying with k.
  - Iterate averaging (theta_bar = running mean of theta): Spall's provably
    noise-robust estimator. theta_bar is what we validate, not noisy theta.

REGIME:
  - Tuning matches: FIXED-NODE (go nodes 100000 via uci_node_proxy.py).
    Node-limited games cannot time out → eliminates the forfeit failure that
    killed v1. concurrency=4 × 2 single-threaded engines = 8 ST procs on 8
    cores, zero contention. N=100K gives ~37% vs Stash20 (healthy gradient
    band, realistic draws).
  - Validation: tc=1+0.01 (the TARGET regime) at the halfway mark and final
    gate — selects best_theta_bar by generalization to real bullet play.

Usage:
  python spsa_tuner.py --opponent /hyperai/home/stash20 \
                       --heldout /hyperai/home/stash21 \
                       --iterations 200
"""
import os, sys, subprocess, json, random, math, time, argparse

# (name, default, lo, hi) — ORDER MUST MATCH SPSAParams::load() in spsa_params.h
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
LOS  = [p[2] for p in PARAMS]
HIS  = [p[3] for p in PARAMS]
RANGE = [HIS[i] - LOS[i] for i in range(N)]

PROXY     = "/hyperai/home/uci_node_proxy.py"
LUMINEX   = "/hyperai/home/Luminex/build_release/luminex"
CUTECHESS = "/hyperai/home/cutechess/cutechess-cli"
OPENINGS  = "/hyperai/home/openings.pgn"

# --- SPSA gains (normalized [0,1] space) ---
C_BASE   = 0.10   # perturbation fraction (10% of range; min ±1 integer enforced)
A_BASE   = 0.0020 # step scale
A_STABLE = 20     # stability constant (≈ iters/10)
GAMMA_A  = 0.602
GAMMA_C  = 0.101
MAX_STEP0 = 0.05  # initial step cap (5% of range/iter); decays with k


def to_norm(theta):
    return [max(0.0, min(1.0, (theta[i] - LOS[i]) / RANGE[i])) for i in range(N)]

def to_real(x):
    return [int(round(LOS[i] + x[i] * RANGE[i])) for i in range(N)]

def write_params(theta, path):
    with open(path, 'w') as f:
        for v in theta:
            f.write(f"{int(round(v))}\n")

def make_cand_wrapper(base_dir, mode, nodes):
    """Candidate = Luminex. Writes spsa_params.txt into base_dir, cd's there."""
    os.makedirs(base_dir, exist_ok=True)
    w = os.path.join(base_dir, "run.sh")
    with open(w, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {base_dir}\n")
        if mode == 'fixed':
            f.write(f"exec python3 {PROXY} {nodes} {LUMINEX}\n")
        else:
            f.write(f"exec {LUMINEX}\n")
    os.chmod(w, 0o755)
    return w

def make_opp_wrapper(mode, opp, nodes):
    if mode == 'real':
        return opp
    w = "/tmp/spsa_opp_fixed.sh"
    with open(w, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"exec python3 {PROXY} {nodes} {opp}\n")
    os.chmod(w, 0o755)
    return w

def run_match(cand_cmd, opp_cmd, rounds, concurrency, tc, label=""):
    cmd = [
        CUTECHESS,
        "-engine", f"cmd={cand_cmd}", "name=Cand", "proto=uci",
        "-engine", f"cmd={opp_cmd}",  "name=Opp",  "proto=uci",
        "-each", f"tc={tc}",
        "-openings", f"file={OPENINGS}", "format=pgn", "order=random",
        "-rounds", str(rounds), "-repeat",
        "-concurrency", str(concurrency),
        "-draw", "movenumber=40", "movecount=10", "score=12",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
        # cutechess prints a cumulative "Score of" line after EVERY game —
        # take the LAST one (final tally), not the first.
        score_line = None
        for line in r.stdout.splitlines():
            if "Score of Cand" in line:
                score_line = line
        if score_line:
            parts = score_line.split(":")[1].strip().split()
            return int(parts[0]), int(parts[2]), int(parts[4])  # W, L, D
        # No score line — likely all games errored
        ferr = r.stderr.strip().splitlines()[-3:] if r.stderr else []
        print(f"  [{label}] NO SCORE LINE. stderr tail: {ferr}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] match TIMEOUT", flush=True)
    except Exception as e:
        print(f"  [{label}] match error: {e}", flush=True)
    return None  # signal failure

def score_to_elo(w, l, d):
    n = w + l + d
    if n == 0: return 0.0
    s = (w + 0.5 * d) / n
    if s <= 0.0: return -800.0
    if s >= 1.0: return 800.0
    return -400.0 * math.log10(1.0 / s - 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", required=True, help="primary tuning opponent (Stash20)")
    ap.add_argument("--heldout", required=True, help="held-out validation opponent (Stash21)")
    ap.add_argument("--nodes", type=int, default=100000)
    ap.add_argument("--rounds", type=int, default=50, help="games per eval (25 opening pairs)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", default="spsa_result.json")
    args = ap.parse_args()

    random.seed(args.seed)
    rnd_state = random.getstate()

    # ---- sanity check: default params vs primary at fixed-node ----
    print("=== SANITY CHECK (default vs primary, fixed-node) ===", flush=True)
    opp_fix = make_opp_wrapper('fixed', args.opponent, args.nodes)
    cand_def = make_cand_wrapper("/tmp/spsa_default", 'fixed', args.nodes)
    write_params(DEFAULTS, "/tmp/spsa_default/spsa_params.txt")
    r = run_match(cand_def, opp_fix, 50, args.concurrency, "999+0", "sanity")
    if r is None:
        print("  *** sanity match failed — aborting (check cutechess/proxy)", flush=True)
        sys.exit(1)
    w, l, d = r
    s = (w + 0.5 * d) / (w + l + d)
    print(f"  vs {os.path.basename(args.opponent)} @N={args.nodes}: {w}W-{l}L-{d}D score={s:.1%}", flush=True)
    if s < 0.20 or s > 0.80:
        print(f"  *** FAIL: degenerate score — harness broken or wrong regime", flush=True)
        sys.exit(1)
    print("  PASS.", flush=True)

    # ---- init ----
    theta = list(DEFAULTS)          # real (int) space, current iterate
    theta_bar = list(DEFAULTS)      # running mean (iterate averaging)
    x = to_norm(theta)
    best_bar = list(DEFAULTS)
    best_elo = None

    print(f"\nSPSA v4: {N} params, {args.iterations} iters, {args.rounds} games/eval, "
          f"N={args.nodes} conc={args.concurrency}", flush=True)
    print(f"Tuning vs {os.path.basename(args.opponent)} | Heldout {os.path.basename(args.heldout)}", flush=True)
    print(f"Gains: c0={C_BASE} a0={A_BASE} A={A_STABLE} maxstep0={MAX_STEP0}", flush=True)
    print(f"Start theta (defaults): {theta}", flush=True)
    print(f"{'k':>4} {'Ep':>5} {'Em':>5} {'dE':>5} | "
          f"{'max|g|':>6} {'step':>5} | top movers", flush=True)

    t0 = time.time()
    opp_fix = make_opp_wrapper('fixed', args.opponent, args.nodes)
    fail_streak = 0

    for k in range(args.iterations):
        a_k = A_BASE / ((1 + k / A_STABLE) ** GAMMA_A)
        c_k = C_BASE / ((1 + k) ** GAMMA_C)
        max_step = MAX_STEP0 / ((1 + k / 30) ** 0.4)

        # perturb: real-space integer, min ±1, scaled by c_k
        deltas = [random.choice([-1, 1]) for _ in range(N)]
        tp, tm, disp_norm = [], [], []
        for i in range(N):
            pert = max(1.0, round(c_k * RANGE[i]))
            pv = int(max(LOS[i], min(HIS[i], theta[i] + pert * deltas[i])))
            mv = int(max(LOS[i], min(HIS[i], theta[i] - pert * deltas[i])))
            tp.append(pv); tm.append(mv)
            disp_norm.append((pv - mv) / RANGE[i])  # actual normalized displacement

        cand_p = make_cand_wrapper("/tmp/spsa_plus",  'fixed', args.nodes)
        cand_m = make_cand_wrapper("/tmp/spsa_minus", 'fixed', args.nodes)
        write_params(tp, "/tmp/spsa_plus/spsa_params.txt")
        write_params(tm, "/tmp/spsa_minus/spsa_params.txt")

        rp = run_match(cand_p, opp_fix, args.rounds, args.concurrency, "999+0", f"k{k}+")
        rm = run_match(cand_m, opp_fix, args.rounds, args.concurrency, "999+0", f"k{k}-")
        if rp is None or rm is None:
            fail_streak += 1
            print(f"[{k+1}/{args.iterations}] match failed ({fail_streak}) — skipping", flush=True)
            if fail_streak >= 5:
                print("  *** 5 consecutive failures — aborting", flush=True)
                break
            continue
        fail_streak = 0
        wp, lp, dp = rp
        wm, lm, dm = rm
        ep, em = score_to_elo(wp, lp, dp), score_to_elo(wm, lm, dm)

        # gradient (normalized) + step (clamped)
        gs, steps = [], []
        for i in range(N):
            if abs(disp_norm[i]) < 1e-9:
                gs.append(0.0); steps.append(0.0); continue
            gi = (ep - em) / disp_norm[i]            # Elo per unit-normalized
            si = a_k * gi
            si = max(-max_step, min(max_step, si))    # noise guard
            x[i] = max(0.0, min(1.0, x[i] + si))
            gs.append(gi); steps.append(si)
        theta = to_real(x)

        # iterate averaging (Spall): running mean of theta
        theta_bar = [(theta_bar[i] * (k + 1) + theta[i]) / (k + 2) for i in range(N)]
        theta_bar_real = to_real(theta_bar)

        # report top movers (largest |gradient|)
        movers = sorted(range(N), key=lambda i: abs(gs[i]), reverse=True)[:3]
        mover_str = " ".join(f"{PARAMS[i][0][:8]}={gs[i]:+6.0f}" for i in movers)
        maxstep = max(abs(s) for s in steps)
        dt = time.time() - t0
        print(f"[{k+1:>3}/{args.iterations}] {ep:5.0f} {em:5.0f} {ep-em:+5.0f} | "
              f"{max(abs(g) for g in gs):6.0f} {maxstep:5.3f} | {mover_str} | {dt:.0f}s", flush=True)

        # checkpoint at halfway: theta_bar vs primary at TARGET tc=1+0.01
        if k + 1 == args.iterations // 2:
            print(f"  === HALFWAY GATE: theta_bar vs primary @ tc=1+0.01 (100g) ===", flush=True)
            cb = make_cand_wrapper("/tmp/spsa_halfgate", 'real', args.nodes)
            write_params(theta_bar_real, "/tmp/spsa_halfgate/spsa_params.txt")
            rb = run_match(cb, args.opponent, 100, args.concurrency, "1+0.01", "halfgate")
            if rb:
                wh, lh, dh = rb
                eh = score_to_elo(wh, lh, dh)
                sh = (wh + 0.5 * dh) / (wh + lh + dh)
                print(f"  halfgate: {wh}W-{lh}L-{dh}D score={sh:.1%} elo={eh:.0f}", flush=True)
                if best_elo is None or eh > best_elo:
                    best_elo = eh; best_bar = list(theta_bar_real)
                    write_params(best_bar, "spsa_best.txt")
                with open(args.output, 'w') as f:
                    json.dump({"k": k+1, "theta": theta, "theta_bar": theta_bar_real,
                               "best": best_bar, "best_elo": best_elo}, f, indent=2)

    # ---- FINAL GATE: theta_bar at tc=1+0.01 vs primary + baseline(default) ----
    print(f"\n=== FINAL VALIDATION @ tc=1+0.01 (500g each) ===", flush=True)
    theta_bar_real = to_real(theta_bar)
    write_params(theta_bar_real, "spsa_tuned.txt")
    write_params(DEFAULTS, "spsa_default_saved.txt")
    cf = make_cand_wrapper("/tmp/spsa_final_tuned", 'real', args.nodes)
    write_params(theta_bar_real, "/tmp/spsa_final_tuned/spsa_params.txt")
    cd = make_cand_wrapper("/tmp/spsa_final_base", 'real', args.nodes)
    # baseline dir must have NO spsa_params.txt to use engine defaults
    if os.path.exists("/tmp/spsa_final_base/spsa_params.txt"):
        os.remove("/tmp/spsa_final_base/spsa_params.txt")

    rf = run_match(cf, args.opponent, 500, args.concurrency, "1+0.01", "final-vs-stash")
    rd = run_match(cd, args.opponent, 500, args.concurrency, "1+0.01", "baseline-vs-stash")
    rself = run_match(cf, cd, 500, args.concurrency, "1+0.01", "tuned-vs-baseline")  # same engine, diff params
    if rf: print(f"  TUNED  vs {os.path.basename(args.opponent)}: {rf[0]}W-{rf[1]}L-{rf[2]}D "
                 f"elo={score_to_elo(*rf):.0f}", flush=True)
    if rd: print(f"  BASE   vs {os.path.basename(args.opponent)}: {rd[0]}W-{rd[1]}L-{rd[2]}D "
                 f"elo={score_to_elo(*rd):.0f}", flush=True)
    if rself:
        ws, ls, ds = rself
        sel = (ws + 0.5 * ds) / (ws + ls + ds)
        print(f"  TUNED vs BASE (self-play): {ws}W-{ls}L-{ds}D score={sel:.1%}", flush=True)

    print(f"\n=== THETA_BAR (tuned) ===", flush=True)
    for i, (n, d, lo, hi) in enumerate(PARAMS):
        chg = theta_bar_real[i] - DEFAULTS[i]
        mark = " *" if abs(chg) >= 1 else ""
        print(f"  {n:18s} {DEFAULTS[i]:>4d} -> {theta_bar_real[i]:>4d} "
              f"(range {lo}-{hi}){mark}", flush=True)
    print(f"\nBest (halfgate) elo={best_elo}. Wrote spsa_tuned.txt + spsa_best.txt", flush=True)
    with open(args.output, 'w') as f:
        json.dump({"theta_bar": theta_bar_real, "defaults": DEFAULTS,
                   "best": best_bar, "best_elo": best_elo,
                   "final_tuned": rf, "final_baseline": rd, "final_self": rself}, f, indent=2)

if __name__ == "__main__":
    main()
