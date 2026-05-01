#!/usr/bin/env python3
"""
Systematic parameter sweep for Luminex eval tuning.

Tests predefined parameter sets (informed by cross-engine analysis)
with 200 games each vs baseline. More reliable than random SPSA
because each test uses enough games for statistical significance.

Parameter sets tested (EG piece values only):
  A: Moderate EG increase (P+10%, N+17%, B+16%, R+17%, Q+11%)
  B: Conservative (P+5%, N+7%, B+6%, R+8%, Q+4%)
  C: R+Q focus only
  D: R-only increase
  E: Iteration 1 from SPSA (P=80, N=319, B=345, R=604, Q=1042)

Usage:
  python3 scripts/param_sweep.py

After sweep, best set is confirmed with 200-game self-test.
"""

import subprocess, time, json, os, sys, re, shutil
from datetime import datetime

# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
ENGINE_DIR  = "/hyperai/home/Luminex"
BUILD_DIR   = "/hyperai/home/Luminex/build_sweep"
EVAL_SRC    = f"{ENGINE_DIR}/src/evaluation.cpp"
EVAL_BAK    = f"{ENGINE_DIR}/src/evaluation.cpp.sweep_bak"
BASELINE    = "/hyperai/home/luminex_baseline"
CUTECHESS   = "/hyperai/home/cutechess/build/cutechess-cli"
CMAKE       = "/hyperai/home/bin/cmake"
ROUNDS      = 200
LOG_FILE    = "/hyperai/home/sweep_log.txt"
RESULT_FILE = "/hyperai/home/sweep_results.json"

# Defaults: P=100, N=290, B=310, R=530, Q=940
PARAM_SETS = {
    "A_moderate": {
        "desc": "Moderate EG increase (~15% across board)",
        "pawn_eg": 110, "knight_eg": 340, "bishop_eg": 360,
        "rook_eg": 620, "queen_eg": 1050,
    },
    "B_conservative": {
        "desc": "Conservative EG increase (~6%)",
        "pawn_eg": 105, "knight_eg": 310, "bishop_eg": 330,
        "rook_eg": 570, "queen_eg": 980,
    },
    "C_rq_focus": {
        "desc": "Only R+Q EG increase (heavy pieces more valuable EG)",
        "pawn_eg": 100, "knight_eg": 290, "bishop_eg": 310,
        "rook_eg": 650, "queen_eg": 1050,
    },
    "D_r_only": {
        "desc": "Only Rook EG increase",
        "pawn_eg": 100, "knight_eg": 290, "bishop_eg": 310,
        "rook_eg": 600, "queen_eg": 940,
    },
    "E_spsa_hit": {
        "desc": "Best from SPSA run (P=80, N=319, B=345, R=604, Q=1042)",
        "pawn_eg": 80, "knight_eg": 319, "bishop_eg": 345,
        "rook_eg": 604, "queen_eg": 1042,
    },
    "F_stash_tiny": {
        "desc": "Tiny step toward Stash ratios (just +20% on N/B, +15% R/Q)",
        "pawn_eg": 100, "knight_eg": 350, "bishop_eg": 370,
        "rook_eg": 610, "queen_eg": 1080,
    },
}

# ═══════════════════════════════════════════
#  Functions
# ═══════════════════════════════════════════
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def read_src():
    with open(EVAL_SRC, "r", encoding="utf-8") as f:
        return f.read()

def write_src(src):
    with open(EVAL_SRC, "w", encoding="utf-8") as f:
        f.write(src)

def patch_piece_eg(src, v):
    pat = r"static int PieceValueEG\[8\]\s*=\s*\{[^}]+\}\s*;"
    rep = (f"static int PieceValueEG[8] = {{ {v['pawn_eg']}, "
           f"{v['knight_eg']}, {v['bishop_eg']}, "
           f"{v['rook_eg']}, {v['queen_eg']}, 0, 0, 0 }};")
    return re.sub(pat, rep, src)

def restore_src():
    shutil.copy2(EVAL_BAK, EVAL_SRC)

def configure():
    os.makedirs(BUILD_DIR, exist_ok=True)
    r = subprocess.run(
        f"{CMAKE} -S {ENGINE_DIR} -B {BUILD_DIR} "
        f"-DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3",
        shell=True, capture_output=True, text=True, timeout=120)
    return r.returncode == 0

def build():
    r = subprocess.run(
        f"{CMAKE} --build {BUILD_DIR} -j$(nproc) 2>&1 | tail -3",
        shell=True, capture_output=True, text=True, timeout=300)
    return r.returncode == 0

def run_match():
    tuned = f"{BUILD_DIR}/luminex"
    cmd = (f'{CUTECHESS} '
           f'-engine cmd={tuned} name=Tuned proto=uci '
           f'-engine cmd={BASELINE} name=Base proto=uci '
           f'-each tc=1+0.01 -rounds {ROUNDS} -repeat 2>&1')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    output = r.stdout + r.stderr
    last = None
    for line in output.split("\n"):
        if "Score of" in line:
            m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", line)
            if m:
                last = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if last:
        w, l, d = last
        t = w + l + d
        return w, l, d, (w + 0.5 * d) / t * 100 if t else 50.0
    return 0, 0, 0, 50.0

def elo_from_pct(pct):
    if pct <= 0 or pct >= 100:
        return 0
    return 400 * (pct / 100 - 0.5)

# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════
def main():
    start = time.time()

    # Pre-checks
    for path, label in [(BASELINE, "Baseline"), (CUTECHESS, "Cutechess"),
                         (EVAL_SRC, "Eval source"), (CMAKE, "CMake")]:
        if not os.path.exists(path):
            log(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # Backup
    if not os.path.exists(EVAL_BAK):
        shutil.copy2(EVAL_SRC, EVAL_BAK)
        log("Backed up evaluation.cpp")
    else:
        log("Backup exists, restoring to original")
        restore_src()

    # Configure
    log("Configuring build...")
    if not configure():
        log("Configure FAILED")
        sys.exit(1)
    log("Configure OK")

    results = {}
    log(f"Starting sweep: {len(PARAM_SETS)} parameter sets, "
        f"{ROUNDS} games each")
    log(f"Defaults: P=100, N=290, B=310, R=530, Q=940")

    for name, params in PARAM_SETS.items():
        log(f"\n{'='*50}")
        log(f"Testing {name}: {params['desc']}")
        log(f"P={params['pawn_eg']} N={params['knight_eg']} "
            f"B={params['bishop_eg']} R={params['rook_eg']} "
            f"Q={params['queen_eg']}")

        # Patch
        src = read_src()
        src = patch_piece_eg(src, params)
        write_src(src)

        # Build
        t0 = time.time()
        if not build():
            log("BUILD FAILED")
            restore_src()
            results[name] = {"error": "build_failed", "pct": 0}
            continue
        build_s = time.time() - t0
        log(f"Build OK ({build_s:.0f}s)")

        # Test
        log(f"Running {ROUNDS} games vs baseline...")
        t0 = time.time()
        w, l, d, pct = run_match()
        game_s = time.time() - t0
        elo = elo_from_pct(pct)
        log(f"Result: {w}-{l}-{d} = {pct:.1f}% ({elo:+.0f} Elo) "
            f"({game_s:.0f}s)")

        results[name] = {
            "w": w, "l": l, "d": d,
            "pct": round(pct, 1),
            "elo": round(elo, 1),
            "params": params,
        }

        # Restore
        restore_src()

        # Save intermediate results
        with open(RESULT_FILE, "w") as f:
            json.dump(results, f, indent=2)

        elapsed = (time.time() - start) / 60
        log(f"Elapsed: {elapsed:.0f}m")

    # ── Summary ──
    log(f"\n{'='*60}")
    log("SWEEP COMPLETE")
    log(f"{'='*60}")

    # Sort by win rate
    ranked = sorted(results.items(),
                    key=lambda x: x[1].get("pct", 0), reverse=True)

    log("\nResults (best to worst):")
    for name, r in ranked:
        if "error" in r:
            log(f"  {name}: BUILD FAILED")
            continue
        elo = r.get("elo", 0)
        log(f"  {name}: {r['pct']:.1f}% ({elo:+.0f} Elo) "
            f"[{r['w']}-{r['l']}-{r['d']}]")

    # Best set
    best_name, best_result = ranked[0]
    best_pct = best_result.get("pct", 50)
    log(f"\nBest: {best_name} at {best_pct:.1f}% "
        f"({best_result.get('elo', 0):+.0f} Elo)")

    if best_pct > 50.5:
        log(f"\nApplying {best_name} parameters to evaluation.cpp")
        src = read_src()
        src = patch_piece_eg(src, PARAM_SETS[best_name])
        write_src(src)
        log("Applied! Build with:")
        log(f"  {CMAKE} --build {ENGINE_DIR}/build_release -j$(nproc)")
    else:
        log("\nNo set beat baseline. Defaults remain optimal.")

    total_m = (time.time() - start) / 60
    log(f"Total time: {total_m:.0f}m")


if __name__ == "__main__":
    main()
