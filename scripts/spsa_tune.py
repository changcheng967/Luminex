#!/usr/bin/env python3
"""
SPSA Hill-Climbing Tuner for Luminex Chess Engine

Tunes eval parameters via random perturbation + cutechess game testing.
Stochastic hill climbing: perturb from current best, test vs baseline,
keep if win rate improves. Runs for up to 2 hours.

Usage:
  python3 scripts/spsa_tune.py

Requirements:
  - Baseline binary at /hyperai/home/luminex_baseline
  - Cutechess at /hyperai/home/cutechess/build/cutechess-cli
  - CMake at /hyperai/home/bin/cmake
  - Luminex source at /hyperai/home/Luminex
"""

import subprocess, random, time, json, os, sys, re, shutil
from datetime import datetime

# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
ENGINE_DIR  = "/hyperai/home/Luminex"
BUILD_DIR   = "/hyperai/home/Luminex/build_tune"
EVAL_SRC    = f"{ENGINE_DIR}/src/evaluation.cpp"
EVAL_BAK    = f"{ENGINE_DIR}/src/evaluation.cpp.spsa_bak"
BASELINE    = "/hyperai/home/luminex_baseline"
CUTECHESS   = "/hyperai/home/cutechess/build/cutechess-cli"
CMAKE       = "/hyperai/home/bin/cmake"
ROUNDS      = 100
MAX_SECONDS = 6900  # ~115 min, leaves margin
STATE_FILE  = "/hyperai/home/spsa_state.json"
LOG_FILE    = "/hyperai/home/spsa_log.txt"

# Tunable parameters: (name, default, min, max, step)
PARAMS = [
    ("pawn_eg",   100,  50,  200,  20),
    ("knight_eg", 290, 150,  550,  40),
    ("bishop_eg", 310, 150,  550,  40),
    ("rook_eg",   530, 300,  950,  60),
    ("queen_eg",  940, 600, 1500,  80),
    ("ks_scale",  500, 200, 1000,  80),
    ("ks_k",      200,  50,  600,  40),
    ("passer_eg", 100,  40,  250,  15),  # % scale of base passed pawn EG
]


# ═══════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════
#  Source Patching
# ═══════════════════════════════════════════
def read_src():
    with open(EVAL_SRC, "r", encoding="utf-8") as f:
        return f.read()

def write_src(src):
    with open(EVAL_SRC, "w", encoding="utf-8") as f:
        f.write(src)

def patch_piece_eg(src, v):
    """Replace PieceValueEG array."""
    pat = r"static int PieceValueEG\[8\]\s*=\s*\{[^}]+\}\s*;"
    rep = (f"static int PieceValueEG[8] = {{ {v['pawn_eg']}, "
           f"{v['knight_eg']}, {v['bishop_eg']}, "
           f"{v['rook_eg']}, {v['queen_eg']}, 0, 0, 0 }};")
    return re.sub(pat, rep, src)

def patch_ks(src, scale, k):
    """Replace king safety sigmoid parameters."""
    pat = r"int danger = std::min\(\d+,\s*\(\d+\s*\*\s*au2\)\s*/\s*\(au2\s*\+\s*\d+\)\)\s*;"
    rep = f"int danger = std::min({scale}, ({scale} * au2) / (au2 + {k}));"
    return re.sub(pat, rep, src)

# Base passed pawn EG values for scaling
PASSER_EG_BASE = [0, 10, 20, 40, 70, 120, 200, 0]

def patch_passer_eg(src, scale_pct):
    """Replace PassedEG array with scaled values."""
    scaled = [int(round(v * scale_pct / 100)) for v in PASSER_EG_BASE]
    pat = r"static constexpr int PassedEG\[8\]\s*=\s*\{[^}]+\}\s*;"
    rep = f"static constexpr int PassedEG[8] = {{ {', '.join(map(str, scaled))} }};"
    return re.sub(pat, rep, src)

def apply_all_patches(v):
    """Apply all parameter patches to evaluation.cpp."""
    src = read_src()
    src = patch_piece_eg(src, v)
    src = patch_ks(src, v["ks_scale"], v["ks_k"])
    src = patch_passer_eg(src, v["passer_eg"])
    write_src(src)

def restore_src():
    shutil.copy2(EVAL_BAK, EVAL_SRC)


# ═══════════════════════════════════════════
#  Build & Test
# ═══════════════════════════════════════════
def cmake_configure():
    os.makedirs(BUILD_DIR, exist_ok=True)
    r = subprocess.run(
        f"{CMAKE} -S {ENGINE_DIR} -B {BUILD_DIR} "
        f"-DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3",
        shell=True, capture_output=True, text=True, timeout=120)
    return r.returncode == 0

def cmake_build():
    r = subprocess.run(
        f"{CMAKE} --build {BUILD_DIR} -j$(nproc) 2>&1 | tail -3",
        shell=True, capture_output=True, text=True, timeout=300)
    return r.returncode == 0

def run_match():
    """Run cutechess, return (w, l, d, win_pct)."""
    tuned = f"{BUILD_DIR}/luminex"
    cmd = (f'{CUTECHESS} '
           f'-engine cmd={tuned} name=Tuned proto=uci '
           f'-engine cmd={BASELINE} name=Base proto=uci '
           f'-each tc=1+0.01 -rounds {ROUNDS} -repeat 2>&1')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)

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


# ═══════════════════════════════════════════
#  Perturbation
# ═══════════════════════════════════════════
def perturb(current):
    """Random perturbation of all params from current values."""
    result = {}
    for name, default, lo, hi, step in PARAMS:
        val = current.get(name, default)
        delta = random.choice([-1, 1]) * step * random.uniform(0.5, 1.5)
        result[name] = max(lo, min(hi, round(val + delta)))
    return result


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════
def main():
    random.seed()
    start = time.time()

    # Pre-checks
    for path, label in [(BASELINE, "Baseline"), (CUTECHESS, "Cutechess"),
                         (EVAL_SRC, "Eval source"), (CMAKE, "CMake")]:
        if not os.path.exists(path):
            log(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # Backup original source (once)
    if not os.path.exists(EVAL_BAK):
        shutil.copy2(EVAL_SRC, EVAL_BAK)
        log("Backed up evaluation.cpp")
    else:
        log("Backup already exists")

    # CMake configure (once — incremental builds after)
    log("Configuring build...")
    if not cmake_configure():
        log("Configure FAILED")
        sys.exit(1)
    log("Configure OK")

    # Load or init state
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        log(f"Resumed: iter={state['iteration']}, "
            f"best={state['best_score']:.1f}%")
    else:
        state = {
            "iteration": 0,
            "best_params": {n: d for n, d, *_ in PARAMS},
            "best_score": 50.0,
            "history": [],
        }

    log(f"SPSA started: {len(PARAMS)} params, "
        f"{ROUNDS} games/test, up to {MAX_SECONDS // 60}m")
    log(f"Best params: {state['best_params']}")

    # ── Main loop ──
    while time.time() - start < MAX_SECONDS:
        state["iteration"] += 1
        it = state["iteration"]
        elapsed = (time.time() - start) / 60
        remaining = MAX_SECONDS / 60 - elapsed

        log(f"\n--- Iteration {it} "
            f"({elapsed:.0f}m done, {remaining:.0f}m left) ---")

        # Generate candidate by perturbing from current best
        candidate = perturb(state["best_params"])
        c = candidate
        log(f"Test: P={c['pawn_eg']} N={c['knight_eg']} "
            f"B={c['bishop_eg']} R={c['rook_eg']} Q={c['queen_eg']} "
            f"KS={c['ks_scale']}/{c['ks_k']} "
            f"Passer={c['passer_eg']}%")

        # Patch source
        apply_all_patches(candidate)

        # Build (incremental — only evaluation.cpp changed)
        t0 = time.time()
        ok = cmake_build()
        build_s = time.time() - t0
        if not ok:
            log(f"BUILD FAILED ({build_s:.0f}s)")
            restore_src()
            continue
        log(f"Build OK ({build_s:.0f}s)")

        # Test
        log(f"Running {ROUNDS} games vs baseline...")
        t0 = time.time()
        w, l, d, pct = run_match()
        game_s = time.time() - t0
        log(f"Result: {w}-{l}-{d} = {pct:.1f}% ({game_s:.0f}s)")

        # Accept / reject
        if pct > state["best_score"] + 0.5:
            state["best_params"] = candidate
            state["best_score"] = pct
            log(f"ACCEPTED (new best {pct:.1f}%)")
        else:
            log(f"REJECTED ({pct:.1f}% <= best {state['best_score']:.1f}%)")
            restore_src()

        # Persist state
        state["history"].append({
            "iter": it,
            "params": candidate,
            "w": w, "l": l, "d": d,
            "pct": round(pct, 1),
            "accepted": pct > state["best_score"] + 0.5,
            "build_s": round(build_s),
            "game_s": round(game_s),
            "elapsed_min": round(elapsed, 1),
        })
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    # ── Done ──
    log(f"\n{'=' * 60}")
    log(f"TUNING COMPLETE: {state['iteration']} iterations")
    log(f"Best score: {state['best_score']:.1f}% vs baseline")
    log(f"Best params: {json.dumps(state['best_params'], indent=2)}")

    # Show diffs from defaults
    defaults = {n: d for n, d, *_ in PARAMS}
    log("\nChanges from defaults:")
    for name in defaults:
        old = defaults[name]
        new = state["best_params"].get(name, old)
        delta = new - old
        pct_ch = f" ({delta / old * 100:+.0f}%)" if old else ""
        log(f"  {name}: {old} -> {new} ({delta:+d}){pct_ch}")

    # Apply best params permanently
    apply_all_patches(state["best_params"])
    log("\nBest parameters applied to evaluation.cpp")
    log("Next steps:")
    log(f"  1. Build: cd {ENGINE_DIR} && "
        f"{CMAKE} -S . -B build_release -DCMAKE_BUILD_TYPE=Release && "
        f"{CMAKE} --build build_release -j$(nproc)")
    log(f"  2. Self-test: {CUTECHESS} "
        f"-engine cmd={ENGINE_DIR}/build_release/luminex name=New proto=uci "
        f"-engine cmd={BASELINE} name=Old proto=uci "
        f"-each tc=1+0.01 -rounds 200 -repeat")


if __name__ == "__main__":
    main()
