#!/usr/bin/env python3
"""
Luminex SPSA Tuner - Self-Engineered Parameter Optimization
Drives cutechess-cli to play self-play games and optimize eval parameters.

Usage:
  python3 spsa_tune.py --engine ./build/luminex --iterations 50000 --rounds 2 --tc 1+0.01

Algorithm: SPSA (Simultaneous Perturbation Stochastic Approximation)
  - All parameters perturbed simultaneously each iteration
  - Two games played (color reversed) between theta+ and theta-
  - Parameters updated based on game outcomes
  - alpha=0.602, gamma=0.101 (Spall's optimal values)
"""

import subprocess
import argparse
import json
import os
import sys
import random
import math
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ============================================================
# SPSA Parameter
# ============================================================
@dataclass
class SPSAParam:
    name: str           # UCI option name (must match engine)
    theta: float        # Current value
    min_val: float
    max_val: float
    c_end: float = 4.0  # Perturbation size at last iteration
    R_end: float = 0.002
    c_k: float = 0.0
    R_k: float = 0.0
    delta: float = 0.0
    theta_plus: float = 0.0
    theta_minus: float = 0.0
    initial: float = 0.0
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.initial = self.theta
        self.history.append(self.theta)


# ============================================================
# Default Tunable Parameters
# ============================================================
def get_default_params() -> List[SPSAParam]:
    return [
        # EvalParams struct (24 params) - UCI names from uci.cpp
        SPSAParam("BishopPairMG",          30,    0,   80),
        SPSAParam("BishopPairEG",          96,   20,  180),
        SPSAParam("RookOpenMG",            29,    0,   60),
        SPSAParam("RookOpenEG",            47,   10,   90),
        SPSAParam("RookSemiOpenMG",        20,    0,   50),
        SPSAParam("RookSemiOpenEG",        15,    0,   40),
        SPSAParam("Rook7thMG",             30,    0,   60),
        SPSAParam("Rook7thEG",             17,    0,   50),
        SPSAParam("PawnShieldCenter",      11,    0,   30),
        SPSAParam("PawnShieldKnight",      15,    0,   40),
        SPSAParam("PawnShieldRook",         8,    0,   25),
        SPSAParam("PawnStorm",              9,    0,   25),
        SPSAParam("OpenFilePenaltyMG",     21,    0,   50),
        SPSAParam("OpenFilePenaltyEG",     18,    0,   50),
        SPSAParam("OutpostKnightMG",       27,    0,   70),
        SPSAParam("OutpostKnightEG",       17,    0,   50),
        SPSAParam("OutpostBishopMG",       50,   10,  100),
        SPSAParam("OutpostBishopEG",       31,    0,   70),
        SPSAParam("HangingPawnMG",          8,  -20,   40),
        SPSAParam("HangingPawnEG",         39,    0,   80),
        SPSAParam("FarKnightMG",           29,    0,   60),
        SPSAParam("FarKnightEG",            8,    0,   30),
        SPSAParam("FarBishopMG",            5,  -20,   30),
        SPSAParam("FarBishopEG",            3,  -10,   20),
    ]


# ============================================================
# SPSA Core
# ============================================================
class SPSATuner:
    def __init__(self, params: List[SPSAParam], cfg: dict):
        self.params = params
        self.iteration = 0
        self.alpha = cfg.get('alpha', 0.602)
        self.gamma = cfg.get('gamma', 0.101)
        self.max_iter = cfg.get('iterations', 50000)
        self.A = cfg.get('A_fraction', 0.10) * self.max_iter
        self.rng = random.Random(cfg.get('seed', 42))

    def update_gains(self):
        k = self.iteration
        for p in self.params:
            c_0 = p.c_end * (self.max_iter ** self.gamma)
            p.c_k = c_0 / ((k + 1) ** self.gamma)
            a_end = p.R_end * p.c_end ** 2
            a_0 = a_end * ((self.A + self.max_iter) ** self.alpha)
            a_k = a_0 / ((self.A + k + 1) ** self.alpha)
            p.R_k = a_k / (p.c_k ** 2) if p.c_k > 0 else 0

    def perturb(self):
        self.update_gains()
        for p in self.params:
            p.delta = 1.0 if self.rng.random() < 0.5 else -1.0
            p.theta_plus = max(p.min_val, min(self.theta + p.c_k * p.delta, p.max_val))
            p.theta_minus = max(p.min_val, min(self.theta - p.c_k * p.delta, p.max_val))

    def update(self, result: float):
        for p in self.params:
            if p.c_k == 0 or p.delta == 0:
                continue
            gradient = result / (2.0 * p.c_k * p.delta)
            p.theta = max(p.min_val, min(p.theta + p.R_k * p.c_k * gradient, p.max_val))
            p.history.append(p.theta)
        self.iteration += 1

    @property
    def theta(self):
        # Use mean of last 10% of history for each param (smoothing)
        pass

    def uci_opts(self, use_plus: bool) -> list:
        """Return list of cutechess option strings like 'option.Name=value'."""
        opts = []
        for p in self.params:
            val = int(round(p.theta_plus if use_plus else p.theta_minus))
            opts.append(f"option.{p.name}={val}")
        return opts

    def save(self, path: str):
        state = {
            'iteration': self.iteration,
            'params': [{
                'name': p.name, 'theta': p.theta,
                'min_val': p.min_val, 'max_val': p.max_val,
                'c_end': p.c_end, 'R_end': p.R_end,
                'initial': p.initial,
            } for p in self.params]
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            state = json.load(f)
        self.iteration = state['iteration']
        for p, s in zip(self.params, state['params']):
            p.theta = s['theta']


# ============================================================
# CuteChess Runner
# ============================================================
def run_games(cutechess: str, engine: str, opts_plus: list, opts_minus: list,
              rounds: int, tc: str, pgn_file: str) -> Tuple[int, int, int]:
    """Run cutechess games. Returns (wins_plus, wins_minus, draws)."""
    cmd = [
        cutechess,
        "-rounds", str(rounds),
        "-engine", "cmd=" + engine, "name=LuminexPlus", *opts_plus,
        "-engine", "cmd=" + engine, "name=LuminexMinus", *opts_minus,
        "-each", "proto=uci", "tc=" + tc,
        "-pgnout", pgn_file,
        "-recover",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = proc.stdout + proc.stderr

        last_match = None
        for line in output.split('\n'):
            if 'Score of LuminexPlus vs LuminexMinus' in line:
                m = re.search(r'Score of \w+ vs \w+:\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)', line)
                if m:
                    last_match = m
        if last_match:
            return int(last_match.group(1)), int(last_match.group(2)), int(last_match.group(3))
        return 0, 0, 0
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return 0, 0, 0


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Luminex SPSA Tuner")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--cutechess", default="cutechess-cli", help="Path to cutechess-cli")
    ap.add_argument("--iterations", type=int, default=50000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tc", default="1+0.01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--output", default="spsa_result.json")
    args = ap.parse_args()

    params = get_default_params()
    tuner = SPSATuner(params, {
        'iterations': args.iterations,
        'alpha': 0.602,
        'gamma': 0.101,
        'A_fraction': 0.10,
        'seed': args.seed,
    })

    if args.resume:
        tuner.load(args.resume)
        print(f"Resumed from iteration {tuner.iteration}")

    print(f"Luminex SPSA Tuner")
    print(f"Params: {len(params)}, Iterations: {args.iterations}, Rounds/iter: {args.rounds}")
    print(f"TC: {args.tc}, Engine: {args.engine}")
    print()

    start_time = time.time()
    results_log = []

    while tuner.iteration < args.iterations:
        tuner.perturb()
        opts_plus = tuner.uci_opts(use_plus=True)
        opts_minus = tuner.uci_opts(use_plus=False)

        w, l, d = run_games(args.cutechess, args.engine, opts_plus, opts_minus,
                           args.rounds, args.tc, "spsa_iter.pgn")

        total = w + l + d
        result = (w - l) / total if total > 0 else 0.0
        tuner.update(result)

        elapsed = time.time() - start_time
        iters_per_sec = tuner.iteration / elapsed if elapsed > 0 else 0

        results_log.append({
            'iter': tuner.iteration,
            'w': w, 'l': l, 'd': d,
            'result': result,
            'theta': {p.name: int(round(p.theta)) for p in tuner.params}
        })

        if tuner.iteration % 10 == 0:
            print(f"[{tuner.iteration:6d}/{args.iterations}] "
                  f"W:{w} L:{l} D:{d} res={result:+.2f} "
                  f"({iters_per_sec:.1f} iter/s)")
            tuner.save(args.output)

        if tuner.iteration % 1000 == 0:
            # Print parameter summary
            print("\n  Current parameters:")
            for p in tuner.params:
                chg = p.theta - p.initial
                print(f"    {p.name:25s}: {int(round(p.theta)):5d}  (chg={chg:+.1f})")
            print()
            tuner.save(f"spsa_checkpoint_{tuner.iteration}.json")

    # Final output
    tuner.save(args.output)
    print(f"\nTuning complete. {tuner.iteration} iterations in {time.time()-start_time:.0f}s")
    print(f"Results: {args.output}")

    # Print final parameters in UCI setoption format
    print("\n# Final parameters (UCI setoption format):")
    for p in tuner.params:
        chg = int(round(p.theta)) - int(round(p.initial))
        print(f"setoption name {p.name} value {int(round(p.theta))}  # chg={chg:+d}")


if __name__ == "__main__":
    main()
