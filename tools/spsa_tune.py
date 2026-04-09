#!/usr/bin/env python3
"""
Luminex SPSA Tuner - Dynamic Scaling Edition
Optimizes chess engine parameters using Simultaneous Perturbation Stochastic Approximation.
Automatically detects hardware and scales concurrency accordingly.
"""

import subprocess
import argparse
import json
import os
import sys
import random
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# ============================================================
# SPSA Parameter Structure
# ============================================================
@dataclass
class SPSAParam:
    name: str           # UCI option name
    theta: float        # Current central value
    min_val: float
    max_val: float
    c_end: float = 4.0  # Final perturbation size
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
# Parameter Definition (Modify these for Luminex)
# ============================================================
def get_default_params() -> List[SPSAParam]:
    return [
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
# SPSA Logic Engine
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
            p.theta_plus = max(p.min_val, min(p.theta + p.c_k * p.delta, p.max_val))
            p.theta_minus = max(p.min_val, min(p.theta - p.c_k * p.delta, p.max_val))

    def update(self, result: float):
        for p in self.params:
            if p.c_k == 0 or p.delta == 0: continue
            gradient = result / (2.0 * p.c_k * p.delta)
            p.theta = max(p.min_val, min(p.theta + p.R_k * p.c_k * gradient, p.max_val))
            p.history.append(p.theta)
        self.iteration += 1

    def uci_opts(self, use_plus: bool) -> list:
        opts = []
        for p in self.params:
            val = int(round(p.theta_plus if use_plus else p.theta_minus))
            opts.append(f"option.{p.name}={val}")
        opts.append("option.Threads=1") # Force single-thread per game
        return opts

    def save(self, path: str):
        state = {
            'iteration': self.iteration,
            'params': [{'name': p.name, 'theta': p.theta, 'initial': p.initial} for p in self.params]
        }
        with open(path, 'w') as f: json.dump(state, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            state = json.load(f)
            self.iteration = state['iteration']
            for p, s in zip(self.params, state['params']): p.theta = s['theta']

# ============================================================
# Match Execution
# ============================================================
def run_games(cutechess: str, engine: str, opts_plus: list, opts_minus: list,
             rounds: int, tc: str, concurrency: int) -> Tuple[int, int, int]:
    cmd = [
        cutechess, "-rounds", str(rounds),
        "-engine", "cmd=" + engine, "name=LuminexPlus", *opts_plus,
        "-engine", "cmd=" + engine, "name=LuminexMinus", *opts_minus,
        "-each", "proto=uci", "tc=" + tc,
        "-concurrency", str(concurrency), "-recover"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        m = re.search(r'Score of \w+ vs \w+:\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)', proc.stdout + proc.stderr)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)
    except Exception as e:
        print(f"Match Error: {e}")
        return 0, 0, 0

# ============================================================
# Main Entry
# ============================================================
def main():
    detected_cores = os.cpu_count() or 1
    
    ap = argparse.ArgumentParser(description="Luminex SPSA Tuner")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--cutechess", default="./cutechess-cli")
    ap.add_argument("--iterations", type=int, default=50000)
    ap.add_argument("--rounds", type=int, default=detected_cores, help=f"Games per iter (Detected cores: {detected_cores})")
    ap.add_argument("--concurrency", type=int, default=detected_cores)
    ap.add_argument("--tc", default="1+0.01")
    ap.add_argument("--output", default="spsa_result.json")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    params = get_default_params()
    tuner = SPSATuner(params, {'iterations': args.iterations})

    if args.resume and os.path.exists(args.resume):
        tuner.load(args.resume)
        print(f"Resuming from iteration {tuner.iteration}...")

    print(f"Starting SPSA Tuner on {args.concurrency} cores...")
    start_time = time.time()

    while tuner.iteration < args.iterations:
        tuner.perturb()
        w, l, d = run_games(args.cutechess, args.engine, tuner.uci_opts(True), 
                           tuner.uci_opts(False), args.rounds, args.tc, args.concurrency)

        total = w + l + d
        result = (w - l) / total if total > 0 else 0.0
        tuner.update(result)

        if tuner.iteration % 1 == 0:
            elapsed = time.time() - start_time
            ips = (tuner.iteration / elapsed) * 3600 if elapsed > 0 else 0
            print(f"[{tuner.iteration:5d}/{args.iterations}] W:{w} L:{l} D:{d} res:{result:+.2f} ({ips:.1f} iters/hr)")
            tuner.save(args.output)

        if tuner.iteration % 100 == 0:
            print("\n  Parameter Updates:")
            for p in tuner.params:
                print(f"    {p.name:25s}: {p.theta:6.1f} (chg:{p.theta - p.initial:+.1f})")
            print()

if __name__ == "__main__":
    main()
