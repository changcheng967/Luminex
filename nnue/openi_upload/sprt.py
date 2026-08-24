#!/usr/bin/env python3
"""SPRT batch harness: sequential probability ratio test over cutechess batches.

Runs candidate-vs-baseline matches in batches, computes the fishtest-style
SPRT LLR after each batch (elo0=0, elo1=10 default), stops early when
LLR >= +2.94 (accept H1: candidate gains >= elo1) or <= -2.94 (accept H0).

Usage: python sprt.py <cand_path> <base_path> <name> [elo0 elo1 max_games batch]
Uses bayeselo score model: draw ratio from data, ties=rasid.
"""
import math
import subprocess
import sys
import os

CUTE = "/hyperai/home/cutechess/cutechess-cli"
OPEN = "/hyperai/home/openings.pgn"

def sprt_llr(w, l, d, elo0, elo1):
    """Gaussian SPRT on the mean per-game score (fishtest-style approx).

    score_i in {1, 0.5, 0}; mu(elo) = 0.5 + elo/800 (bayeselo-ish mapping).
    LLR = n * ((s - mu0)^2 - (s - mu1)^2) / (2 * Var_per_game).
    """
    n = w + l + d
    if n == 0:
        return 0.0
    s = (w + 0.5 * d) / n
    var = (w + 0.25 * d) / n - s * s
    if var <= 1e-9:
        var = 1e-9
    mu0, mu1 = 0.5 + elo0 / 800.0, 0.5 + elo1 / 800.0
    return n * ((s - mu0) ** 2 - (s - mu1) ** 2) / (2.0 * var)

def run_batch(cand, base, batch_games, seed_offset, log_path):
    ch = os.environ.get("CAND_HASH", "256")
    bh = os.environ.get("BASE_HASH", "256")
    cmd = [CUTE,
           "-engine", f"name=cand", f"cmd={cand}", "proto=uci", f"option.Hash={ch}",
           "-engine", f"name=base", f"cmd={base}", "proto=uci", f"option.Hash={bh}",
           "-each", "tc=1+0.01", "-openings", f"file={OPEN}", "format=pgn",
           f"-rounds", str(batch_games), "-repeat", "-concurrency", os.environ.get("SPRT_CONC", "4"),
           "-pgnout", log_path.replace(".log", f"_{seed_offset}.pgn")]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    txt = out.stdout + out.stderr
    # parse final score line
    w = l = d = 0
    for ln in txt.splitlines():
        if ln.startswith("Score of cand vs base:"):
            parts = ln.split(":")[1].strip().split("[")[0].strip()
            w, l, d = [int(x) for x in parts.split("-")]
    return w, l, d

def main():
    cand, base, name = sys.argv[1], sys.argv[2], sys.argv[3]
    elo0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    elo1 = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    max_games = int(sys.argv[6]) if len(sys.argv) > 6 else 20000
    batch = int(sys.argv[7]) if len(sys.argv) > 7 else 250
    conc = os.environ.get("SPRT_CONC", "4")
    W = L = D = 0
    n_done = 0
    log = f"/hyperai/home/sftest/sprt_{name}.log"
    # resume: parse cumulative W/L/D from an existing log's last stat line
    if os.path.exists(log):
        for ln in open(log):
            if ln.startswith("n="):
                import re
                m = re.match(r"n=(\d+) W/L/D=(\d+)/(\d+)/(\d+)", ln)
                if m:
                    n_done, W, L, D = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if n_done:
            print(f"resuming from n={n_done} W/L/D={W}/{L}/{D}", flush=True)
    with open(log, "a") as fh:
        if not n_done:
            fh.write(f"SPRT {name} elo0={elo0} elo1={elo1} max={max_games} conc={conc}\n")
    while n_done < max_games:
        w, l, d = run_batch(cand, base, batch, n_done // batch, log)
        if w + l + d == 0:
            # transient wedge: retry the same batch seed once before aborting
            w, l, d = run_batch(cand, base, batch, n_done // batch, log)
        if w + l + d == 0:
            with open(log, "a") as fh:
                fh.write(f"BATCH {n_done}: cutechess returned no score twice — abort\n")
            return
        W += w; L += l; D += d; n_done += w + l + d
        llr = sprt_llr(W, L, D, elo0, elo1)
        elo = 800.0 * ((W + 0.5 * D) / n_done - 0.5)
        line = (f"n={n_done} W/L/D={W}/{L}/{D} score={(W+0.5*D)/n_done:.3f} "
                f"elo~{elo:+.1f} LLR={llr:+.2f}")
        with open(log, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
        if llr >= 2.94:
            print(f"ACCEPT H1 (candidate >= +{elo1}) after {n_done} games")
            with open(log, "a") as fh:
                fh.write("ACCEPT_H1\n")
            return
        if llr <= -2.94:
            print(f"ACCEPT H0 (no gain) after {n_done} games")
            with open(log, "a") as fh:
                fh.write("ACCEPT_H0\n")
            return
    print(f"MAX GAMES reached: n={n_done}")

if __name__ == "__main__":
    main()
