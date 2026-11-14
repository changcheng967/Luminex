#!/usr/bin/env python3
"""Near-tie ordering: on pairs of moves both strong engines' searches see as close but separable,
how often does each static eval prefer the better child?

usage: near_tie.py WC.jsonl LUMINEX NET STOCKFISH [MIN_GAP]

WC.jsonl is warm_cold.py's output: positions where our cold re-search at the game's nodes picked
another move than the game (fen, played, cold, es_played, es_cold; es = Stockfish's expected score
for the mover after a 300K-node search restricted to that move). A pair counts when the two moves'
es differ by at least MIN_GAP (default 0.05). For each, our static eval and Stockfish's static eval
of both children, from the mover's point of view; a static eval "picks the better child" when its
order matches the search's. Ties count half. 95% intervals from a bootstrap over pairs."""
import json
import random
import sys

import chess

from referee import UCI
from replay import sf_static

wc, lux_path, net, sf_path = sys.argv[1:5]
gap = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
pairs = [json.loads(l) for l in open(wc)]
pairs = [p for p in pairs if p.get("es_played") is not None and p.get("es_cold") is not None
         and abs(p["es_cold"] - p["es_played"]) >= gap]
lux = UCI(lux_path, {"Threads": 1, "Hash": 16, "UseNNUE": "true", "NNUEFile": net})
sf = UCI(sf_path, {"Threads": 1, "Hash": 16})


def lux_static(fen, mv):
    lux.send(f"position fen {fen} moves {mv}")
    lux.send("eval")
    lines = lux.wait("eval cp")
    return -int(lines[-1].split()[2])     # printed for the side to move after the move: the mover's opponent


scores = []
for p in pairs:
    white = chess.Board(p["fen"]).turn == chess.WHITE
    better_cold = p["es_cold"] > p["es_played"]
    ours = (lux_static(p["fen"], p["cold"]), lux_static(p["fen"], p["played"]))
    s_c, s_p = (sf_static(sf, p["fen"], p["cold"]), sf_static(sf, p["fen"], p["played"]))
    if s_c is None or s_p is None:
        continue
    theirs = (s_c, s_p) if white else (-s_c, -s_p)   # Stockfish prints White's side

    def pick(a):
        if a[0] == a[1]:
            return 0.5
        return 1.0 if (a[0] > a[1]) == better_cold else 0.0
    scores.append((pick(ours), pick(theirs)))
lux.send("quit")
sf.send("quit")
n = len(scores)
if not n:
    sys.exit("no pairs")
random.seed(5)


def ci(k):
    vals = [s[k] for s in scores]
    boots = sorted(sum(random.choice(vals) for _ in range(n)) / n for _ in range(2000))
    return sum(vals) / n, boots[50], boots[1949]


o, t = ci(0), ci(1)
print(f"{n} near-tie pairs (search gap >= {gap}) from {len(pairs)} candidates")
print(f"  our static eval picks the better child  {o[0]:.1%}  (95%: {o[1]:.1%} .. {o[2]:.1%})")
print(f"  Stockfish's static eval                 {t[0]:.1%}  (95%: {t[1]:.1%} .. {t[2]:.1%})")
