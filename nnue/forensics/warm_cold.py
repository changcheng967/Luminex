#!/usr/bin/env python3
"""Is the game's (warm) move worse than a cold re-search at the same nodes?

usage: warm_cold.py REF.jsonl ENGINE_NAME LUMINEX NET STOCKFISH N OUT.jsonl
Samples N of ENGINE_NAME's moves at random (not only mistakes), re-searches
each cold (ucinewgame) at the game's 100K nodes, and where the cold move
differs, has Stockfish score both from the same root (300K nodes,
searchmoves). One engine process alive at a time."""
import json
import random
import sys

import chess

from referee import UCI

ref, name, lux_path, net, sf_path, n, out = sys.argv[1:8]
n = int(n)
random.seed(7)
moves = []
for line in open(ref):
    r = json.loads(line)
    if r.get("mover") == name and r.get("played") and r.get("sf") and r["ply"] >= 8:
        moves.append(r)
sample = random.sample(moves, min(n, len(moves)))
lux = UCI(lux_path, {"Threads": 1, "Hash": 32, "UseNNUE": "true", "NNUEFile": net})
for r in sample:
    lux.send("ucinewgame")
    lux.send("isready")
    lux.wait("readyok")
    r["cold"] = lux.go(r["fen"], 100000)["best"]
lux.send("quit")
diff = [r for r in sample if r["cold"] != r["played"]]
sf = UCI(sf_path, {"Threads": 1, "Hash": 64, "UCI_ShowWDL": "true"})
with open(out, "w") as fo:
    for r in diff:
        es = {}
        for m in (r["played"], r["cold"]):
            sf.send("ucinewgame")
            es[m] = sf.go(r["fen"], 300000, moves=[m]).get("es")
        fo.write(json.dumps({"fen": r["fen"], "played": r["played"], "cold": r["cold"],
                             "es_played": es[r["played"]], "es_cold": es[r["cold"]]}) + "\n")
        fo.flush()
pairs = [json.loads(l) for l in open(out)]
pairs = [p for p in pairs if p["es_played"] is not None and p["es_cold"] is not None]
d = [p["es_cold"] - p["es_played"] for p in pairs]
mean = sum(d) / len(d) if d else 0.0
sd = (sum((x - mean) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5
print(f"{len(sample)} random moves; cold move differs on {len(diff)} ({100 * len(diff) / len(sample):.1f}%)")
print(f"where they differ: cold minus warm expected score {mean:+.4f} +- {1.96 * sd / max(1, len(d)) ** 0.5:.4f} "
      f"(95%); cold better on {sum(1 for x in d if x > 0.01)}, warm better on {sum(1 for x in d if x < -0.01)}")
