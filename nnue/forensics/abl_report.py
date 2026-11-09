#!/usr/bin/env python3
"""Score the ablation runs' moves and report which switched-off rule fixes the cases.

usage:
  abl_report.py score CASES.jsonl ABL.jsonl STOCKFISH NODES WORKERS NEW.jsonl
      Stockfish scores every chosen move that the case file does not already
      score, from the same root (go nodes N searchmoves m). Restartable.
  abl_report.py report CASES.jsonl ABL.jsonl NEW.jsonl
      Per mask: cases fixed (chosen move loses < half the played move's loss,
      against Stockfish's best from the same root), their loss mass, and the
      mean loss of the chosen moves; then the cases the rule fixes that the
      unchanged search (mask 0) does not."""
import collections
import json
import multiprocessing as mp
import os
import sys

from referee import UCI

NAMES = {0: "none (current main)", 1: "RFP", 2: "razoring", 4: "null move", 8: "ProbCut",
         16: "history pruning", 32: "LMP", 64: "futility", 128: "quiet LMR", 256: "losing-capture LMR",
         512: "qsearch SEE pruning", 1024: "qsearch delta pruning", 2048: "IIR",
         4096: "singular multicut / negative ext", 8192: "hindsight depth", 16384: "check ext for all checks",
         32768: "eval-margin LMR", 65536: "capture-response LMR", 131072: "TT-capture LMR",
         262144: "not-improving LMR"}

_G = {}


def _init(sf, nodes):
    _G["eng"] = UCI(sf, {"Threads": 1, "Hash": 64, "UCI_ShowWDL": "true"})
    _G["nodes"] = nodes


def _score(item):
    cid, fen, mv = item
    eng = _G["eng"]
    eng.send("ucinewgame")
    r = eng.go(fen, _G["nodes"], moves=[mv])
    return {"id": cid, "move": mv, "es": r.get("es")}


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    mode = sys.argv[1]
    cases = {c["id"]: c for c in load(sys.argv[2])}
    abl = load(sys.argv[3])
    if mode == "score":
        sf, nodes, workers, new_path = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
        have = {(r["id"], r["move"]) for r in load(new_path)}
        todo = set()
        for r in abl:
            if r["id"] not in cases:
                continue
            c = cases[r["id"]]
            if r["move"] not in c["es_move"] and (r["id"], r["move"]) not in have:
                todo.add((r["id"], c["fen"], r["move"]))
        print(f"{len(todo)} new moves to score", file=sys.stderr, flush=True)
        with mp.Pool(workers, initializer=_init, initargs=(sf, nodes)) as pool, open(new_path, "a") as fo:
            for r in pool.imap_unordered(_score, sorted(todo)):
                fo.write(json.dumps(r) + "\n")
                fo.flush()
        print("SCORE-DONE", file=sys.stderr, flush=True)
        return
    new = {(r["id"], r["move"]): r["es"] for r in load(sys.argv[4])}

    def es(cid, mv):
        c = cases[cid]
        return c["es_move"].get(mv, new.get((cid, mv)))

    by_mask = collections.defaultdict(dict)
    for r in abl:
        if r["id"] in cases:
            by_mask[r["mask"]][r["id"]] = r["move"]
    base_fixed = set()
    rows = []
    for mask in sorted(by_mask):
        fixed, mass, losses, n = set(), 0.0, [], 0
        for cid, mv in by_mask[mask].items():
            c = cases[cid]
            e = es(cid, mv)
            if e is None:
                continue
            n += 1
            loss = max(0.0, c["es_best"] - e)
            losses.append(loss)
            if loss < 0.5 * c["loss"]:
                fixed.add(cid)
                mass += c["loss"]
        if mask == 0:
            base_fixed = fixed
        rows.append((mask, n, fixed, mass, sum(losses) / max(1, len(losses))))
    total = sum(c["loss"] for c in cases.values())
    print(f"{len(cases)} cases, total loss {total:.2f} (as played in the games)")
    print(f"{'rule switched off':>34} {'n':>4} {'fixed':>6} {'loss fixed':>11} {'mean loss now':>14} {'new vs mask 0':>14}")
    for mask, n, fixed, mass, ml in rows:
        extra = len(fixed - base_fixed) if mask else 0
        lost = len(base_fixed - fixed) if mask else 0
        print(f"{NAMES.get(mask, str(mask)):>34} {n:4d} {len(fixed):6d} {mass:11.2f} {ml:14.4f} {f'+{extra} / -{lost}':>14}")


if __name__ == "__main__":
    main()
