#!/usr/bin/env python3
"""Attribute each mistake (analyze.py's error list) to search, eval or labels.

usage:
  replay.py lux ERRORS.jsonl OUT_LUX.jsonl LUMINEX NET WORKERS
      our engine, cold (ucinewgame before each search): the move at 1x, 8x and
      64x the game's 100K nodes; our static eval and our depth-1 (qsearch-
      resolved) score after the played move and after the referee's best move.
  replay.py sf ERRORS.jsonl OUT_LUX.jsonl OUT_SF.jsonl STOCKFISH NODES WORKERS
      Stockfish 19 scores every candidate move from the same root at NODES
      (go nodes N searchmoves m), plus its static eval and a depth-9 search
      (the label proxy) after the played and the best move.
All scores are from the mover's point of view. One engine process per worker.
"""
import json
import multiprocessing as mp
import sys

from referee import UCI


def lux_task(e):
    eng = _G["eng"]
    out = {"id": e["id"]}
    for tag, n in (("r1", 100000), ("r8", 800000), ("r64", 6400000)):
        eng.send("ucinewgame")
        eng.send("isready")
        eng.wait("readyok")
        r = eng.go(e["fen"], n)
        out[tag] = r["best"]
        out[tag + "_cp"] = r.get("cp")
        out[tag + "_d"] = r.get("d")
    for tag, mv in (("played", e["played"]), ("best", e["best"])):
        eng.send(f"position fen {e['fen']} moves {mv}")
        eng.send("eval")
        lines = eng.wait("eval cp")
        out["s_" + tag] = -int(lines[-1].split()[2])
        eng.send("ucinewgame")
        eng.send(f"position fen {e['fen']} moves {mv}")
        eng.send("go depth 1")
        lines = eng.wait("bestmove")
        sc = None
        for line in lines:
            if line.startswith("info") and " score cp " in line:
                sc = int(line.split(" score cp ")[1].split()[0])
            elif line.startswith("info") and " score mate " in line:
                m = int(line.split(" score mate ")[1].split()[0])
                sc = 30000 if m > 0 else -30000
        out["q_" + tag] = -sc if sc is not None else None
    return out


def sf_static(eng, fen, mv):
    eng.send(f"position fen {fen} moves {mv}")
    eng.send("eval")
    eng.send("isready")
    lines = eng.wait("readyok")
    for line in lines:
        if line.startswith("Final evaluation"):
            tok = line.split()[2]
            try:
                return float(tok)
            except ValueError:
                return None
    return None


def sf_task(item):
    e, lux = item
    eng = _G["eng"]
    nodes = _G["nodes"]
    out = {"id": e["id"]}
    eng.send("ucinewgame")
    r = eng.go(e["fen"], nodes)
    out["es_best"], out["best"] = r.get("es"), r["best"]
    cands = {e["played"], e["best"], lux["r1"], lux["r8"], lux["r64"]}
    out["es_move"] = {}
    for m in cands:
        if m == r["best"]:
            out["es_move"][m] = r.get("es")
            continue
        eng.send("ucinewgame")
        rm = eng.go(e["fen"], nodes, moves=[m])
        out["es_move"][m] = rm.get("es")
    # static eval and depth-9 search after the played and after the referee's best move
    b = chess.Board(e["fen"])
    white_mover = b.turn
    for tag, mv in (("played", e["played"]), ("best", e["best"])):
        st = sf_static(eng, e["fen"], mv)
        out["sfs_" + tag] = None if st is None else (st if white_mover else -st) * 100
        eng.send("ucinewgame")
        eng.send(f"position fen {e['fen']} moves {mv}")
        eng.send("go depth 9")
        lines = eng.wait("bestmove")
        info = None
        for line in lines:
            if line.startswith("info") and " score " in line and " pv " in line:
                info = line
        from referee import parse_info
        p = parse_info(info) if info else {}
        out["d9_" + tag] = (1 - p["es"]) if "es" in p else None
    return out


_G = {}


def _init_lux(path, net):
    _G["eng"] = UCI(path, {"Threads": 1, "Hash": 32, "UseNNUE": "true", "NNUEFile": net})


def _init_sf(path, nodes):
    _G["eng"] = UCI(path, {"Threads": 1, "Hash": 64, "UCI_ShowWDL": "true"})
    _G["nodes"] = nodes


def load_errors(path):
    errs = []
    for i, line in enumerate(open(path)):
        e = json.loads(line)
        e["id"] = i
        errs.append(e)
    return errs


import chess  # noqa: E402  (after the worker helpers so referee's import order is kept)


def main():
    mode = sys.argv[1]
    errs = load_errors(sys.argv[2])
    if mode == "lux":
        out_path, lux, net, workers = sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
        with mp.Pool(workers, initializer=_init_lux, initargs=(lux, net)) as pool, open(out_path, "w") as fo:
            for k, r in enumerate(pool.imap_unordered(lux_task, errs)):
                fo.write(json.dumps(r) + "\n")
                fo.flush()
                if k % 100 == 0:
                    print(f"lux {k}/{len(errs)}", file=sys.stderr, flush=True)
    else:
        lux_path, out_path, sf, nodes, workers = sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
        lux = {}
        for line in open(lux_path):
            r = json.loads(line)
            lux[r["id"]] = r
        items = [(e, lux[e["id"]]) for e in errs if e["id"] in lux]
        with mp.Pool(workers, initializer=_init_sf, initargs=(sf, nodes)) as pool, open(out_path, "w") as fo:
            for k, r in enumerate(pool.imap_unordered(sf_task, items)):
                fo.write(json.dumps(r) + "\n")
                fo.flush()
                if k % 100 == 0:
                    print(f"sf {k}/{len(items)}", file=sys.stderr, flush=True)
    print("REPLAY-DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
