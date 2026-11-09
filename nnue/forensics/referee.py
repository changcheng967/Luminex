#!/usr/bin/env python3
"""Score every position of a set of games with a referee engine (Stockfish 19,
UCI_ShowWDL on), so each move's expected-score loss can be read off.

usage: referee.py OUT.jsonl SF_PATH NODES WORKERS PGN [PGN ...]

One JSON line per position, before each move and once after the last move:
  gid, ply, fen, stm, mover, played (uci, None at the end), white, black,
  result, term (None, or the side-to-move's expected score in a terminal
  position: 0 mated, 0.5 stalemate / rule draw), and sf = {d, cp, mate, wdl,
  es (expected score of the side to move), best, pv, nodes}.
Games are handed out whole to the workers (one engine each, ucinewgame per
game, Hash kept along the game). Restartable: games already in OUT are skipped.
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys

import chess
import chess.pgn


class UCI:
    def __init__(self, path, opts):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        for k, v in opts.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            line = line.strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def go(self, fen, nodes, moves=None):
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}" + (" searchmoves " + " ".join(moves) if moves else ""))
        lines = self.wait("bestmove")
        info = None
        for line in lines:
            if line.startswith("info") and " score " in line and " pv " in line:
                info = line
        r = parse_info(info) if info else {}
        r["best"] = lines[-1].split()[1]
        return r


def parse_info(line):
    t = line.split()
    r = {"d": None, "cp": None, "mate": None, "wdl": None, "nodes": None, "pv": []}
    i = 0
    while i < len(t):
        k = t[i]
        if k == "depth":
            r["d"] = int(t[i + 1]); i += 2
        elif k == "score":
            if t[i + 1] == "cp":
                r["cp"] = int(t[i + 2])
            else:
                r["mate"] = int(t[i + 2])
            i += 3
            while i < len(t) and t[i] in ("lowerbound", "upperbound"):
                i += 1
        elif k == "wdl":
            r["wdl"] = [int(t[i + 1]), int(t[i + 2]), int(t[i + 3])]; i += 4
        elif k == "nodes":
            r["nodes"] = int(t[i + 1]); i += 2
        elif k == "pv":
            r["pv"] = t[i + 1:i + 13]; break
        else:
            i += 1
    if r["wdl"]:
        w, d, l = r["wdl"]
        r["es"] = (w + 0.5 * d) / max(1, w + d + l)
    elif r["mate"] is not None:
        r["es"] = 1.0 if r["mate"] > 0 else 0.0
    return r


_eng = None
_cfg = None


def _init(cfg):
    global _eng, _cfg
    _cfg = cfg
    _eng = UCI(cfg["sf"], {"Threads": 1, "Hash": 64, "UCI_ShowWDL": "true"})


def terminal_es(board, result):
    if board.is_checkmate():
        return 0.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    if result == "1/2-1/2" and (board.can_claim_draw() or board.halfmove_clock >= 100):
        return 0.5
    return None


def run_game(task):
    gid, hdr, start_fen, moves = task
    _eng.send("ucinewgame")
    _eng.send("isready")
    _eng.wait("readyok")
    board = chess.Board(start_fen)
    out = []
    for k in range(len(moves) + 1):
        fen = board.fen()
        rec = {"gid": gid, "ply": board.ply(), "fen": fen, "stm": "w" if board.turn else "b",
               "mover": hdr["White"] if board.turn else hdr["Black"],
               "played": moves[k] if k < len(moves) else None,
               "white": hdr["White"], "black": hdr["Black"], "result": hdr["Result"], "term": None}
        if k == len(moves):
            rec["term"] = terminal_es(board, hdr["Result"])
        if rec["term"] is None and not board.is_game_over(claim_draw=False):
            rec["sf"] = _eng.go(fen, _cfg["nodes"])
        elif rec["term"] is None:
            rec["term"] = terminal_es(board, hdr["Result"])
        out.append(rec)
        if k < len(moves):
            board.push_uci(moves[k])
    return out


def load_tasks(pgns, done):
    tasks = []
    for path in pgns:
        tag = os.path.basename(path).rsplit(".", 1)[0]
        with open(path) as f:
            idx = 0
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                gid = f"{tag}:{idx}"
                idx += 1
                if gid in done:
                    continue
                hdr = {k: g.headers.get(k, "?") for k in ("White", "Black", "Result")}
                moves = [m.uci() for m in g.mainline_moves()]
                if not moves:
                    continue
                tasks.append((gid, hdr, g.board().fen(), moves))
    return tasks


def main():
    out_path, sf, nodes, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    pgns = sys.argv[5:]
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["gid"])
                except Exception:
                    pass
    tasks = load_tasks(pgns, done)
    print(f"{len(tasks)} games to referee ({len(done)} already done), {nodes} nodes, {workers} workers",
          file=sys.stderr, flush=True)
    n = 0
    with mp.Pool(workers, initializer=_init, initargs=({"sf": sf, "nodes": nodes},)) as pool, \
            open(out_path, "a") as fo:
        for recs in pool.imap_unordered(run_game, tasks):
            for r in recs:
                fo.write(json.dumps(r) + "\n")
            fo.flush()
            n += 1
            if n % 10 == 0:
                print(f"{n}/{len(tasks)} games", file=sys.stderr, flush=True)
    print("REFEREE-DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
