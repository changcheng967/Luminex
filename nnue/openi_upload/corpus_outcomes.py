#!/usr/bin/env python3
"""Corpus -> outcome rows for the logistic (Texel/Grant) solve.

Every position of every game, labeled with the final result from WHITE's
perspective. Output: fen \t z (0.0 / 0.5 / 1.0). The trainer computes
features itself; no engine evals needed here.
"""
import sys
import glob
import chess
import chess.pgn

def main(pattern, out):
    n = 0
    games = 0
    with open(out, "w") as fo:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8", errors="ignore") as fh:
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break
                    res = game.headers.get("Result", "*")
                    if res == "1-0":
                        z = "1.0"
                    elif res == "0-1":
                        z = "0.0"
                    elif res == "1/2-1/2":
                        z = "0.5"
                    else:
                        continue
                    games += 1
                    board = game.board()
                    for mv in game.mainline_moves():
                        fo.write(f"{board.fen()}\t{z}\n")
                        board.push(mv)
                    n += 1 if False else 0
                    if games % 2000 == 0:
                        print(f"{games} games...", flush=True)
    print(f"games={games} -> {out}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
