#!/usr/bin/env python3
"""Extract our decision points from corpus PGNs for win-prob calibration.

Emits one line per sampled our-move position:
  fen \t game_id \t our_color \t result (1/0.5/0 from our side)
Sampled every 4th our-move ply to keep the eval pass cheap (~30K rows).
"""
import sys
import chess
import chess.pgn

def main(paths, out, every=4):
    n_rows = 0
    n_games = 0
    with open(out, "w") as fo:
        for path in paths:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                gid = 0
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break
                    gid += 1
                    h = game.headers
                    w, b = h.get("White", ""), h.get("Black", "")
                    if "luminex" in w.lower():
                        our_color, res = chess.WHITE, h.get("Result", "*")
                    elif "luminex" in b.lower():
                        our_color, res = chess.BLACK, h.get("Result", "*")
                    else:
                        continue
                    if res == "1/2-1/2":
                        wdl = 0.5
                    elif res == "*":
                        continue
                    else:
                        white_won = res == "1-0"
                        wdl = 1.0 if (white_won == (our_color == chess.WHITE)) else 0.0
                    board = game.board()
                    k = 0
                    for mv in game.mainline_moves():
                        if board.turn == our_color and (k % every == 0):
                            fo.write(f"{board.fen()}\t{path}:{gid}\t{0 if our_color == chess.WHITE else 1}\t{wdl}\n")
                            n_rows += 1
                        if board.turn == our_color:
                            k += 1
                        board.push(mv)
                    n_games += 1
    print(f"games={n_games} rows={n_rows} -> {out}")

if __name__ == "__main__":
    out = sys.argv[1]
    main(sys.argv[2:], out)
