#!/usr/bin/env python3
"""Dump pre-swing FENs from tactic-class losses (see analyze_gap.py classes).

For each loss where material swung >= 5 within <= 4 plies against us,
emit: the FEN at the ply BEFORE the swing starts, the move played in the
game, our material edge before, the swing size, ply, phase.
Output: one line per position, tab-separated, ready for a fixed-depth
engine showdown.
"""
import sys
import chess
import chess.pgn

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

def mat_diff(board):
    return sum(v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
               for pt, v in VAL.items())

def phase(board):
    npm = sum(v * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
              for pt, v in VAL.items() if pt != chess.PAWN)
    return "MG" if npm >= 20 else ("EG-lite" if npm >= 8 else "EG")

def main(path, our_name="luminex", out="swing_fens.txt"):
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as fh, open(out, "w") as fo:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            h = game.headers
            w, b = h.get("White", ""), h.get("Black", "")
            if our_name in w.lower():
                sign, res = 1, h.get("Result", "*")
            elif our_name in b.lower():
                sign, res = -1, h.get("Result", "*")
            else:
                continue
            if res not in ("1-0", "0-1", "1/2-1/2"):
                continue
            white_won = res == "1-0"
            ours = 1.0 if (white_won == (sign == 1)) else (0.5 if res == "1/2-1/2" else 0.0)
            if ours != 0.0:
                continue
            board = game.board()
            states = []  # (fen, uci, my_mat, phase)
            for mv in game.mainline_moves():
                states.append((board.fen(), mv.uci(), sign * mat_diff(board), phase(board)))
                board.push(mv)
            states.append((board.fen(), "", sign * mat_diff(board), phase(board)))
            N = len(states) - 1
            best_sw, best_i, span = 0, -1, 99
            md = [sign * mat_diff(game.board())] * 0  # unused
            mats = [s[2] for s in states]
            for i in range(N):
                for j in range(i + 1, min(i + 5, N + 1)):
                    sw = mats[i] - mats[j]
                    if sw > best_sw:
                        best_sw, best_i, span = sw, i, j - i
            if best_sw >= 5 and span <= 4 and not any(m <= -4 for m in mats[:min(20, N)]):
                s = states[best_i]
                fo.write(f"{s[0]}\t{s[1]}\t{best_sw}\t{s[3]}\t{best_i + 1}\n")
                n += 1
    print(f"wrote {n} pre-swing loss positions to {out}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "luminex",
         sys.argv[3] if len(sys.argv) > 3 else "swing_fens.txt")
