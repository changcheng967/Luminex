#!/usr/bin/env python3
"""Gap analysis: attribute Luminex results vs a strong engine from a cutechess PGN.

Loss kinds (from Luminex's perspective):
  opening_disaster : material deficit >= 4 by ply 20
  tactic           : material swing against us >= 5 within <= 4 plies
  squeeze          : swing >= 5 but gradual (> 4 plies)
  endgame_fail     : entered endgame up/equal, then lost
  flag             : time forfeit
Win kinds mirror them. Draws: conversion_leak = up >= 4 in endgame but drew.
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

def classify(traj, sign, won):
    """traj: list of (ply, md, phase); sign: +1 white-perspective->ours if we are white."""
    my = lambda i: sign * traj[i][1]
    n = len(traj)
    key = "opening" if won else "opening_disaster"
    if n <= 2:
        return key
    early = any((my(i) >= 4) if won else (my(i) <= -4) for i in range(min(20, n)))
    best_swing, span = 0, 99
    for i in range(n):
        for j in range(i + 1, min(i + 5, n)):
            sw = (my(j) - my(i)) if won else (my(i) - my(j))
            if sw > best_swing:
                best_swing, span = sw, j - i
    eg_entry = next((i for i in range(n) if traj[i][2] != "MG"), n)
    if early:
        return key
    if best_swing >= 5 and span <= 4:
        return "tactic"
    if best_swing >= 5:
        return "squeeze"
    if won:
        return "endgame_convert" if eg_entry < n and my(eg_entry) <= 0 else "other"
    return "endgame_fail" if eg_entry < n and my(eg_entry) >= 0 else "other"

def main(path, our_name="luminex"):
    losses, wins_k = {}, {}
    flag_l = flag_w = draws = wins = loss_n = leak = 0
    loss_phase = {"MG": 0, "EG-lite": 0, "EG": 0}
    lens = {"w": [], "d": [], "l": []}
    n_games = 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
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
            n_games += 1
            if res == "1/2-1/2":
                ours = 0.5
            elif res != "*":
                white_won = res == "1-0"
                ours = 1.0 if (white_won == (sign == 1)) else 0.0
            else:
                ours = None
            term = h.get("Termination", "").lower()
            traj = []
            board = game.board()
            for mv in game.mainline_moves():
                board.push(mv)
                traj.append((len(traj) + 1, mat_diff(board), phase(board)))
            if ours is None:
                continue
            if ours == 0.5:
                draws += 1
                lens["d"].append(len(traj))
                if any(sign * traj[i][1] >= 4 and traj[i][2] != "MG" for i in range(len(traj))):
                    leak += 1
                continue
            if "time" in term or "forfeit" in term:
                if ours == 1.0:
                    flag_w += 1
                    wins += 1
                    lens["w"].append(len(traj))
                else:
                    flag_l += 1
                    loss_n += 1
                    lens["l"].append(len(traj))
                continue
            k = classify(traj, sign, ours == 1.0)
            if ours == 1.0:
                wins += 1
                wins_k[k] = wins_k.get(k, 0) + 1
                lens["w"].append(len(traj))
            else:
                loss_n += 1
                losses[k] = losses.get(k, 0) + 1
                lens["l"].append(len(traj))
                if traj:
                    loss_phase[traj[-1][2]] += 1
    tot = wins + draws + loss_n
    avg = lambda k: (sum(lens[k]) / len(lens[k])) if lens[k] else 0
    print(f"games={tot} (parsed {n_games})  W/D/L = {wins}/{draws}/{loss_n}  "
          f"score={((wins + 0.5 * draws) / tot) if tot else 0:.3f}")
    print(f"avg plies: win={avg('w'):.0f} draw={avg('d'):.0f} loss={avg('l'):.0f}")
    print(f"LOSS kinds: {losses}  (+flag {flag_l})")
    print(f"WIN  kinds: {wins_k}  (+flag {flag_w})")
    print(f"loss final phase: {loss_phase}")
    print(f"conversion leaks: {leak}/{draws} draws were up >=4 in endgame")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "luminex")
