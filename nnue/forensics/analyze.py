#!/usr/bin/env python3
"""Weakness profile from refereed games (referee.py output).

usage: analyze.py REF.jsonl ENGINE_NAME ERRORS_OUT.jsonl [MIN_LOSS]

Per move: expected-score loss = es(before, mover) - (1 - es(after, opponent)),
es from the referee's WDL (Stockfish's own model). A second loss uses a
logistic on the referee's cp fitted to these games' actual results, so the
profile can be checked on our level of play instead of Stockfish's.
Each move gets position features (phase, material, king safety, passers,
hanging pieces, tactical refutation / missed tactic along the referee PVs).
Prints the profile for ENGINE_NAME and for its opponents (same games, same
referee), and writes ENGINE_NAME's moves with loss >= MIN_LOSS (default 0.05)
to ERRORS_OUT for replay attribution.
"""
import collections
import json
import math
import sys

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board, color):
    return sum(VAL[p] * len(board.pieces(p, color)) for p in VAL)


def mat_diff(board, color):
    return material(board, color) - material(board, not color)


def line_material(fen, pv, color, max_plies=6):
    """Material change (color's POV) along the first even number of PV plies <= max_plies."""
    b = chess.Board(fen)
    start = mat_diff(b, color)
    n = min(len(pv), max_plies)
    n -= n % 2
    for mv in pv[:n]:
        try:
            m = chess.Move.from_uci(mv)
            if m not in b.legal_moves:
                break
            b.push(m)
        except Exception:
            break
    return mat_diff(b, color) - start


def passed_pawns(board, color):
    out = []
    opp = board.pieces(chess.PAWN, not color)
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for o in opp:
            of, orr = chess.square_file(o), chess.square_rank(o)
            if abs(of - f) <= 1 and ((orr > r) if color == chess.WHITE else (orr < r)):
                blocked = True
                break
        if not blocked:
            out.append(r if color == chess.WHITE else 7 - r)
    return out


def king_features(board, color):
    ks = board.king(color)
    if ks is None:
        return 0, 0
    zone = list(chess.SquareSet(chess.BB_KING_ATTACKS[ks])) + [ks]
    attacks = sum(len(board.attackers(not color, s)) for s in zone)
    kf, kr = chess.square_file(ks), chess.square_rank(ks)
    shield = 0
    for p in board.pieces(chess.PAWN, color):
        pf, pr = chess.square_file(p), chess.square_rank(p)
        dr = (pr - kr) if color == chess.WHITE else (kr - pr)
        if abs(pf - kf) <= 1 and 1 <= dr <= 2:
            shield += 1
    return attacks, shield


def hanging(board, color):
    n = 0
    for sq, pc in board.piece_map().items():
        if pc.color != color or pc.piece_type == chess.KING:
            continue
        att = board.attackers(not color, sq)
        if not att:
            continue
        defs = board.attackers(color, sq)
        min_att = min(VAL[board.piece_at(a).piece_type] or 100 for a in att)
        if not defs or min_att < VAL[pc.piece_type]:
            n += 1
    return n


def eg_type(board):
    q = len(board.pieces(chess.QUEEN, True)) + len(board.pieces(chess.QUEEN, False))
    r = len(board.pieces(chess.ROOK, True)) + len(board.pieces(chess.ROOK, False))
    m = sum(len(board.pieces(p, c)) for p in (chess.KNIGHT, chess.BISHOP) for c in (True, False))
    if q:
        return "Q" + ("+R" if r else "") + ("+m" if m else "")
    if r and m:
        return "R+m"
    if r:
        return "R"
    if m:
        return "m"
    return "P"


def features(rec, nxt):
    b = chess.Board(rec["fen"])
    me = b.turn
    mv = chess.Move.from_uci(rec["played"])
    best = rec["sf"].get("best")
    bm = chess.Move.from_uci(best) if best and best != "(none)" else None
    npcs = len(b.piece_map())
    f = {"pieces": npcs, "bucket": (npcs - 1) * 8 // 32, "ply": rec["ply"],
         "queens": len(b.pieces(chess.QUEEN, True)) + len(b.pieces(chess.QUEEN, False)),
         "mat": mat_diff(b, me), "check": b.is_check(), "legal": b.legal_moves.count(),
         "p_capture": b.is_capture(mv), "p_check": b.gives_check(mv),
         "p_piece": chess.piece_symbol(b.piece_at(mv.from_square).piece_type),
         "b_capture": bool(bm) and b.is_capture(bm), "b_check": bool(bm) and b.gives_check(bm),
         "same": bm == mv, "eg": eg_type(b) if npcs <= 12 else "-"}
    f["k_att_me"], f["k_shield_me"] = king_features(b, me)
    f["k_att_opp"], f["k_shield_opp"] = king_features(b, not me)
    pm, po = passed_pawns(b, me), passed_pawns(b, not me)
    f["pass_me"], f["pass_opp"] = len(pm), len(po)
    f["pass_adv"] = max(pm + po) if (pm or po) else 0
    f["hang_me"] = hanging(b, me)
    f["best_line_mat"] = line_material(rec["fen"], rec["sf"].get("pv", []), me)
    b.push(mv)
    f["hang_after"] = hanging(b, me)
    if nxt is not None and nxt.get("sf"):
        f["refute_mat"] = line_material(nxt["fen"], nxt["sf"].get("pv", []), me)
    else:
        f["refute_mat"] = 0
    if f["refute_mat"] <= -1.5:
        f["kind"] = "tactic_allowed"
    elif f["best_line_mat"] >= 1.5:
        f["kind"] = "tactic_missed"
    else:
        f["kind"] = "positional"
    return f


def es_cp(sf, k):
    if sf.get("mate") is not None:
        return 1.0 if sf["mate"] > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-sf["cp"] / k))


def fit_k(games):
    """Logistic scale k: es = 1/(1+exp(-cp/k)) fitted to the games' results (mover POV)."""
    pts = []
    for recs in games.values():
        res = recs[0]["result"]
        sw = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
        if sw is None:
            continue
        for r in recs:
            sf = r.get("sf")
            if not sf or sf.get("cp") is None or r["ply"] < 16:
                continue
            s = sw if r["stm"] == "w" else 1 - sw
            pts.append((max(-1500, min(1500, sf["cp"])), s))
    best = None
    for k in range(60, 801, 10):
        ll = 0.0
        for cp, s in pts:
            p = min(1 - 1e-6, max(1e-6, 1 / (1 + math.exp(-cp / k))))
            ll += s * math.log(p) + (1 - s) * math.log(1 - p)
        if best is None or ll > best[1]:
            best = (k, ll)
    return best[0], len(pts)


def main():
    ref, me_name, err_out = sys.argv[1], sys.argv[2], sys.argv[3]
    min_loss = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    games = collections.defaultdict(list)
    for line in open(ref):
        r = json.loads(line)
        games[r["gid"]].append(r)
    for g in games.values():
        g.sort(key=lambda r: r["ply"])
    k, npts = fit_k(games)
    print(f"{len(games)} games; result-fitted logistic k = {k} cp ({npts} positions)")

    moves = []
    for gid, recs in games.items():
        for i, r in enumerate(recs[:-1]):
            nxt = recs[i + 1]
            if not r.get("sf") or r.get("played") is None or nxt["ply"] != r["ply"] + 1:
                continue
            if nxt.get("sf") and "es" in nxt["sf"]:
                after = 1 - nxt["sf"]["es"]
                after_k = 1 - es_cp(nxt["sf"], k)
            elif nxt.get("term") is not None:
                after = after_k = 1 - nxt["term"]
            else:
                continue
            if "es" not in r["sf"]:
                continue
            m = {"gid": gid, "ply": r["ply"], "fen": r["fen"], "mover": r["mover"],
                 "played": r["played"], "best": r["sf"].get("best"),
                 "es_before": r["sf"]["es"], "loss": r["sf"]["es"] - after,
                 "loss_k": es_cp(r["sf"], k) - after_k, "cp_before": r["sf"].get("cp"),
                 "pv_before": r["sf"].get("pv", []), "pv_after": (nxt.get("sf") or {}).get("pv", []),
                 "result": r["result"], "white": r["white"], "black": r["black"]}
            m["f"] = features(r, nxt)
            moves.append(m)

    def mine(m):
        return m["mover"] == me_name

    for who, sel in ((me_name, mine), ("opponents", lambda m: not mine(m))):
        ms = [m for m in moves if sel(m)]
        ng = len({m["gid"] for m in ms})
        tot = sum(max(0, m["loss"]) for m in ms)
        totk = sum(max(0, m["loss_k"]) for m in ms)
        print(f"\n=== {who}: {len(ms)} moves in {ng} games; loss per game {tot / ng:.3f} (WDL) {totk / ng:.3f} (fitted k)")
        for th in (0.05, 0.10, 0.20):
            c = sum(1 for m in ms if m["loss"] >= th)
            print(f"  moves with loss >= {th:.2f}: {c} ({c / ng:.2f} per game)")
        report(ms, ng)

    errs = [m for m in moves if mine(m) and m["loss"] >= min_loss]
    with open(err_out, "w") as fo:
        for m in errs:
            fo.write(json.dumps(m) + "\n")
    print(f"\nwrote {len(errs)} {me_name} moves with loss >= {min_loss} to {err_out}")


def report(ms, ng):
    def table(title, keyf, order=None):
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
        for m in ms:
            key = keyf(m)
            a = agg[key]
            a[0] += 1
            a[1] += max(0, m["loss"])
            a[2] += max(0, m["loss_k"])
            a[3] += m["loss"] >= 0.10
        tot = sum(a[1] for a in agg.values()) or 1
        print(f"  -- {title}: key | moves | loss/game (share) | loss/move | fitted-k loss/game | errors>=0.1/game")
        keys = order if order else sorted(agg, key=lambda x: -agg[x][1])
        for key in keys:
            if key not in agg:
                continue
            a = agg[key]
            print(f"     {str(key):>16} | {a[0]:6d} | {a[1] / ng:.4f} ({100 * a[1] / tot:4.1f}%) | "
                  f"{a[1] / a[0]:.4f} | {a[2] / ng:.4f} | {a[3] / ng:.3f}")

    def regime(m):
        e = m["es_before"]
        return "losing<0.15" if e < 0.15 else ("winning>0.85" if e > 0.85 else "balanced")

    table("phase (pieces bucket 0=2-4 .. 7=29-32)", lambda m: m["f"]["bucket"], order=list(range(8)))
    table("regime before the move", regime, order=["losing<0.15", "balanced", "winning>0.85"])
    table("kind", lambda m: m["f"]["kind"])
    table("kind x phase3", lambda m: (m["f"]["kind"], "eg" if m["f"]["pieces"] <= 12 else ("mg" if m["f"]["pieces"] <= 26 else "op")))
    table("queens on board", lambda m: m["f"]["queens"])
    table("attack on own king zone", lambda m: min(m["f"]["k_att_me"], 6))
    table("attack on opp king zone", lambda m: min(m["f"]["k_att_opp"], 6))
    table("most advanced passer (rel. rank, 0 = none)", lambda m: m["f"]["pass_adv"])
    table("own hanging pieces before", lambda m: min(m["f"]["hang_me"], 3))
    table("in check", lambda m: m["f"]["check"])
    table("endgame type (<=12 pieces)", lambda m: m["f"]["eg"])
    table("material (mover POV, clipped)", lambda m: max(-4, min(4, m["f"]["mat"])))
    table("piece moved", lambda m: m["f"]["p_piece"])


if __name__ == "__main__":
    main()
