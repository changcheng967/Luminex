"""Engine's own search score (PGN comments) against the referee's score, by referee bin.

usage: scale_games.py REF.jsonl PGN ENGINE_NAME [ENGINE_NAME ...]
For each move by a named engine: its search score (mover's POV, cp) from the
PGN comment and the referee's score of the same position (mover's POV)."""
import collections
import json
import os
import sys

import chess.pgn

ref_path, pgn_path, names = sys.argv[1], sys.argv[2], sys.argv[3:]
tag = os.path.basename(pgn_path).rsplit(".", 1)[0]
ref = {}
for line in open(ref_path):
    r = json.loads(line)
    sf = r.get("sf")
    if sf and sf.get("cp") is not None:
        ref[(r["gid"], r["ply"])] = sf["cp"]
rows = collections.defaultdict(list)
with open(pgn_path) as f:
    idx = 0
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        gid = f"{tag}:{idx}"
        idx += 1
        board = g.board()
        for node in g.mainline():
            mover = g.headers["White"] if board.turn else g.headers["Black"]
            ply = board.ply()
            c = node.comment.strip()
            if mover in names and c and "/" in c and (gid, ply) in ref:
                s = c.split("/")[0]
                if not s.startswith(("+M", "-M", "M")):
                    try:
                        rows[mover].append((ref[(gid, ply)], float(s) * 100))
                    except ValueError:
                        pass
            board.push(node.move)
bins = [(-10000, -600), (-600, -300), (-300, -150), (-150, -50), (-50, 50), (50, 150), (150, 300), (300, 600), (600, 10000)]
for name in names:
    pts = rows[name]
    print(f"{name}: {len(pts)} moves with both scores")
    print(f"  {'referee cp (mover)':>20} {'n':>6} {'ref mean':>9} {'engine mean':>12} {'engine/ref':>11} {'corr in bin':>12}")
    for lo, hi in bins:
        b = [(x, y) for x, y in pts if lo <= x < hi]
        if len(b) < 30:
            continue
        mx = sum(x for x, _ in b) / len(b)
        my = sum(y for _, y in b) / len(b)
        sx = (sum((x - mx) ** 2 for x, _ in b) / len(b)) ** 0.5
        sy = (sum((y - my) ** 2 for _, y in b) / len(b)) ** 0.5
        cov = sum((x - mx) * (y - my) for x, y in b) / len(b)
        corr = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
        print(f"  {f'[{lo}, {hi})':>20} {len(b):6d} {mx:9.0f} {my:12.0f} {my / mx if abs(mx) > 1e-9 else float('nan'):11.2f} {corr:12.2f}")
