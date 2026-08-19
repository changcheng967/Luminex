#!/usr/bin/env python3
"""Three-way: SF18 (truth) vs luminex vs stash on loss-swing positions."""
import sys

def load(path):
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 2:
                r = dict(zip(hdr, f))
                rows[r["fen"]] = r
    return rows

def cpv(r, pfx):
    c, m = r.get(pfx + "_cp"), r.get(pfx + "_mate")
    if c not in (None, "None", ""):
        return int(c)
    if m not in (None, "None", ""):
        return 1500 if int(m) > 0 else -1500
    return None

def main(ref_path, duel_path):
    ref = load(ref_path)
    duel = load(duel_path)
    n = sf_plays = 0
    status = {"won": 0, "balanced": 0, "lost": 0}
    lu_mae = st_mae = 0.0
    n_mae = 0
    lu_agree = st_agree = 0
    lu_rep = st_rep = 0
    for fen, rr in ref.items():
        d = duel.get(fen)
        if not d:
            continue
        n += 1
        played = rr["played"]
        sfc = cpv(rr, "sf")
        luc = cpv(d, "lu")
        stc = cpv(d, "st")
        if rr["sf_bm"] == played:
            sf_plays += 1
        if sfc is not None:
            status["won" if sfc >= 100 else "lost" if sfc <= -100 else "balanced"] += 1
        if None not in (luc, stc, sfc):
            lu_mae += abs(luc - sfc)
            st_mae += abs(stc - sfc)
            n_mae += 1
        if d["lu_bm"] == rr["sf_bm"]:
            lu_agree += 1
        if d["st_bm"] == rr["sf_bm"]:
            st_agree += 1
        pos_ok = (sfc is None) or (sfc >= -50)
        if pos_ok and rr["sf_bm"] != played:
            if d["lu_bm"] == played:
                lu_rep += 1
            if d["st_bm"] == played:
                st_rep += 1
    print(f"n={n}")
    print(f"SF18 plays the game move: {sf_plays} ({100*sf_plays/max(n,1):.0f}%)")
    print(f"position per SF18 (mover view): {status}")
    print(f"cp MAE vs SF18: luminex {lu_mae/max(n_mae,1):.0f}  stash {st_mae/max(n_mae,1):.0f}  (n={n_mae})")
    print(f"bestmove agree w/ SF18: luminex {lu_agree}/{n} ({100*lu_agree/max(n,1):.0f}%)  stash {st_agree}/{n} ({100*st_agree/max(n,1):.0f}%)")
    print(f"true-blunder repeats (pos ok, SF avoids, engine repeats): luminex {lu_rep}  stash {st_rep}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
