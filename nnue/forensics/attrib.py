#!/usr/bin/env python3
"""Attribute our mistakes to search, eval or labels (analyze.py + replay.py output).

usage: attrib.py ERRORS.jsonl LUX.jsonl SF.jsonl

For each mistake, re-scored by Stockfish at the replay's node count from the
same root (loss = es(best) - es(move), mover's point of view):
  unconfirmed     the played move loses < 0.03 at the replay's depth
  cold-1x ok      our engine, cold, at the game's 100K nodes already avoids it
                  (the game's error came from search state: TT, history)
  fixed 8x / 64x  our engine finds a move losing < half as much with 8x / 64x
                  the nodes: search-limited
  persistent      not fixed at 64x; split by who ranks (best, played) right:
     eval-learnable   our static eval prefers played, Stockfish's static eval
                      prefers best: a net can know this without search
     eval-deep        both static evals prefer played, Stockfish depth 9 (the
                      kind of label we train on) prefers best
     beyond-labels    depth 9 prefers played too: our labels cannot teach it
     search-not-eval  our static eval already prefers best
Loss mass per class, overall and by phase and kind of mistake."""
import collections
import json
import sys


def load(path, key="id"):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r[key]] = r
    return out


def main():
    errs = [json.loads(l) for l in open(sys.argv[1])]
    for i, e in enumerate(errs):
        e["id"] = i
    lux, sf = load(sys.argv[2]), load(sys.argv[3])
    rows = []
    for e in errs:
        if e["id"] not in lux or e["id"] not in sf:
            continue
        L, S = lux[e["id"]], sf[e["id"]]
        esb = S.get("es_best")
        em = S.get("es_move", {})
        if esb is None or em.get(e["played"]) is None:
            continue

        def loss(m):
            v = em.get(m)
            return None if v is None else max(0.0, esb - v)

        lp = loss(e["played"])
        cls = None
        if lp < 0.03:
            cls = "unconfirmed"
        else:
            l1, l8, l64 = loss(L["r1"]), loss(L["r8"]), loss(L["r64"])
            if l1 is not None and l1 < 0.5 * lp:
                cls = "cold-1x ok"
            elif l8 is not None and l8 < 0.5 * lp:
                cls = "fixed 8x"
            elif l64 is not None and l64 < 0.5 * lp:
                cls = "fixed 64x"
            else:
                s_pref_played = L["s_played"] >= L["s_best"]
                sfs_p, sfs_b = S.get("sfs_played"), S.get("sfs_best")
                d9p, d9b = S.get("d9_played"), S.get("d9_best")
                if not s_pref_played:
                    cls = "search-not-eval"
                elif sfs_p is not None and sfs_b is not None and sfs_b > sfs_p:
                    cls = "eval-learnable"
                elif d9p is not None and d9b is not None and d9b > d9p:
                    cls = "eval-deep"
                else:
                    cls = "beyond-labels"
        f = e["f"]
        phase = "eg<=12" if f["pieces"] <= 12 else ("mg13-26" if f["pieces"] <= 26 else "op27+")
        rows.append({"cls": cls, "loss": lp, "phase": phase, "kind": f["kind"], "bucket": f["bucket"],
                     "regime": "winning" if e["es_before"] > 0.85 else ("losing" if e["es_before"] < 0.15 else "balanced")})
    order = ["unconfirmed", "cold-1x ok", "fixed 8x", "fixed 64x", "search-not-eval",
             "eval-learnable", "eval-deep", "beyond-labels"]
    conf = [r for r in rows if r["cls"] != "unconfirmed"]
    tot = sum(r["loss"] for r in conf) or 1.0
    print(f"{len(rows)} mistakes re-scored; {len(conf)} confirmed (loss >= 0.03), total loss {tot:.1f}")
    c = collections.Counter(r["cls"] for r in rows)
    m = collections.defaultdict(float)
    for r in rows:
        m[r["cls"]] += r["loss"]
    for k in order:
        share = "" if k == "unconfirmed" else f"{100 * m[k] / tot:5.1f}% of confirmed loss"
        print(f"  {k:>16}: {c[k]:5d} mistakes, loss {m[k]:7.2f}  {share}")

    def split(key, keys=None):
        agg = collections.defaultdict(lambda: collections.defaultdict(float))
        for r in conf:
            agg[r[key]][r["cls"]] += r["loss"]
        print(f"\n  by {key}: share of each row's loss (row loss / share of all confirmed loss)")
        print("  " + " " * 14 + "".join(f"{k[:12]:>13}" for k in order[1:]))
        for g in (keys or sorted(agg)):
            if g not in agg:
                continue
            s = sum(agg[g].values()) or 1.0
            print(f"  {str(g):>14}" + "".join(f"{100 * agg[g][k] / s:12.1f}%" for k in order[1:])
                  + f"   ({s:.1f} / {100 * s / tot:.1f}%)")

    split("phase", ["op27+", "mg13-26", "eg<=12"])
    split("kind")
    split("regime", ["balanced", "winning", "losing"])
    split("bucket", list(range(8)))


if __name__ == "__main__":
    main()
