"""Cases the unchanged search (mask 0) still gets wrong: chosen move loses >= half the played move's loss."""
import json
import sys

cases = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
new = {}
try:
    for l in open(sys.argv[3]):
        r = json.loads(l)
        new[(r["id"], r["move"])] = r["es"]
except FileNotFoundError:
    pass
keep = 0
with open(sys.argv[4], "w") as fo:
    for l in open(sys.argv[2]):
        r = json.loads(l)
        if r["mask"] != 0:
            continue
        c = cases[r["id"]]
        e = c["es_move"].get(r["move"], new.get((r["id"], r["move"])))
        if e is None or c["es_best"] - e >= 0.5 * c["loss"]:
            fo.write(json.dumps(c) + "\n")
            keep += 1
print(keep, "cases still wrong on current main")
