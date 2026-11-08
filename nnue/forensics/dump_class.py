"""Write the search-not-eval mistakes (attrib.py's definition) to a jsonl."""
import json
import sys

errs = [json.loads(l) for l in open(sys.argv[1])]
for i, e in enumerate(errs):
    e["id"] = i
lux = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2])}
sf = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[3])}
out = open(sys.argv[4], "w")
n = 0
for e in errs:
    L, S = lux.get(e["id"]), sf.get(e["id"])
    if not L or not S or S.get("es_best") is None or S["es_move"].get(e["played"]) is None:
        continue
    esb, em = S["es_best"], S["es_move"]
    lp = esb - em[e["played"]]
    if lp < 0.03:
        continue
    loss = lambda m: None if em.get(m) is None else esb - em[m]
    if any(v is not None and v < 0.5 * lp for v in (loss(L["r1"]), loss(L["r8"]), loss(L["r64"]))):
        continue
    if L["s_played"] >= L["s_best"]:
        continue
    out.write(json.dumps({"id": e["id"], "fen": e["fen"], "played": e["played"], "best": e["best"],
                          "sf_best": S["best"], "es_best": esb, "es_move": em, "loss": lp,
                          "pieces": e["f"]["pieces"], "es_before": e["es_before"],
                          "s_played": L["s_played"], "s_best": L["s_best"],
                          "q_played": L["q_played"], "q_best": L["q_best"]}) + "\n")
    n += 1
print(n, "search-not-eval mistakes written")
