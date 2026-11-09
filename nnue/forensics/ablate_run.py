#!/usr/bin/env python3
"""Search each case cold with one pruning / reduction rule switched off.

usage: ablate_run.py CASES.jsonl LUMINEX NET NODES MASKS WORKERS OUT.jsonl
MASKS: comma-separated AblateMask values (0 = the unchanged search). Every
(case, mask) pair is one cold search (ucinewgame) at NODES; OUT gets the
chosen move, score and depth. Restartable: pairs already in OUT are skipped."""
import json
import multiprocessing as mp
import os
import sys

from referee import UCI

_G = {}


def _init(path, net):
    _G["eng"] = UCI(path, {"Threads": 1, "Hash": int(os.environ.get("ABL_HASH", "64")), "UseNNUE": "true", "NNUEFile": net})
    _G["mask"] = None


def task(item):
    case, mask, nodes = item
    eng = _G["eng"]
    if _G["mask"] != mask:
        eng.send(f"setoption name AblateMask value {mask}")
        _G["mask"] = mask
    eng.send("ucinewgame")
    eng.send("isready")
    eng.wait("readyok")
    r = eng.go(case["fen"], nodes)
    return {"id": case["id"], "mask": mask, "move": r["best"], "cp": r.get("cp"), "mate": r.get("mate"), "d": r.get("d")}


def main():
    cases_path, lux, net, nodes, masks, workers, out = sys.argv[1:8]
    nodes, workers = int(nodes), int(workers)
    masks = [int(m) for m in masks.split(",")]
    cases = [json.loads(l) for l in open(cases_path)]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["id"], r["mask"]))
    # mask-major order keeps each worker on one mask for long stretches
    items = [(c, m, nodes) for m in masks for c in cases if (c["id"], m) not in done]
    print(f"{len(items)} searches to run ({len(done)} done)", file=sys.stderr, flush=True)
    with mp.Pool(workers, initializer=_init, initargs=(lux, net)) as pool, open(out, "a") as fo:
        for k, r in enumerate(pool.imap_unordered(task, items, chunksize=4)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if k % 100 == 0:
                print(f"{k}/{len(items)}", file=sys.stderr, flush=True)
    print("ABLATE-DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
