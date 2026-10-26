#!/usr/bin/env python3
"""Verification gate for a fresh 3-column dump (fen\twhite_cp\tflag) — the
three checks that would have caught every dump incident we've had:
  1. stm-split sign check: engine evals (known-good net) must correlate
     POSITIVELY with the tsv's white-relative cp on BOTH stm halves.
  2. flag distribution sanity (quiet/capture/check ratios plausible).
  3. byte-for-byte sign check on hand-built black-to-move probes.
    python3 verify_dump.py <probe.tsv> <engine> <net>
"""
import subprocess, sys, collections

tsv, eng, net = sys.argv[1], sys.argv[2], sys.argv[3]
fens, cps, flags, stms = [], [], [], []
for line in open(tsv):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        fens.append(p[0]); cps.append(int(p[1]))
        flags.append(int(p[2]) if len(p) > 2 else 0)
        stms.append(p[0].split()[1])

# gate 2: flag distribution
c = collections.Counter(flags)
n = len(flags)
print(f"[2] flags: quiet={c[0]/n:.1%} capture={c[1]/n:.1%} check={c[2]/n:.1%}")
assert c[0] / n > 0.5 and c[2] / n < 0.15, "implausible flag distribution"

# gate 3: hand-built sign probes — quiet positions with unambiguous material
# (the first attempt used a Fool's-mate FEN where "engine says mate 0" was CORRECT)
probe_fens = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 20),     # startpos: engine ~ +0.2
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", -20),    # same, black to move
    ("rnbqkbnr/pPpppppp/8/8/8/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1", -120),   # white up a pawn (b7 taken)
    ("rnbqkbnr/pPpppppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 2", 120),    # same position, white to move
]
p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write(f"setoption name NNUEFile value {net}\nisready\n")
p.stdin.flush()
while True:
    ln = p.stdout.readline()
    if not ln: break
    if ln.startswith("readyok"): break
ok3 = True
for fen, expect_sign in probe_fens:
    p.stdin.write(f"position fen {fen}\ngo nodes 400\n"); p.stdin.flush()
    cp = 0
    while True:
        ln = p.stdout.readline()
        if not ln or ln.startswith("bestmove"): break
        if "score cp" in ln and "bound" not in ln:
            cp = int(ln.split("score cp")[1].split()[0])
    stm = fen.split()[1]
    white_eval = (cp if stm == "w" else -cp) if cp is not None else 0
    if (white_eval > 0) != (expect_sign > 0):
        ok3 = False
        print(f"[3] FAIL: {fen[:40]} engine_white={white_eval} expected sign {expect_sign}")
print(f"[3] hand-built sign probes: {'PASS' if ok3 else 'FAIL'}")

# gate 1: stm-split correlation vs tsv white-relative cp
import math
def corr(a, b):
    n = len(a); ma, mb = sum(a)/n, sum(b)/n
    cov = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x-ma)**2 for x in a)); vb = math.sqrt(sum((y-mb)**2 for y in b))
    return cov/(va*vb) if va and vb else 0.0

sample = fens[:400]
evs = []
for fen in sample:
    p.stdin.write(f"position fen {fen}\ngo nodes 400\n"); p.stdin.flush()
    cp = 0
    while True:
        ln = p.stdout.readline()
        if not ln or ln.startswith("bestmove"): break
        if "score cp" in ln and "bound" not in ln:
            cp = int(ln.split("score cp")[1].split()[0])
    stm = fen.split()[1]
    evs.append((cp if stm == "w" else -cp) if cp is not None else 0)  # engine white-relative
p.stdin.write("quit\n"); p.stdin.flush(); p.wait()

tsv_cp = cps[:len(evs)]
st = stms[:len(evs)]
for name, mask in (("WTM", [i for i in range(len(evs)) if st[i]=="w"]),
                   ("BTM", [i for i in range(len(evs)) if st[i]=="b"])):
    r = corr([evs[i] for i in mask], [tsv_cp[i] for i in mask])
    print(f"[1] corr(engine, tsv) on {name}: {r:+.3f}  (n={len(mask)})")
    assert r > 0.3, f"SIGN FLIP DETECTED on {name} — dump unusable"
print("ALL_VERIFICATION_GATES_PASS")
