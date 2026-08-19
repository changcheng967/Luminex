#!/usr/bin/env python3
"""Single deciding readout: static-eval MAE vs fixed references, sharp vs quiet.

Usage: python readout.py <label>
  Reads nnue/openi_upload/{swing_refs.txt,quiet5k.txt}, evaluates the CURRENT
  local build's static eval on both corpora (via `eval` UCI, one persistent
  process), reports white-relative MAE per corpus. Run after rebuilding the
  engine with a candidate's eval_fitted.h. Baseline for comparison: run once
  on the current (fit6) build first and record the numbers.
"""
import subprocess
import sys

ENG = r"C:\Users\chang\Downloads\Luminex\build\luminex.exe"
DIR = r"C:\Users\chang\Downloads\Luminex\nnue\openi_upload"

def load_refs():
    # sharp: swing FENs + SF18 depth-24 refs (from ref_out.txt: fen\t...\tsf_bm\tsf_cp\tsf_mate)
    sharp = []
    with open(f"{DIR}\\ref_out.txt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            r = dict(zip(hdr, f))
            if r["sf_cp"] not in ("None", ""):
                ref = int(r["sf_cp"])
            elif r["sf_mate"] not in ("None", ""):
                ref = 1500 if int(r["sf_mate"]) > 0 else -1500
            else:
                continue
            sharp.append((r["fen"], ref))
    # quiet: fen\twhite_eval (SF15-era refs)
    quiet = []
    for ln in open(f"{DIR}\\quiet5k.txt"):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 2:
            quiet.append((f[0], int(float(f[1]))))
    return sharp, quiet

def eval_all(fens):
    p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while "uciok" not in p.stdout.readline():
        pass
    vals = []
    for fen, ref in fens:
        stm_white = " w " in fen
        p.stdin.write(f"position fen {fen}\neval\n"); p.stdin.flush()
        cp = None
        while True:
            ln = p.stdout.readline().strip()
            if ln.startswith("eval cp"):
                cp = int(ln.split()[2])
                break
            if not ln:
                break
        if cp is not None:
            vals.append((ref, cp if stm_white else -cp))
    p.stdin.write("quit\n"); p.stdin.flush()
    return vals

def mae(vals, cap=None):
    if not vals:
        return float("nan"), 0
    errs = [abs(r - e) for r, e in vals]
    if cap:
        errs = [min(e, cap) for e in errs]
    return sum(errs) / len(errs), len(vals)

def main(label):
    sharp, quiet = load_refs()
    s = eval_all(sharp)
    q = eval_all(quiet)
    sm, sn = mae(s, cap=1500)
    qm, qn = mae(q, cap=1000)
    print(f"READOUT {label}: sharp MAE = {sm:.1f} (n={sn})   quiet MAE = {qm:.1f} (n={qn})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "unnamed")
