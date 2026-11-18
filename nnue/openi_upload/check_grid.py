"""The trainer's feature-layer grid against the engine's. Exit 0 = they agree.

resident_train.py keeps the grid in one place, FT_GRID: QAT rounds the first layer's weights and biases to it
(qat_ft, qat_bias, and NNUE_QAT_FT8's int8 table). It has to equal NNUE_FT_SCALE in src/nnue_kernels.h, the grid
the engine's int16 lanes hold the layer on. Nothing else would notice a difference: the engine rounds a net's
floats again when it loads it, so a net trained on another grid loads, plays, and is not the net QAT trained (#93
moved the engine from x8192 to x2048 while the trainer stack's QAT stayed on x8192).

Two checks, both needing only the files (no torch: CI runs this on every push, and check_fused.py runs it before
any of its GPU checks):
  1. FT_GRID, assigned once at the trainer's module level, equals NNUE_FT_SCALE, defined once in the header.
  2. No torch.round() in the trainer rounds to a literal power-of-two grid (1024 to 32768): the rounding reads
     FT_GRID, so the next change of the engine's grid cannot leave a copy behind.

usage: python check_grid.py [--engine-header PATH] [--trainer PATH]
       (defaults: src/nnue_kernels.h and nnue/openi_upload/resident_train.py of this checkout)"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "nnue_kernels.h")
TRAINER = os.path.join(HERE, "resident_train.py")
NUM = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
GRID_LITERAL = re.compile(r"torch\.round\([^\n]*?[*/]\s*(?:1024|2048|4096|8192|16384|32768)(?:\.0*)?\b")


def engine_grid(header=HEADER):
    """NNUE_FT_SCALE from the engine's header."""
    found = re.findall(r"\bNNUE_FT_SCALE\s*=\s*" + NUM + r"[fF]?\s*;", open(header, encoding="utf-8").read())
    if len(found) != 1:
        raise ValueError(f"{header}: NNUE_FT_SCALE is defined {len(found)} times, not once")
    return float(found[0])


def trainer_grid(trainer=TRAINER):
    """FT_GRID from the trainer's source: its one assignment at module level."""
    found = re.findall(r"(?m)^FT_GRID\s*=\s*" + NUM + r"\s*(?:#.*)?$", open(trainer, encoding="utf-8").read())
    if len(found) != 1:
        raise ValueError(f"{trainer}: FT_GRID is assigned {len(found)} times at module level, not once")
    return float(found[0])


def literal_grids(trainer=TRAINER):
    """Lines of the trainer whose torch.round() rounds to a literal power-of-two grid instead of FT_GRID."""
    return [f"{i}: {line.strip()}" for i, line in enumerate(open(trainer, encoding="utf-8"), 1)
            if GRID_LITERAL.search(line.split("#")[0])]


def check(header=HEADER, trainer=TRAINER):
    """Both checks; one GRID_OK or GRID_FAIL line each way, True if both pass."""
    try:
        g, e = trainer_grid(trainer), engine_grid(header)
    except (OSError, ValueError) as err:
        print(f"GRID_FAIL: the trainer's grid cannot be compared with the engine's: {err}", flush=True)
        return False
    ok = True
    if g != e:
        print(f"GRID_FAIL: the trainer's FT_GRID is x{g:g} and the engine's NNUE_FT_SCALE is x{e:g} ({header}): "
              f"QAT would train the first layer on another grid than the engine holds it on", flush=True)
        ok = False
    stray = literal_grids(trainer)
    if stray:
        print(f"GRID_FAIL: {len(stray)} rounding(s) in {os.path.basename(trainer)} use a literal grid, not FT_GRID: "
              + " | ".join(stray), flush=True)
        ok = False
    if ok:
        print(f"GRID_OK: the trainer's FT_GRID equals the engine's NNUE_FT_SCALE, x{g:g}, and every rounding "
              f"to it reads FT_GRID", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine-header", default=HEADER)
    ap.add_argument("--trainer", default=TRAINER)
    a = ap.parse_args()
    sys.exit(0 if check(a.engine_header, a.trainer) else 1)
