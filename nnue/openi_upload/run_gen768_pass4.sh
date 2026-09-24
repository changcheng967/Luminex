#!/bin/bash
# Gen768 PASS 4 — max-Elo warm restart on the SAME 4.29B pack, DOSL actually
# trained (lin head now has its own optimizer group — the pass-3 bug class is
# structurally closed).
#
# Prereqs on the OpenI job:
#   1. gamepack.tar        — same 4.29B pack as pass 3 (132 frames)
#   2. luminex_gen768.pt   — pass-3 checkpoint (model + moments). NOTE: the old
#      3-group optimizer state will fail to load into the new 4-group optimizer
#      (lin group added) — the trainer rebuilds fresh moments by design. Expected
#      "[opt] state restore FAILED ... rebuilding" banner on session 1. Harmless.
#   3. these scripts: c500_train_stream.py, luminex_nnue_train.py (this repo)
#
# Recipe (every knob deliberate):
#   LR      1e-4   — half of pass-3 peak; warm-restart ladder, avoids memorization
#   SEED    43     — new permutation (pass 3 used 42): decorrelates this epoch
#   T_MAX   33000  — pinned so cosine completes exactly at 1 pass (4.29B/131072)
#   SWA     24750  — running weight average over the last 25% of the pass,
#                    persisted across budget sessions; export uses the average
#   TAILx3         — converged FT stays at base LR; the tiny tail (~100K params)
#                    still has room to adapt to the final FT state
#   LINx5 / lam0.15 — DOSL head trains fast during its 2000-step warmup, then
#                    anchors at lam=0.15 (not 0.25) so the main net keeps 85% of
#                    the gradient budget for its own final pass
set -e
export NNUE_L1=768
export NNUE_OUT_NAME=luminex_gen768p4.nnue
export NNUE_RESUME=/tmp/code/luminex_gen768.pt        # pass-3 checkpoint — MUST exist
export NNUE_LR=1e-4                                    # pass-4 peak (pass 3 was 2e-4)
export NNUE_BS=131072                                  # match pass-3 log if different
export NNUE_EPOCHS=1
export NNUE_T_MAX_STEPS=33000                          # pinned 1-pass cosine horizon
export NNUE_FRAME_SHUFFLE=1
export NNUE_SHUFFLE_SEED=43                            # new order vs pass 3 (42)
export NNUE_DUAL_HEAD=1                                # default now, kept explicit
export NNUE_LIN_WARMUP=2000
export NNUE_LIN_LAMBDA=0.15
export NNUE_LIN_LR_MULT=5
export NNUE_TAIL_LR_MULT=3
export NNUE_SWA_START_STEP=24750                       # last 25% of the pass
export NNUE_LOSS_POWER=2.6
python c500_train_stream.py
# after: python quantize_i8.py luminex_gen768p4.nnue luminex_gen768p4_i8.nnue
# then A/B in-engine: QsearchLinear=true (uses the trained head) vs false, 2x50g STC
