#!/bin/bash
# Gen768 PASS 4 — multi-epoch warm restart on the SAME 4.29B pack, WITH DOSL
# dual-head actually trained (the pass-3 zero-LINH bug is now guarded: trainer
# refuses to export an all-zero head; engine ignores one at load time).
#
# Prereqs on the OpenI job:
#   1. gamepack.tar        — same 4.29B pack as pass 3 (132 frames)
#   2. luminex_gen768.pt   — pass-3 checkpoint (model + Adam moments). If the
#      job's /tmp/code doesn't have it, upload it from the pass-3 artifacts.
#   3. these scripts: c500_train_stream.py, luminex_nnue_train.py (this repo)
#
# Recipe notes:
#   LR     1e-4  — half of the pass-3 warm-restart peak (descending ladder)
#   SEED   43    — pass 3 used 42; a new permutation decorrelates this epoch
#   T_MAX  33000 — pinned so cosine completes exactly at 1 pass (4.29B / 131072)
#   DOSL   warmup 2000 steps @ lam=1.0 (lin head trains alone from zero),
#          then lam=0.25 anchor for the rest of the pass; linMAE in HEALTH.
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
export NNUE_LIN_LAMBDA=0.25
export NNUE_LOSS_POWER=2.6
python c500_train_stream.py
# after: python quantize_i8.py luminex_gen768p4.nnue luminex_gen768p4_i8.nnue
