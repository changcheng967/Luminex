#!/bin/bash
# Session 2 seed-variance: three runs at T0's exact recipe (p4 warm start,
# 6000 steps, quiet 1/0.3/0, LR 1e-4, BS 131072), only the shuffle seed
# differs (7, 123, 2024). Together with T0 (default seed) and r0b (99) that
# is five seed draws -> a real sigma_seed for the training program.
# Sequential on GPU 1 (GPU 0 carries driver-level leaked VRAM on this stack).
set -u
WS=/workspace/luminex
cd $WS
source /opt/dtk/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/.hyhal/lib:$LD_LIBRARY_PATH
DATA=$WS/sf_train_full_q.tsv.xz
CKPT=$WS/luminex_gen768p4.pt
FEAT=$WS/luminex-featurize
HARVEST=/work/home/scnmyrr3ge/luminex-dcu/session1
mkdir -p logs "$HARVEST"

run () {
  NAME=$1; TMAX=$2; shift 2
  echo "=== $NAME start $(date +%H:%M:%S) $*" | tee -a logs/session2.log
  CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=1 ROCr_VISIBLE_DEVICES=1 \
  env FEAT=$FEAT NNUE_L1=768 NNUE_RESUME=$CKPT \
      NNUE_FENEVAL_FILE=$DATA \
      NNUE_LR=1e-4 NNUE_BS=131072 NNUE_T_MAX_STEPS=$TMAX \
      NNUE_LOSS_POWER=2.6 NNUE_GRAD_CLIP=1.0 NNUE_VRAM_POS=60000000 \
      NNUE_FEAT_THREADS=8 \
      NNUE_OUT_NAME=$NAME.nnue "$@" \
      python3 $WS/Luminex/nnue/openi_upload/c500_train_stream.py > logs/$NAME.log 2>&1
  D=$(grep -E "^DONE" logs/$NAME.log | tail -1)
  Q=$(grep "weighted loss-mass" logs/$NAME.log | tail -1 | grep -oE "q=[0-9.]+ c=[0-9.]+")
  echo "$NAME | $D | $Q" >> logs/session2.log
  # trainer writes to Luminex/nnue/openi_upload/out/ (relative to the trainer
  # file, NOT cwd) — harvest from there
  TD=$WS/Luminex/nnue/openi_upload/out
  python3 $WS/Luminex/nnue/openi_upload/quantize_i8.py $TD/$NAME.nnue $TD/${NAME}_i8.nnue >> logs/$NAME.log 2>&1 \
    && cp -f $TD/$NAME.nnue $TD/${NAME}_i8.nnue "$HARVEST/" \
    && echo "$NAME harvested $(date +%H:%M:%S)" >> logs/session2.log \
    || echo "$NAME QUANTIZE_OR_HARVEST_FAILED" >> logs/session2.log
}

run s7    6000 NNUE_QUIET_LOSS="1,0.3,0" NNUE_SHUFFLE_SEED=7
run s123  6000 NNUE_QUIET_LOSS="1,0.3,0" NNUE_SHUFFLE_SEED=123
run s2024 6000 NNUE_QUIET_LOSS="1,0.3,0" NNUE_SHUFFLE_SEED=2024
echo SESSION2_SEED_DONE >> logs/session2.log
cat logs/session2.log
