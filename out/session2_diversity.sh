#!/bin/bash
# Session 2 diversity screen (redundancy axis, no new dump needed):
#   arm A = h300 : 300M prefix of the SAME corpus, 12000 steps  (5.2 passes)
#   arm B = d2   : 597M full corpus,       12000 steps  (2.6 passes)  [already trained]
# Total visits identical (1.573B). Only per-position redundancy differs.
# Decision rule (pre-registered): A vs B difference >= 20 Elo (h2h at 2000
# games, half-width +-15) => redundancy/pass-penalty is real at this ratio
# and the 2B unique-corpus arm is justified. Tie => the diversity mechanism
# does not pay at 2x and 2B almost certainly will not either.
# Seed: DEFAULT (42) — paired with both d2 and T0, so the h2h is seed-clean.
set -u
WS=/workspace/luminex
cd $WS
source /opt/dtk/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/.hyhal/lib:$LD_LIBRARY_PATH
CKPT=$WS/luminex_gen768p4.pt
FEAT=$WS/luminex-featurize
HARVEST=/work/home/scnmyrr3ge/luminex-dcu/session1
HALF=$WS/half300.tsv.xz
mkdir -p logs "$HARVEST"

echo "=== h300 corpus cut start $(date +%H:%M:%S)" | tee -a logs/session2.log
if [ ! -s "$HALF" ]; then
  xz -dc $WS/sf_train_full_q.tsv.xz | head -n 300000000 | xz -1 -T0 > "$HALF" \
    || { echo "h300 CUT_FAILED" >> logs/session2.log; exit 1; }
fi
# sanity: compressed size should be ~half the full corpus (7.2G -> ~3.6G)
ls -la "$HALF" | tee -a logs/session2.log
echo "=== h300 train start $(date +%H:%M:%S)" | tee -a logs/session2.log
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=1 ROCr_VISIBLE_DEVICES=1 \
env FEAT=$FEAT NNUE_L1=768 NNUE_RESUME=$CKPT \
    NNUE_FENEVAL_FILE=$HALF \
    NNUE_LR=1e-4 NNUE_BS=131072 NNUE_T_MAX_STEPS=12000 \
    NNUE_LOSS_POWER=2.6 NNUE_GRAD_CLIP=1.0 NNUE_VRAM_POS=60000000 \
    NNUE_FEAT_THREADS=8 \
    NNUE_QUIET_LOSS="1,0.3,0" \
    NNUE_OUT_NAME=h300.nnue \
    python3 $WS/Luminex/nnue/openi_upload/c500_train_stream.py > logs/h300.log 2>&1
D=$(grep -E "^DONE" logs/h300.log | tail -1)
Q=$(grep "weighted loss-mass" logs/h300.log | tail -1 | grep -oE "q=[0-9.]+ c=[0-9.]+")
echo "h300 | $D | $Q" >> logs/session2.log
python3 $WS/Luminex/nnue/openi_upload/quantize_i8.py h300.nnue h300_i8.nnue >> logs/h300.log 2>&1 \
  && cp -f h300.nnue h300_i8.nnue "$HARVEST/" \
  && echo "h300 harvested $(date +%H:%M:%S)" >> logs/session2.log \
  || echo "h300 QUANTIZE_OR_HARVEST_FAILED" >> logs/session2.log
echo SESSION2_DIVERSITY_DONE >> logs/session2.log
cat logs/session2.log
