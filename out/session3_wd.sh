#!/bin/bash
# Session 3 (corrected) — the DILUTION probe, single informative arm.
# Decay budget arithmetic (verified in code): clamped-cosine LR 1e-4 over
# 6000 steps gives sum(lr) ~ 0.3; TAIL_LR_MULT default is 1.0 (the "3" is a
# header-comment example, never set in our recipe). AdamW total shrink =
# exp(-wd * 0.3): wd=0.01 -> 0.3% (inert), wd=0.1 -> 3% (still dead zone),
# wd=0.35 -> 10% (first measurable), wd=1.0 -> 26% (THIS arm), wd=2.3 -> 50%.
# T0 (wd=0.01) is the control; everything else T0's exact recipe, default
# seed (seed-paired). Pre-registered: wd100 >= +5 vs T0 at 2,500-4,000 games
# -> tail capacity excess real (buckets worth running, MoE prior up);
# ~0 (±3) -> tail saturated, skip buckets, MoE's case rests on FT
# unsaturation alone; negative -> architecture axis closes for good.
# PATHOLOGY WATCH: NNUE_WCLAMP is OFF with root cause recorded as
# "tail wd + amsgrad" — raising wd 100x re-enters that region. The per-step
# "L2|max|" log line is the tail-norm monitor (prior runs ~0.76-0.79); if it
# GROWS under wd=1.0 the old pathology has recurred and the reading is void
# regardless of the gate.
set -u
WS=/workspace/luminex
SRC=/work/home/scnmyrr3ge/luminex-dcu
mkdir -p $WS/logs
cd $WS
source /opt/dtk/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/.hyhal/lib:$LD_LIBRARY_PATH

for f in sf_train_full_q.tsv.xz luminex_gen768p4.pt; do
  [ -s $WS/$f ] || cp $SRC/$f $WS/
done
if [ ! -s $WS/luminex-featurize ]; then cp $SRC/luminex-featurize-fresh $WS/luminex-featurize; fi
chmod +x luminex-featurize
if [ ! -f $WS/Luminex/nnue/openi_upload/c500_train_stream.py ]; then
  mkdir -p $WS/Luminex/nnue
  cp -r $SRC/gates/Luminex/nnue/openi_upload $WS/Luminex/nnue/ 2>/dev/null \
    || cp -r $SRC/nnue/openi_upload $WS/Luminex/nnue/
fi
[ "$(stat -c%s $WS/sf_train_full_q.tsv.xz)" = "7198919356" ] || { echo CORPUS_BAD >> $WS/logs/session3.log; exit 1; }

DATA=$WS/sf_train_full_q.tsv.xz
CKPT=$WS/luminex_gen768p4.pt
FEAT=$WS/luminex-featurize
TD=$WS/Luminex/nnue/openi_upload/out
HARVEST=$SRC/session1

run () {
  NAME=$1; shift
  echo "=== $NAME start $(date +%H:%M:%S) $*" | tee -a logs/session3.log
  CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=1 ROCr_VISIBLE_DEVICES=1 \
  env FEAT=$FEAT NNUE_L1=768 NNUE_RESUME=$CKPT \
      NNUE_FENEVAL_FILE=$DATA \
      NNUE_LR=1e-4 NNUE_BS=131072 NNUE_T_MAX_STEPS=6000 \
      NNUE_LOSS_POWER=2.6 NNUE_GRAD_CLIP=1.0 NNUE_VRAM_POS=60000000 \
      NNUE_FEAT_THREADS=8 NNUE_QUIET_LOSS="1,0.3,0" \
      NNUE_OUT_NAME=$NAME.nnue "$@" \
      python3 $WS/Luminex/nnue/openi_upload/c500_train_stream.py > logs/$NAME.log 2>&1
  D=$(grep -E "^DONE" logs/$NAME.log | tail -1)
  echo "$NAME | $D" >> logs/session3.log
  python3 $WS/Luminex/nnue/openi_upload/quantize_i8.py $TD/$NAME.nnue $TD/${NAME}_i8.nnue >> logs/$NAME.log 2>&1 \
    && cp -f $TD/$NAME.nnue $TD/${NAME}_i8.nnue "$HARVEST/" \
    && echo "$NAME harvested $(date +%H:%M:%S)" >> logs/session3.log \
    || echo "$NAME HARVEST_FAILED" >> logs/session3.log
}

run wd100 NNUE_TAIL_WD=1.0
echo SESSION3_WD_DONE >> logs/session3.log
cat logs/session3.log
