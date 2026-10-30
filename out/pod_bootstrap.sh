#!/bin/bash
# One-shot bootstrap after a pod restart: rebuild /workspace/luminex from the
# NFS home and relaunch the session-2 chain (seed runs -> h300 screen).
# Idempotent: skips anything already present.
set -eu
WS=/workspace/luminex
SRC=/work/home/scnmyrr3ge/luminex-dcu
mkdir -p $WS/logs
cd $WS

for f in sf_train_full_q.tsv.xz luminex_gen768p4.pt \
         session2_seed.sh session2_diversity.sh session2_chain.sh; do
  [ -s $WS/$f ] || cp $SRC/$f $WS/
done
# featurizer: the NFS home carries luminex-featurize-fresh (session-1 era);
# the old pod ran it renamed to luminex-featurize
if [ ! -s $WS/luminex-featurize ]; then
  cp $SRC/luminex-featurize-fresh $WS/luminex-featurize
fi
chmod +x luminex-featurize session2_*.sh
$WS/luminex-featurize --fen-eval --help </dev/null >/dev/null 2>&1 || true  # exec-permission smoke test only

# corpus and checkpoint integrity guards (sizes as measured on the good runs)
[ "$(stat -c%s $WS/sf_train_full_q.tsv.xz)" = "7198919356" ] || { echo CORPUS_SIZE_WRONG; exit 1; }
[ "$(stat -c%s $WS/luminex_gen768p4.pt)"  = "378530589"  ] || { echo CKPT_SIZE_WRONG;   exit 1; }

# trainer: prefer the NFS copy (no network dependency), fall back to git clone
if [ ! -f $WS/Luminex/nnue/openi_upload/c500_train_stream.py ]; then
  mkdir -p $WS/Luminex/nnue
  cp -r $SRC/nnue/openi_upload $WS/Luminex/nnue/ 2>/dev/null \
    || git clone --depth 1 https://github.com/changcheng967/Luminex.git $WS/Luminex
fi
[ -f $WS/Luminex/nnue/openi_upload/c500_train_stream.py ] || { echo TRAINER_MISSING; exit 1; }
[ -f $WS/Luminex/nnue/openi_upload/quantize_i8.py ]       || { echo QUANTIZER_MISSING; exit 1; }

# only (re)launch if the chain isn't already running
if ! pgrep -f session2_seed.sh >/dev/null; then
  nohup setsid bash session2_seed.sh > logs/nohup_session2.log 2>&1 < /dev/null &
fi
if ! pgrep -f session2_chain.sh >/dev/null; then
  nohup setsid bash session2_chain.sh > logs/nohup_chain.log 2>&1 < /dev/null &
fi
sleep 2
pgrep -fa 'session2_(seed|chain)' || true
echo BOOTSTRAP_DONE $(date +%H:%M:%S)
