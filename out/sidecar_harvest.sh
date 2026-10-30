#!/bin/bash
# Sidecar harvester: the trainer writes nets to Luminex/nnue/openi_upload/out/
# (relative to the trainer file, NOT cwd), but the session-2 scripts try to
# quantize/copy them from cwd — which fails. This loop harvests every net that
# appears in the trainer's out/ dir until the whole chain is done, then exits.
# (The scripts' own failed harvest lines are harmless echoes; the chain runs on.)
set -u
WS=/workspace/luminex
TD=$WS/Luminex/nnue/openi_upload/out
HARVEST=/work/home/scnmyrr3ge/luminex-dcu/session1
cd $WS
while true; do
  for f in "$TD"/*.nnue; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .nnue)
    case $n in *_i8) continue;; esac
    [ -s "$HARVEST/$n.nnue" ] && continue
    python3 $WS/Luminex/nnue/openi_upload/quantize_i8.py "$f" "$TD/${n}_i8.nnue" \
      && cp -f "$f" "$TD/${n}_i8.nnue" "$HARVEST/" \
      && echo "sidecar harvested $n $(date +%H:%M:%S)" >> $WS/logs/sidecar.log
  done
  if grep -q SESSION2_DIVERSITY_DONE $WS/logs/session2.log 2>/dev/null; then
    # final sweep already done above this iteration; give a slow quantize 60s
    sleep 60
    break
  fi
  sleep 120
done
echo SIDECAR_EXIT $(date +%H:%M:%S) >> $WS/logs/sidecar.log
