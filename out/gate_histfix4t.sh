#!/bin/bash
# Repair gate 1b: the MULTITHREADED read of the history-table fix.
# fix1 vs base, both Threads=4 (CCRL condition), same T0 net.
# 2,000 games in 10 chunks of 200. Threads=4 per engine -> 8 cores per game,
# concurrency 1 on the 14-core quota. Openings 5001-6000 (disjoint).
# The 1-thread gate was a 50.00% bit-identity no-regression proof; THIS gate
# measures the cache-contention removal that the fix exists for.
set -u
G=/work/home/scnmyrr3ge/luminex-dcu/gates
cd $G
NEW=$G/luminex-fix1-modern
OLD=$G/luminex-legacy-modern
T0=$G/t0_i8.nnue
FC=$G/fastchess-linux-x86-64/fastchess
RES=$G/histfix4t_results.txt
mkdir -p $G/pgn

[ -x "$NEW" ] && [ -x "$OLD" ] && [ -s "$T0" ] && [ -x "$FC" ] || { echo "MISSING_PREREQ" >> $RES; exit 1; }

for i in $(seq 1 10); do
  grep -q "CHUNK $i DONE" $RES 2>/dev/null && continue
  START=$((5001 + (i-1)*100))
  "$FC" \
    -engine cmd=$NEW name=fix1_4t option.NNUEFile=$T0 option.Threads=4 option.Hash=64 \
    -engine cmd=$OLD name=base_4t option.NNUEFile=$T0 option.Threads=4 option.Hash=64 \
    -each proto=uci tc=60+0 option.NodesPerMove=20000 \
    -openings file=$G/8moves_v3.pgn format=pgn order=sequential start=$START \
    -rounds 100 -repeat -concurrency 1 \
    -pgnout file=$G/pgn/histfix4t_$i.pgn > $G/histfix4t_$i.log 2>&1
  S=$(grep -E "^Games: " $G/histfix4t_$i.log | tail -1)
  if [ -z "$S" ]; then
    echo "CHUNK $i FAILED (no score line)" >> $RES
    tail -3 $G/histfix4t_$i.log >> $RES
    break
  fi
  echo "CHUNK $i: $S" >> $RES
  echo "CHUNK $i DONE" >> $RES
done
echo HISTFIX4T_GATE_DONE >> $RES
cat $RES
