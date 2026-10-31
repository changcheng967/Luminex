#!/bin/bash
# Repair gate 2: extension plumbing. fix2 binary (extensions live in every
# branch, capped at 2) vs fix1 binary (history fix only), same T0 net,
# 1 thread. 4,000 games in 20 chunks of 200. Openings 6001-8000 (disjoint
# from 1-1000, 2001-3000, 3001-4000, 5001-6000).
# Expectation pre-registered: extensions activate never-fired machinery —
# sign UNKNOWN (tactical depth helps; node inflation at fixed budget hurts;
# SPSA margins were tuned with extensions dead). This gate measures net effect;
# a retune follows if positive-trending.
set -u
G=/work/home/scnmyrr3ge/luminex-dcu/gates
cd $G
NEW=$G/luminex-fix2-modern
OLD=$G/luminex-fix1-modern
T0=$G/t0_i8.nnue
FC=$G/fastchess-linux-x86-64/fastchess
RES=$G/fix2_results.txt
mkdir -p $G/pgn

[ -x "$NEW" ] && [ -x "$OLD" ] && [ -s "$T0" ] && [ -x "$FC" ] || { echo "MISSING_PREREQ" >> $RES; exit 1; }

for i in $(seq 1 20); do
  grep -q "CHUNK $i DONE" $RES 2>/dev/null && continue
  START=$((6001 + (i-1)*100))
  "$FC" \
    -engine cmd=$NEW name=fix2 option.NNUEFile=$T0 option.Threads=1 option.Hash=64 \
    -engine cmd=$OLD name=fix1 option.NNUEFile=$T0 option.Threads=1 option.Hash=64 \
    -each proto=uci tc=60+0 option.NodesPerMove=20000 \
    -openings file=$G/8moves_v3.pgn format=pgn order=sequential start=$START \
    -rounds 100 -repeat -concurrency 7 \
    -pgnout file=$G/pgn/fix2_$i.pgn > $G/fix2_$i.log 2>&1
  S=$(grep -E "^Games: " $G/fix2_$i.log | tail -1)
  if [ -z "$S" ]; then
    echo "CHUNK $i FAILED (no score line)" >> $RES
    tail -3 $G/fix2_$i.log >> $RES
    break
  fi
  echo "CHUNK $i: $S" >> $RES
  echo "CHUNK $i DONE" >> $RES
done
echo FIX2_GATE_DONE >> $RES
cat $RES
