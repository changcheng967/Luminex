#!/bin/bash
# Repair gate 1: history tables moved per-worker. New binary vs the current
# release binary, BOTH running the same T0 net — a pure search-side change.
# 4,000 games in 20 chunks of 200 (resumable). Openings 4001-5000 (disjoint
# from h2h 1-1000, seeds 2001-3000, wd100 3001-4000).
# NOTE: this is a single-thread gate (Threads=1, like all our gates), so the
# change's multithreading benefit (cache contention removal) is NOT exercised
# here — this gate measures the semantics change (per-worker isolation vs
# shared racing updates) at 1 thread. A separate N-thread gate follows if
# this passes, since CCRL is 4-core.
set -u
G=/work/home/scnmyrr3ge/luminex-dcu/gates
cd $G
NEW=$G/luminex-fix1-modern
OLD=$G/luminex-legacy-modern
T0=$G/t0_i8.nnue
FC=$G/fastchess-linux-x86-64/fastchess
RES=$G/histfix_results.txt
mkdir -p $G/pgn

[ -x "$NEW" ] && [ -x "$OLD" ] && [ -s "$T0" ] && [ -x "$FC" ] || { echo "MISSING_PREREQ" >> $RES; exit 1; }

for i in $(seq 1 20); do
  grep -q "CHUNK $i DONE" $RES 2>/dev/null && continue
  START=$((4001 + (i-1)*100))
  "$FC" \
    -engine cmd=$NEW name=fix1 option.NNUEFile=$T0 option.Threads=1 option.Hash=64 \
    -engine cmd=$OLD name=base option.NNUEFile=$T0 option.Threads=1 option.Hash=64 \
    -each proto=uci tc=60+0 option.NodesPerMove=20000 \
    -openings file=$G/8moves_v3.pgn format=pgn order=sequential start=$START \
    -rounds 100 -repeat -concurrency 7 \
    -pgnout file=$G/pgn/histfix_$i.pgn > $G/histfix_$i.log 2>&1
  S=$(grep -E "^Games: " $G/histfix_$i.log | tail -1)
  if [ -z "$S" ]; then
    echo "CHUNK $i FAILED (no score line)" >> $RES
    tail -3 $G/histfix_$i.log >> $RES
    break
  fi
  echo "CHUNK $i: $S" >> $RES
  echo "CHUNK $i DONE" >> $RES
done
echo HISTFIX_GATE_DONE >> $RES
cat $RES
