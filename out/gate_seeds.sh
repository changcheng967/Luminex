#!/bin/bash
# Seed-variance gates: s7/s123/s2024 vs T0, 2,000 games each (10 chunks of
# 200, resumable, distinct openings per gate via offset start). Together with
# r0b (seed 99, already gated at 600g) and T0 itself these give the sigma_seed
# estimate for single-seed experiments.
set -u
G=/work/home/scnmyrr3ge/luminex-dcu/gates
cd $G
ENG=$G/luminex-legacy-modern
T0=$G/t0_i8.nnue
FC=$G/fastchess-linux-x86-64/fastchess
S1=$G/../session1

for NET in s7 s123 s2024; do
  RES=$G/seed_${NET}_results.txt
  mkdir -p $G/pgn
  for i in $(seq 1 10); do
    grep -q "CHUNK $i DONE" $RES 2>/dev/null && continue
    START=$((2001 + (i-1)*100))   # openings 2001-3000: disjoint from h2h gate
    "$FC" \
      -engine cmd=$ENG name=$NET option.NNUEFile=$S1/${NET}_i8.nnue option.Threads=1 option.Hash=64 \
      -engine cmd=$ENG name=T0    option.NNUEFile=$T0                  option.Threads=1 option.Hash=64 \
      -each proto=uci tc=60+0 option.NodesPerMove=20000 \
      -openings file=$G/8moves_v3.pgn format=pgn order=sequential start=$START \
      -rounds 100 -repeat -concurrency 7 \
      -pgnout file=$G/pgn/seed_${NET}_$i.pgn > $G/seed_${NET}_$i.log 2>&1
    S=$(grep -E "^Games: " $G/seed_${NET}_$i.log | tail -1)
    if [ -z "$S" ]; then
      echo "CHUNK $i FAILED (no score line)" >> $RES
      tail -5 $G/seed_${NET}_$i.log >> $RES
      break
    fi
    echo "CHUNK $i: $S" >> $RES
    echo "CHUNK $i DONE" >> $RES
  done
  echo "SEED_${NET}_GATE_DONE" >> $RES
done
echo SEED_GATES_ALL_DONE >> $G/seed_gates_master.txt
cat $G/seed_s7_results.txt $G/seed_s123_results.txt $G/seed_s2024_results.txt 2>/dev/null
