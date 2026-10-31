#!/bin/bash
# wd100 vs T0 — the DILUTION pilot decision gate. 4,000 games in 20 chunks of
# 200 (resumable per chunk). Openings 3001-4000 (disjoint from h2h 1-1000 and
# seeds 2001-3000). Decision tree pre-registered in DEVLOG round 13:
#   >= +5  -> tail capacity excess real: buckets worth running, MoE prior up
#   ~0 ±3  -> tail saturated: skip buckets; MoE rests on FT unsaturation alone
#   < 0    -> architecture axis closes for good
set -u
G=/work/home/scnmyrr3ge/luminex-dcu/gates
cd $G
ENG=$G/luminex-legacy-modern
T0=$G/t0_i8.nnue
WD100=$G/../session1/wd100_i8.nnue
FC=$G/fastchess-linux-x86-64/fastchess
RES=$G/wd100_results.txt
mkdir -p $G/pgn

[ -x "$ENG" ] && [ -s "$T0" ] && [ -s "$WD100" ] && [ -x "$FC" ] || { echo "MISSING_PREREQ" >> $RES; exit 1; }

for i in $(seq 1 20); do
  grep -q "CHUNK $i DONE" $RES 2>/dev/null && continue
  START=$((3001 + (i-1)*100))
  "$FC" \
    -engine cmd=$ENG name=wd100 option.NNUEFile=$WD100 option.Threads=1 option.Hash=64 \
    -engine cmd=$ENG name=T0    option.NNUEFile=$T0    option.Threads=1 option.Hash=64 \
    -each proto=uci tc=60+0 option.NodesPerMove=20000 \
    -openings file=$G/8moves_v3.pgn format=pgn order=sequential start=$START \
    -rounds 100 -repeat -concurrency 7 \
    -pgnout file=$G/pgn/wd100_$i.pgn > $G/wd100_$i.log 2>&1
  S=$(grep -E "^Games: " $G/wd100_$i.log | tail -1)
  if [ -z "$S" ]; then
    echo "CHUNK $i FAILED (no score line)" >> $RES
    tail -3 $G/wd100_$i.log >> $RES
    break
  fi
  echo "CHUNK $i: $S" >> $RES
  echo "CHUNK $i DONE" >> $RES
done
echo WD100_GATE_DONE >> $RES
cat $RES
