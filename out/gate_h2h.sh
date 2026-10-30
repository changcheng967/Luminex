#!/bin/bash
# d2 vs h300 — the redundancy-screen DECISION gate. 2,000 games in 10 chunks
# of 200 (pod-death resilient: each chunk's score appends to h2h_results.txt
# and completed chunks are skipped on relaunch).
# Pre-registered rule: h300 >= 20 Elo worse than d2 => position-redundancy is
# real at 2x, build the 2B strict-superset corpus; tie => the data axis is
# closed and the dump program is saved.
# Settings identical to the session-1 gates: same engine binary both sides,
# fixed NodesPerMove=20000, tc wrapper 60+0, book 8moves_v3, colors swapped.
set -u
G=/work/home/scnmyrr3ge/luminex-dcu/gates
cd $G
ENG=$G/luminex-legacy-modern   # glibc-2.35 static-stdlib CI artifact
D2=$G/../session1/d2_i8.nnue
H300=$G/../session1/h300_i8.nnue
FC=$G/fastchess-linux-x86-64/fastchess
RES=$G/h2h_results.txt
mkdir -p $G/pgn

[ -x "$ENG" ] || { echo "ENGINE MISSING" >> $RES; exit 1; }
[ -s "$D2" ] && [ -s "$H300" ] || { echo "NET MISSING" >> $RES; exit 1; }
[ -x "$FC" ] || { echo "FASTCHESS MISSING" >> $RES; exit 1; }

for i in $(seq 1 10); do
  grep -q "CHUNK $i DONE" $RES 2>/dev/null && continue
  START=$((1 + (i-1)*100))   # 100 distinct openings per chunk, colors swapped
  "$FC" \
    -engine cmd=$ENG name=d2   option.NNUEFile=$D2   option.Threads=1 option.Hash=64 \
    -engine cmd=$ENG name=h300 option.NNUEFile=$H300 option.Threads=1 option.Hash=64 \
    -each proto=uci tc=60+0 option.NodesPerMove=20000 \
    -openings file=$G/8moves_v3.pgn format=pgn order=sequential start=$START \
    -rounds 100 -repeat -concurrency 7 \
    -pgnout file=$G/pgn/h2h_$i.pgn > $G/h2h_$i.log 2>&1
  S=$(grep -E "^Games: " $G/h2h_$i.log | tail -1)
  if [ -z "$S" ]; then
    echo "CHUNK $i FAILED (no score line)" >> $RES
    tail -5 $G/h2h_$i.log >> $RES
    break
  fi
  echo "CHUNK $i: $S" >> $RES
  echo "CHUNK $i DONE" >> $RES
done
echo H2H_GATE_DONE >> $RES
cat $RES
