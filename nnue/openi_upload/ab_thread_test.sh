#!/bin/bash
# Interleaved A/B/A thread-scaling test for the correction-history worker fix.
# Usage: ab_thread_test.sh <engine_old> <engine_new> <net> [depth]
# One session, alternating builds per thread count to cancel session drift.
OLD=$1; NEW=$2; NET=$3; DEPTH=${4:-16}
FENS=(
"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K1R1 w KQkq - 0 1"
"r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
)

run_once () {  # $1=engine $2=threads -> prints nps
  local ENG=$1 T=$2 TN=0 OUT T0 T1 DT
  T0=$(date +%s%N)
  for fen in "${FENS[@]}"; do
    OUT=$(printf 'setoption name Threads value %d\nsetoption name Hash value 256\nsetoption name NNUEFile value %s\nposition fen %s\ngo depth %d\nquit\n' "$T" "$NET" "$fen" "$DEPTH" | "$ENG" 2>/dev/null | grep -oE "nodes [0-9]+" | tail -1 | awk '{print $2}')
    TN=$((TN + OUT))
  done
  T1=$(date +%s%N)
  DT=$(( (T1 - T0) / 1000000 ))
  echo $(( TN * 1000 / (DT > 0 ? DT : 1) ))
}

echo "depth=$DEPTH  A=$(basename $OLD)  B=$(basename $NEW)"
echo "threads  A_nps  A2_nps  B_nps"
for T in 1 2 4 8 12 14; do
  A=$(run_once "$OLD" $T)
  B=$(run_once "$NEW" $T)
  A2=$(run_once "$OLD" $T)
  echo "T=$T  $A  $A2  $B"
done
echo AB_DONE
