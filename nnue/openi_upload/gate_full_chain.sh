#!/bin/bash
# Gates for gen768sf_full vs sf1 (the v6.2.0 net): fixed-node first (pure
# quality, immune to scale/nps), then fishtest TC. Scale probe passed
# (q50 ratio 0.963); tail compression noted, SPSA retune pending if promoted.
CC=/usr/local/bin/cutechess-cli
ENG=/mnt/workspace/lichex-engine
N=/mnt/workspace/r1
NEW=$N/gen768sf_full_i8.nnue
SF1=$N/r1sf_dcu_pilot_i8.nnue

$CC -engine name=full cmd=$ENG proto=uci option.UseNNUE=true option.NNUEFile=$NEW \
    -engine name=sf1 cmd=$ENG proto=uci option.UseNNUE=true option.NNUEFile=$SF1 \
    -each proto=uci tc=60+0 option.NodesPerMove=20000 option.Threads=1 option.Hash=64 \
    -rounds 300 -repeat -concurrency 8 \
    -pgnout $N/gatefull_fixed.pgn > $N/gatefull_fixed.log 2>&1
echo "FIXED: $(grep 'Score of' $N/gatefull_fixed.log | tail -1)" >> $N/gatefull.done

$CC -engine name=full cmd=$ENG proto=uci option.UseNNUE=true option.NNUEFile=$NEW \
    -engine name=sf1 cmd=$ENG proto=uci option.UseNNUE=true option.NNUEFile=$SF1 \
    -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=64 \
    -rounds 100 -repeat -concurrency 8 \
    -pgnout $N/gatefull_tc.pgn > $N/gatefull_tc.log 2>&1
echo "TC: $(grep 'Score of' $N/gatefull_tc.log | tail -1)" >> $N/gatefull.done
echo GATES_DONE >> $N/gatefull.done
