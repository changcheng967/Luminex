#!/bin/bash
# Chain: wait for SESSION3_WD_DONE (wd100 training+harvest), then run the
# wd100-vs-T0 decision gate.
G=/work/home/scnmyrr3ge/luminex-dcu/gates
while ! grep -q SESSION3_WD_DONE /workspace/luminex/logs/session3.log 2>/dev/null; do
  sleep 60
done
# harvest check: the net must exist on NFS before gating
for i in $(seq 1 10); do
  [ -s $G/../session1/wd100_i8.nnue ] && break
  sleep 30
done
exec bash $G/gate_wd100.sh
