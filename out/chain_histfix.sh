#!/bin/bash
# Chain: wait for the wd100 gate to release the arena, then run the
# history-fix repair gate.
G=/work/home/scnmyrr3ge/luminex-dcu/gates
while ! grep -q WD100_GATE_DONE $G/wd100_results.txt 2>/dev/null; do
  sleep 60
done
exec bash $G/gate_histfix.sh
