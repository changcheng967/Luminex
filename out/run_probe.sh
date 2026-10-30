#!/bin/bash
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/.hyhal/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
cd /tmp
python3 icc_probe.py 2>&1 | grep -viE "warn|futur"
