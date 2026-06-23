#!/bin/bash
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
# full cores per run, sequential -> fast per-step; 20k steps to restore faithfulness
tr () { $PY experiments/e1_spd_decompose.py order3_pinwheel --C "$1" --seed "$2" --steps 20000 \
        --out results/spd_hi > "logs/hi2_order3_C$1_sd$2.log" 2>&1
        $PY experiments/e1b_budget.py order3_pinwheel "$1" "$2" --pthdir results/spd_hi --out-suffix _hi >> logs/hi2_analyze.log 2>&1; }
tr 80 0
tr 120 0
tr 80 1
echo "[hi2] ALL DONE at $(date)"
