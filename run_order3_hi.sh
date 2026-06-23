#!/bin/bash
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
tr () { $PY experiments/e1_spd_decompose.py order3_pinwheel --C "$1" --seed "$2" --steps 30000 \
        --out results/spd_hi > "logs/hi_order3_C$1_sd$2.log" 2>&1; }
tr 80 0 & tr 80 1 & tr 120 0 & tr 120 1 & wait
echo "[hi] training done at $(date)"
for C in 80 120; do for s in 0 1; do
  $PY experiments/e1b_budget.py order3_pinwheel $C $s --pthdir results/spd_hi --out-suffix _hi >> logs/hi_analyze.log 2>&1
done; done
echo "[hi] ALL DONE at $(date)"
