#!/bin/bash
# Budget sweep for the rank-inflation claim: order-2/3 at C in {80,120}, seeds {0,1}.
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

train () { $PY experiments/e1_spd_decompose.py "$1" --C "$2" --seed "$3" --steps 10000 \
           > "logs/budget_$1_C$2_sd$3.log" 2>&1; }

# train in waves of 3 (8 jobs)
train order2_model 80 0 &  train order3_pinwheel 80 0 &  train order2_model 80 1 &  wait
train order3_pinwheel 80 1 &  train order2_model 120 0 &  train order3_pinwheel 120 0 &  wait
train order2_model 120 1 &  train order3_pinwheel 120 1 &  wait
echo "[budget] training done at $(date)"

# uniform analysis for every (model, C, seed) including the existing C=40
for C in 40 80 120; do for m in order2_model order3_pinwheel; do for s in 0 1; do
  $PY experiments/e1b_budget.py $m $C $s >> logs/budget_analyze.log 2>&1
done; done; done
echo "[budget] ALL DONE at $(date)"
