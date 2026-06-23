#!/bin/bash
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
probe () {  # schatten
  $PY experiments/e1_spd_decompose.py order3_pinwheel --C 80 --seed 0 --steps 20000 \
      --schatten "$1" --out results/spd_probe > "logs/probe_C80_sc$1.log" 2>&1
  $PY experiments/e1b_budget.py order3_pinwheel 80 0 --sc "$1" --pthdir results/spd_probe \
      --out-suffix "_probe_sc$1" >> logs/probe_analyze.log 2>&1
}
probe 0.1
probe 0.01
echo "[probe] ALL DONE at $(date)"
