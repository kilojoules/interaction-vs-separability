#!/bin/bash
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
r () { $PY experiments/e1f_country_only.py "$1" "$2" >> logs/country_only.log 2>&1; }
for C in 40 80 120; do
  r order2_model $C & r order3_pinwheel $C & r e2b_cubic $C & wait
done
echo "[country-only] ALL DONE at $(date)" >> logs/country_only.log
