#!/bin/bash
# Pareto-area test: identical faithful config (C=40, lr=1e-3, pure-recon, 20k) for
# each code; saves the recon-curve (AUC vs #components = the minimality-faithfulness
# frontier) and pareto_area (normalized deficit). Hypothesis: phase area is largest.
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
echo "[pareto] waiting for main stress sweep..."
while ! grep -q "ALL DONE" logs/stress_orch.log 2>/dev/null; do sleep 20; done
echo "[pareto] starting $(date)"; : > logs/pareto.log
run () { $PY experiments/e7_phase_faithfulness_stress.py --C 40 --steps 20000 --lr 1e-3 --model "$1" >> logs/pareto.log 2>&1; }
run phase &  run poly &  wait
run xor   &  run gate &  wait
echo "[pareto] ALL DONE $(date)"
