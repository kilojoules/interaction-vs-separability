#!/bin/bash
# Stress-test the phase faithfulness ceiling. Waits for the in-flight XOR-3
# country-only run, then sweeps configs 3-at-a-time. Pure-reconstruction configs
# (default: schatten none, topk none, unit_norm 0) test representability; the
# --schatten/--topk/--unit_norm-on configs test the standard APD recipe's ceiling.
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
: > logs/stress.log
echo "[stress] waiting for in-flight xor3 country-only..."
while pgrep -f "e1f_country_only.py order3_xor" >/dev/null; do sleep 15; done
echo "[stress] starting sweep $(date)"
run () { $PY experiments/e7_phase_faithfulness_stress.py "$@" >> logs/stress.log 2>&1; }

# ---- PURE-RECONSTRUCTION (representability test) ----
run --C 40 --steps 20000 --lr 1e-3 &  run --C 40 --steps 20000 --lr 3e-3 &  run --C 40 --steps 20000 --lr 1e-2 &  wait
run --C 40 --steps 40000 --lr 3e-3 &  run --C 80 --steps 20000 --lr 3e-3 &  run --C 120 --steps 20000 --lr 3e-3 &  wait
run --C 40 --steps 20000 --lr 3e-3 --seed 1 &  run --C 40 --steps 20000 --lr 3e-3 --seed 2 &  run --C 40 --steps 20000 --lr 3e-3 --unit_norm 1 &  wait
run --C 40 --steps 20000 --lr 3e-3 --init_scale 2.0 --init_type kaiming_uniform &  run --C 40 --steps 20000 --lr 3e-3 --m 1 &  run --C 40 --steps 40000 --lr 3e-3 --pm 10 &  wait
# ---- STANDARD APD RECIPE + levers (the original ceiling) ----
run --C 40 --steps 30000 --lr 1e-3 --unit_norm 1 --schatten 1.0 --topk 10 --sched cosine &  run --C 40 --steps 50000 --lr 1e-3 --unit_norm 1 --schatten 1.0 --topk 10 --sched cosine &  run --C 80 --steps 50000 --lr 1e-3 --unit_norm 1 --schatten 1.0 --topk 10 --sched cosine &  wait
run --C 40 --steps 30000 --lr 3e-3 --unit_norm 1 --schatten 1.0 --topk 10 &  run --C 40 --steps 20000 --lr 1e-3 --schatten 1.0 --topk 10 &  run --C 40 --steps 20000 --lr 1e-3 --unit_norm 1 --schatten 0.1 --topk 10 --pm 10 &  wait
run --C 80 --steps 30000 --lr 3e-3 --schatten 1.0 --topk 10 &  run --C 40 --steps 50000 --lr 3e-3 --pm 10 &  run --C 80 --steps 50000 --lr 3e-3 --pm 10 &  wait
# ---- MULTI-OUTPUT cross-check ----
run --C 40 --steps 20000 --lr 3e-3 --multi 1 &  run --C 40 --steps 30000 --lr 1e-3 --unit_norm 1 --schatten 1.0 --topk 10 --multi 1 &  wait
echo "[stress] ALL DONE $(date)"
