#!/bin/bash
# Orchestrates all remaining compute after the seed-0 SPD runs finish.
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
BASE=python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

echo "[orch] waiting for seed-0 SPD to finish..."
while pgrep -f "e1_spd_decompose.py .* --seed 0" >/dev/null; do sleep 10; done
echo "[orch] seed-0 done at $(date)"

# --- base-env quick training (Task 2b clean cubic, Task 5 blind) ---
USE_TF=0 $BASE experiments/e2b_clean_cubic.py     > logs/e2b.log 2>&1 &
USE_TF=0 $BASE experiments/e5_blind_prediction.py > logs/e5.log  2>&1 &

# --- Task 1 stability: seed-1 SPD for the 3 models (3 parallel) ---
for k in order1_vanilla order2_model order3_pinwheel; do
  $PY experiments/e1_spd_decompose.py $k --steps 10000 --seed 1 > logs/${k}_sd1.log 2>&1 &
done
wait
echo "[orch] seed-1 + base training done at $(date)"

# --- Task 3: minimality sweep (schatten 0.1, 1.0, 10.0) x (order2, order3), 6000 steps ---
mkdir -p results/spd_sweep
run_sweep () {  # model sc
  $PY experiments/e1_spd_decompose.py "$1" --steps 6000 --schatten "$2" \
      --out results/spd_sweep > "logs/sweep_$1_sc$2.log" 2>&1
}
run_sweep order2_model 0.1 &  run_sweep order3_pinwheel 0.1 &  run_sweep order2_model 1.0 &  wait
run_sweep order3_pinwheel 1.0 &  run_sweep order2_model 10.0 &  run_sweep order3_pinwheel 10.0 &  wait
echo "[orch] sweep done at $(date)"

# --- figures ---
$PY experiments/e1_figure.py        > logs/e1_figure.log 2>&1
$PY experiments/e3_sweep_figure.py  > logs/e3_figure.log 2>&1
echo "[orch] ALL DONE at $(date)"
