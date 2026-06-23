#!/bin/bash
# Faithfulness-vs-parsimony test (user's conjecture): as we push the phase code
# toward faithfulness, does the effective # of components (PR) blow up — i.e. does
# it need a high-rank interpolation — while the poly/XOR codes stay faithful at low
# PR?  Sweep training steps (pure-reconstruction, C=40, lr=1e-3) per code and read
# PR / recon-95 at each faithfulness level.
set -u
cd /Users/julianquick/bdc/interaction-vs-separability
PY=/Users/julianquick/miniconda3/envs/apd/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
echo "[conj] waiting for main stress sweep to finish..."
while ! grep -q "ALL DONE" logs/stress_orch.log 2>/dev/null; do sleep 20; done
echo "[conj] starting $(date)"
: > logs/conjecture.log
run () { $PY experiments/e7_phase_faithfulness_stress.py --C 40 --lr 1e-3 "$@" >> logs/conjecture.log 2>&1; }

# phase: span faithfulness via steps
run --model phase --steps 2000  &  run --model phase --steps 5000  &  run --model phase --steps 10000 &  wait
run --model phase --steps 20000 &  run --model phase --steps 40000 &  run --model poly  --steps 3000  &  wait
# poly + xor controls (reach faithful fast, should stay low PR)
run --model poly  --steps 20000 &  run --model xor   --steps 3000  &  run --model xor   --steps 20000 &  wait
run --model poly  --steps 40000 &  run --model xor   --steps 40000 &  wait
echo "[conj] ALL DONE $(date)"
