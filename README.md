# Does feature interaction degrade the separability of parameter decomposition?

A toy-scale, ground-truth-known **steelman** of an interpretability claim: that as
the *interaction order* of a feature's encoding rises, it becomes harder to
decompose the responsible weights into a small set of stable, feature-aligned
**parameter components** — even while reconstruction stays cheap.

The setup is the BlueDot "pinwheel" puzzle head: a frozen MiniLM encoder + a small
MLP that predicts 8 binary text features. Seven features are linear at the analyzed
layer **L**; one (country) is re-encoded with controllable interaction order:

| order | encoding | country logit is … |
|---|---|---|
| 1 | linear | one direction in L |
| 2 | gated sign-flip on `food ⊕ sentiment` | a reader that switches with a 2-feature gate |
| 3 | phase parity `sign(sin 3θ)` | a cubic function of a 2-D carrier plane |

The decomposition **target** is the readout stack `net.layers[6:]`
(`L(64) → 64 → ReLU → 8 logits`); in the order-3 model the nonlinear country
machinery lives entirely there.

## The claim, stated at the strength the evidence licenses

> For a **phase/periodic** encoding, attribution-based parameter decomposition
> (APD/SPD) does **not resolve the feature into a bounded set of components** — the
> count of components dominated by it tracks the total budget (95–100% across
> C = 40/80/120), whereas a lower-order **gated** feature resolves into a stable
> ~15%. This is **not** a reconstruction-cost effect (both reconstruct from ~5–6
> components): it is a failure of the *attribution/selection* step, specific to
> the phase geometry. Two corroborated side-claims: the gated feature is separable
> only **conditional on the features it interacts with**; and first-order
> attribution is blind **only** to phase/periodic codes, not to high-order codes
> in general (a homogeneous polynomial of the same degree is fully recoverable).
> None of this is impossibility — faithful decomposition exists at every order;
> what fails at order-3 is selection, not reconstruction.

See `FINDINGS.md` for the headline tables and `LIMITATIONS.md` for what this does
and does not show.

## What's here

```
src/
  separability.py     first-order instruments (readers, fixed-reader, first-order
                      recon, nonlinear unit count) — the cheap parameter-Jacobian view
  spd_readout.py      wrap a readout as an ApolloResearch `spd` target + SPD twin
  spd_analyze.py      causal (non-gradient) analysis of a trained SPD decomposition
experiments/
  e1_spd_decompose.py  run the REAL APD/SPD solver on a readout (Task 1)
  e1b_budget.py /      budget sweep C=40/80/120: recon-95 vs dominant-count
  e1c_budget_summary.py  -> figs/spd_budget_sweep.png  (the saturation result)
  e1_figure.py         aggregate Task-1 runs -> figs/spd_components.png
  e2_train_nondedicated.py / e2b_clean_cubic.py  non-dedicated order-3 (Task 2)
  e3_sweep_figure.py   minimality sweep (Task 3 — no tension found)
  e4_nongradient_gap.py  re-derive the gap with no gradients / no PCA (Task 4)
  e5_blind_prediction.py blind prediction on new structures (Task 5)
models/   trained checkpoints + the training scripts that produce them
results/  JSON outputs per experiment       figs/  figures
FINDINGS.md  LIMITATIONS.md  STEELMAN_MEMO.md
```

## Reproduce

The cheap first-order experiments (e2, e4, e5) run in any recent Python with
numpy/torch/scikit-learn:

```bash
USE_TF=0 python experiments/e4_nongradient_gap.py        # Task 4 (no training)
USE_TF=0 python experiments/e2b_clean_cubic.py           # Task 2
USE_TF=0 python experiments/e5_blind_prediction.py       # Task 5
```

The real **APD/SPD** experiments (e1, e3) need the ApolloResearch `spd` package
(`github.com/ApolloResearch/apd`, Python ≥ 3.11) in its own env:

```bash
git clone https://github.com/ApolloResearch/apd && (cd apd && pip install -e .)
APD_PY=/path/to/apd-env/bin/python
$APD_PY experiments/e1_spd_decompose.py order3_pinwheel --steps 10000   # decompose a readout
$APD_PY experiments/e1_figure.py                                        # headline figure
```

Checkpoints and cached MiniLM embeddings are produced by the BlueDot puzzle repo
(`bluedot-tais-puzzle`); the experiment scripts read them via the `BDC` path at
the top of each file — point it at your checkout.

## Method notes / honesty

- **The central (order-3) result is budget *saturation*, measured budget-robustly.**
  We sweep the total component budget (C = 40/80/120, 2 seeds) and track two
  metrics: *recon-95* (components needed to reconstruct the feature to 95% of full
  AUC) and the *dominant-component count*. recon-95 is flat at ~5–6 for both
  features; the dominant count tracks the budget (95–100%) only for the phase
  feature, vs a stable ~15% for the gated one. Saturation is invariant to a 100×
  minimality (Schatten) sweep and a 2× step sweep — so it is **not** a
  regularization or underfitting artifact, only a clean faithful anchor (C=40).
- **The order-2 *gated* dissociation is a corroborated side-claim**, reproduced by
  three independent instruments: the first-order reader (gradient+PCA), a causal
  hidden-unit analysis (no gradients, no PCA), and the real APD/SPD solver.
- We measure separability on axes that **do not reduce to rank** (reader
  stability, a fixed-vs-flexible reconstruction gap, causal single-component
  ablation, SPD component counts) — and we **withdrew** the earlier
  "2× / 38-of-40 rank inflation" framing as metric-fragile.
- Every place the plan met the code's reality is logged in `FINDINGS.md`
  (Divergences) and bounded in `LIMITATIONS.md`.
