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

> **The clean geometry-specific effect is on the *attribution* axis, not the
> decomposition axis.** Run with the real APD solver, the country readout
> decomposes **faithfully** at every order and the feature **resolves into a small,
> bounded set of components at every order and geometry** (recon-95 ~2–7; the
> country-only effective-component count plateaus far below the budget) — **no
> saturation, no rank inflation**. What *is* phase-specific is that **first-order
> attribution is blind to phase/periodic codes** (the gradient is tangential to the
> level sets) while it fully recovers a homogeneous polynomial of the *same degree*.
> A modest, bounded dimensionality difference remains (phase spans somewhat more
> components), and the phase code uniquely resists *faithful* high-budget
> decomposition. Side-claim: the order-2 gated feature is separable only
> **conditional on the features it interacts with**.
>
> *(An earlier version of this claim asserted budget "saturation" on the
> decomposition axis; that rested on a `dominant = argmax-over-8-outputs` metric
> confounded by readout composition, and is **withdrawn** — see FINDINGS Task 1.)*

See `FINDINGS.md` for the headline tables and `LIMITATIONS.md` for what this does
and does not show.

## Which solver, and which axis (methods)

**Which solver?** — All decomposition-axis numbers come from **Apollo's
`optimize()` in the `spd` package** (`github.com/ApolloResearch/apd`), i.e. the
real **APD** (Attribution-based Parameter Decomposition) algorithm: each readout
weight is decomposed into `C` subnetwork components (W = Σ_c A_cB_c), trained with
param-match + top-k reconstruction + activation reconstruction + Schatten
minimality, `attribution_type="gradient"`
(`experiments/e1_spd_decompose.py`, `e1b_budget.py`, `e1e_e2b_saturation.py`,
`e1f_country_only.py`). The parameter-Jacobian "reader" fallback
(`src/separability.py`) is used **only** for the first-order/attribution-axis
metrics (blindness, reader alignment, nonlinear-unit count in Tasks 2/4/5).
Component effects are read off the *trained* APD components by **causal ablation**
(`set_subnet_to_zero`), not gradients.

**Two metrics, and why one was withdrawn.** A `dominant = argmax-over-8-outputs`
metric showed country tracking the budget at order-3 (95–100% of C) — but it is
**confounded by readout composition** (the phase model's readout is country-heavy
by construction, so nearly every component is nominally country-dominant). The
confound-free re-test decomposes the **country-only readout** (single output, no
argmax) and measures the participation ratio of per-component country effect: it
**plateaus far below the budget for every encoding** (`figs/spd_country_only.png`),
so there is **no saturation**. The `dominant`/redundancy result (incl. the e2b
"redundancy 0 vs 7–27" contrast) is therefore **withdrawn** as confounded and kept
only for completeness in `figs/spd_budget_sweep.png`. The clean decomposition-axis
finding is just: faithful + bounded resolution (~2–7 components) everywhere; the
geometry-specific signal is the attribution-axis blindness.

## What's here

```
src/
  separability.py     first-order instruments (readers, fixed-reader, first-order
                      recon, nonlinear unit count) — the cheap parameter-Jacobian view
  spd_readout.py      wrap a readout as an ApolloResearch `spd` target + SPD twin
  spd_analyze.py      causal (non-gradient) analysis of a trained SPD decomposition
experiments/
  e1_spd_decompose.py  run the REAL APD/SPD solver on a readout (Task 1)
  e1b_budget.py /      budget sweep C=40/80/120 (dominant-count: CONFOUNDED metric,
  e1c_budget_summary.py  withdrawn) -> figs/spd_budget_sweep.png
  e1f_country_only.py / confound-free country-only decomposition (PR vs budget)
  e1g_country_only_figure.py  -> figs/spd_country_only.png  (no saturation)
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

- **The central geometry-specific result is the attribution-axis blindness**, not
  a decomposition-axis count. On the decomposition axis we find only: APD is
  faithful at every order, and the feature resolves into a small bounded set of
  components everywhere (recon-95 ~2–7; country-only effective-component count
  plateaus far below the budget — `figs/spd_country_only.png`). **No saturation, no
  inflation.** A modest, bounded dimensionality difference and a phase-only
  faithfulness ceiling remain.
- **We withdrew two successive decomposition-axis claims as confounded:** "2× /
  38-of-40 rank inflation" (metric-fragile), then "budget saturation" (the
  `dominant = argmax-over-8-outputs` count is driven by readout composition, not
  geometry). The confound-free country-only re-test shows no saturation. Lesson:
  **component-count metrics on a shared multi-output readout are confounded** —
  isolate the feature or stay on the attribution axis.
- **The order-2 *gated* dissociation is a corroborated side-claim**, reproduced by
  three independent instruments: the first-order reader (gradient+PCA), a causal
  hidden-unit analysis (no gradients, no PCA), and the real APD/SPD solver.
- Every place the plan met the code's reality is logged in `FINDINGS.md`
  (Divergences) and bounded in `LIMITATIONS.md`.
