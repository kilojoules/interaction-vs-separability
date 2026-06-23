# Steelman memo — what survived the attack

One line per task: **survived / weakened / strengthened / refuted**, then the
claim at the strength the new evidence licenses. This is the memo the proposal is
written from.

| # | task | verdict | one-line outcome |
|---|------|---------|------------------|
| 1 | run the real APD/SPD solver | **REVISED → no decomposition-axis effect (saturation withdrawn)** | Solver faithful on all 3 readouts; country resolves into a small bounded set everywhere (recon-95 ~2–7). The "saturation" (dominant-count tracking the budget) was a **confounded metric** — `argmax-over-8-outputs` is dominated by readout composition (phase readout is country-heavy by construction). A confound-free country-**only** decomposition shows the effective component count **plateaus far below the budget for every encoding** (no saturation). What remains: a *modest, bounded* dimensionality difference (phase PR ~11–17 vs poly ~8) and a phase-only faithfulness ceiling. The clean geometry signal is the **attribution-axis blindness** (Task 2), not the decomposition. |
| 2 | kill the by-construction confound | **SPLIT: inflation STRENGTHENED, blindness REFUTED-as-stated** | Inflation appears in a non-dedicated from-scratch cubic (24 units). But a homogeneous cubic `a³−3ab²` (AUC 0.979) is **not** first-order-blind, while `sin3θ` is — so blindness is a *phase-geometry* property, not an order property. |
| 3 | minimality/simplicity tradeoff | **DROPPED (no tension found)** | Faithfulness is flat across a 100× minimality sweep; component count varies modestly. The asserted tension does not appear — framing removed. |
| 4 | non-gradient / non-PCA metric | **SURVIVED** | The order-2 dissociation reproduces with single-unit vs causal-units (no grad/PCA): gap 0.37 vs original 0.44. |
| 5 | blind prediction on a new order | **MIXED** | Order-2 XOR on a new pair: predicted dissociation *appeared but attenuated* — direction right (gap 0.32, 4 units, reader rotates) yet weaker than predicted (probe 0.87 not ≈chance, conditional restore only 0.62). Order-4 `sin4θ`: untestable — SGD couldn't train it (0.565), corroborating Task 2. |

## What changed because of the attack

The original write-up bundled **two distinct effects** under "order-3 breaks
separability," and the steelman corrected the framing of both:

1. **No decomposition-axis effect (saturation/inflation both withdrawn).** Two
   successive metrics looked dramatic and both fell: "rank inflation" (2× / 38-of-40)
   was metric-fragile, and its replacement "budget saturation" (dominant-count
   tracks the budget) was **confounded by readout composition** — the phase
   readout is country-heavy by construction, so nearly every component is nominally
   country-dominant. The confound-free country-only decomposition (single output,
   participation-ratio spread metric) shows the effective component count
   **plateaus far below the budget for every encoding** — country resolves into a
   bounded set everywhere (recon-95 ~2–7). The clean residual is small: phase is a
   *modestly* higher-dimensional but bounded decomposition (~11–17 vs ~8 components)
   and uniquely resists faithful high-budget decomposition. **Lesson: every
   component-*count* metric we tried on the multi-output readout was confounded;
   only the country-only spread metric is clean.**
2. **First-order blindness (gradient tangentiality) — the load-bearing result.**
   *Not about order — about geometry.* Periodic/phase codes hide signal from
   first-order attribution; a homogeneous polynomial of the same degree does not
   (Euler: x·∇f = d·f). This is the clean, confound-free geometry-specific finding,
   directly tested on two degree-3 codes with opposite behavior.

## Final bounded claim (proposal seed) — held at exactly this strength

> **The geometry-specific effect is on the attribution axis, not the decomposition
> axis.** Run with the real APD solver, a feature's readout decomposes faithfully
> at every order and the feature resolves into a small, bounded set of components
> at every order and geometry (recon-95 ~2–7; the country-only effective-component
> count plateaus far below the budget) — there is **no saturation and no rank
> inflation**. What is phase-specific is that **first-order attribution is blind to
> phase/periodic codes** (the gradient is tangential to the level sets) while it
> fully recovers a homogeneous polynomial of the *same degree* (Euler's identity).
> A modest, bounded decomposition-dimensionality difference remains (phase spans
> somewhat more components), and the phase code uniquely resists *faithful*
> high-budget decomposition.

Plus a corroborated side-claim: order-2 *gated* features are not carried by one
stable component but become separable conditional on the interacting features
(survived a gradient-free, PCA-free metric; appeared attenuated on a held-out
gate). None of this is impossibility — faithful decomposition exists at every
order; country reconstructs from a handful of components regardless of geometry.

**Methodological lesson for the proposal:** component-*count* / "saturation"
metrics on a multi-output readout are confounded by readout composition; isolate
the feature (single-output decomposition) and use a threshold-free spread metric,
or stay on the attribution axis where the geometry signal is clean.

## What would move this from toy to real (proposal hooks)

- The Task-2 gap is the prerequisite: ordinary training does **not** readily
  produce clean high-order *phase* codes, so a naturally-occurring high-order
  feature (or a verified construction in a real LM) is needed before any scaled
  claim about phase-blindness.
- The clean, scalable screen is the **attribution-axis blindness**, not any
  component-count metric: does first-order attribution recover the feature, or is
  its gradient tangential to the level sets? It needs no budget sweep and was the
  one geometry signal that survived every confound check.
- **Do not use multi-output dominant-component or "saturation" metrics** — we
  showed twice that component-count metrics on a shared multi-output readout are
  confounded by readout composition. If a decomposition-axis count is wanted,
  isolate the feature (single-output decomposition) and use a threshold-free spread
  metric (participation ratio) vs budget; pair with recon-95 for "how many does it
  genuinely need."
