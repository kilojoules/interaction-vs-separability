# Steelman memo — what survived the attack

One line per task: **survived / weakened / strengthened / refuted**, then the
claim at the strength the new evidence licenses. This is the memo the proposal is
written from.

| # | task | verdict | one-line outcome |
|---|------|---------|------------------|
| 1 | run the real APD/SPD solver | **REFRAMED → budget saturation, not cost** | Solver faithful on all 3 readouts. The order-3 effect is *saturation*: country-dominated components track the budget ceiling (**95–100%** across C=40/80/120, both seeds) vs a stable **~15%** for the order-2 gate — but **recon-95 (genuine need) is flat at ~5–6 for both**. So it's a failure of the attribution/*selection* step to find bounded structure, specific to the phase geometry — not a higher reconstruction cost. (The earlier "2× / 38-of-40 rank inflation" framing was metric-fragile, withdrawn.) |
| 2 | kill the by-construction confound | **SPLIT: inflation STRENGTHENED, blindness REFUTED-as-stated** | Inflation appears in a non-dedicated from-scratch cubic (24 units). But a homogeneous cubic `a³−3ab²` (AUC 0.979) is **not** first-order-blind, while `sin3θ` is — so blindness is a *phase-geometry* property, not an order property. |
| 3 | minimality/simplicity tradeoff | **DROPPED (no tension found)** | Faithfulness is flat across a 100× minimality sweep; component count varies modestly. The asserted tension does not appear — framing removed. |
| 4 | non-gradient / non-PCA metric | **SURVIVED** | The order-2 dissociation reproduces with single-unit vs causal-units (no grad/PCA): gap 0.37 vs original 0.44. |
| 5 | blind prediction on a new order | **MIXED** | Order-2 XOR on a new pair: predicted dissociation *appeared but attenuated* — direction right (gap 0.32, 4 units, reader rotates) yet weaker than predicted (probe 0.87 not ≈chance, conditional restore only 0.62). Order-4 `sin4θ`: untestable — SGD couldn't train it (0.565), corroborating Task 2. |

## What changed because of the attack

The original write-up bundled **two distinct effects** under "order-3 breaks
separability," and the steelman corrected the framing of both:

1. **Budget saturation, not cost.** The budget sweep killed the "rank inflation"
   reading: country reconstructs from ~5–6 components at *every* order, so it is
   not intrinsically more expensive. What is order-3-specific is that the
   *attribution/selection* step never resolves the phase feature into a bounded
   set — its dominant-component count tracks whatever budget it is given
   (95–100%), vs ~15% for the order-2 gate. The defensible claim is about
   **selection failure specific to phase geometry**, not cost.
2. **First-order blindness (gradient tangentiality).** *Not about order — about
   geometry.* Periodic/phase codes hide signal from first-order attribution;
   homogeneous polynomial codes of the same degree do not. Corrected, not
   retracted: blindness is real for the *phase* family the puzzle used, but is not
   a generic consequence of interaction order.

## Final bounded claim (proposal seed) — held at exactly this strength

> For a phase/periodic encoding, attribution-based parameter decomposition does
> not resolve the feature into a bounded set of components: the count of
> components dominated by the feature tracks the total budget (95–100% across
> C = 40/80/120, both seeds), whereas a gated feature of lower order resolves into
> a stable ~15%. Notably this is **not** a reconstruction-cost effect — both
> features reconstruct from ~5–6 components — so it is a failure of the
> attribution/selection step to find stable structure, specific to the phase
> geometry, not a statement that the feature is intrinsically more expensive.

Plus two corroborated side-claims: order-2 *gated* features are not carried by one
stable component but become separable conditional on the interacting features
(survived a gradient-free, PCA-free metric; appeared attenuated on a held-out
gate); and first-order attribution is blind only to *phase/periodic* codes, not to
high-order codes in general. None of this is impossibility — faithful
decomposition exists at every order; what fails at order-3 is *selection*, not
reconstruction.

## What would move this from toy to real (proposal hooks)

- The Task-2 gap is the prerequisite: ordinary training does **not** readily
  produce clean high-order *phase* codes, so a naturally-occurring high-order
  feature (or a verified construction in a real LM) is needed before any scaled
  claim about phase-blindness.
- The scalable screen is **selection stability, not component count**: run APD/SPD
  at ≥2 budgets and test whether a feature's dominant-component count is a stable
  fraction of the budget (resolves) or tracks the ceiling (saturates). Pair it
  with recon-95 to separate "expensive" from "unresolved."
- Distinguish three things in any scaled study and do not conflate them:
  *reconstruction cost* (recon-95, measurable everywhere), *selection/saturation*
  (dominant-fraction vs budget), and *blindness* (encoding-geometry-specific).
