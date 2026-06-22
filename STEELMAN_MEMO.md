# Steelman memo — what survived the attack

One line per task: **survived / weakened / strengthened / refuted**, then the
claim at the strength the new evidence licenses. This is the memo the proposal is
written from.

| # | task | verdict | one-line outcome |
|---|------|---------|------------------|
| 1 | run the real APD/SPD solver | **STRENGTHENED** | Solver is faithful on all 3 readouts; country's component share rises with order — 2× a linear feature and 38/40 components at order-3 (rank inflation, real). |
| 2 | kill the by-construction confound | **SPLIT: inflation STRENGTHENED, blindness REFUTED-as-stated** | Inflation appears in a non-dedicated from-scratch cubic (24 units). But a homogeneous cubic `a³−3ab²` (AUC 0.979) is **not** first-order-blind, while `sin3θ` is — so blindness is a *phase-geometry* property, not an order property. |
| 3 | minimality/simplicity tradeoff | **DROPPED (no tension found)** | Faithfulness is flat across a 100× minimality sweep; component count varies modestly. The asserted tension does not appear — framing removed. |
| 4 | non-gradient / non-PCA metric | **SURVIVED** | The order-2 dissociation reproduces with single-unit vs causal-units (no grad/PCA): gap 0.37 vs original 0.44. |
| 5 | blind prediction on a new order | **MIXED** | Order-2 XOR on a new pair: dissociation confirmed (gap 0.32, 4 units), but the gate was *softer* than predicted (probe 0.87, partial conditional restore). Order-4 `sin4θ`: untestable — SGD couldn't train it (0.565), corroborating Task 2. |

## What changed because of the attack

The original write-up bundled **two distinct effects** under "order-3 breaks
separability." The steelman separates them:

1. **Rank inflation (cost).** *Robust, architecture-independent, solver-confirmed.*
   Higher-complexity features need more parameter components. This is the load-
   bearing, defensible claim.
2. **First-order blindness (gradient tangentiality).** *Not about order — about
   geometry.* Periodic/phase codes hide signal from first-order attribution;
   homogeneous polynomial codes of the same degree do not. The original claim is
   corrected, not retracted: blindness is real for the *phase* family the puzzle
   used, but is not a generic consequence of interaction order.

## Final bounded claim (proposal seed)

> In a toy model with known ground truth, raising a feature's
> interaction/computational complexity **inflates the cost** of an
> attribution-based parameter decomposition: the real APD/SPD solver stays
> faithful but spends a growing share of components on the complex feature (2× a
> linear feature at order-3), and this reproduces in a non-dedicated,
> from-scratch model. Separately, order-2 *gated* features are not carried by one
> stable component but become separable once you condition on the interacting
> features — robust to a gradient-free, PCA-free metric. We **withdraw** the
> stronger claim that high interaction order makes a feature invisible to
> first-order attribution: that holds only for *phase/periodic* encodings (where
> the gradient is tangential), not for homogeneous polynomial encodings of the
> same degree. None of this is impossibility — faithful decomposition exists at
> every order; the **budget** is what grows.

## What would move this from toy to real (proposal hooks)

- The Task-2 gap is the prerequisite: ordinary training does **not** readily
  produce clean high-order *phase* codes, so a naturally-occurring high-order
  feature (or a verified construction in a real LM) is needed before any scaled
  claim about phase-blindness.
- The robust, scalable screen is **rank inflation**: run APD/SPD on a real small
  LM where an interaction feature is independently localized and test whether
  component count tracks measured interaction order. The non-gradient causal-unit
  metric (Task 4) is the cheap pre-screen that needs no solver.
- Distinguish *cost* (component count, measurable everywhere) from *blindness*
  (encoding-geometry-specific) in any scaled study — do not conflate them again.
