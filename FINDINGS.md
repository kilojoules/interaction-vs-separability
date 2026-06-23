# FINDINGS — steelman of interaction-vs-separability

We attacked an earlier first-order result with the **real** APD/SPD solver and
four adversarial controls. The attack changed the claim: one part **survived and
strengthened** (rank inflation), one part was **refuted as stated** (first-order
blindness is not a generic property of interaction order — it is specific to
*phase/periodic* codes), one asserted tension **did not appear** (dropped), and
the order-2 dissociation **survived** a gradient-free re-derivation.

## The earlier (first-order) result being attacked

A parameter-Jacobian "reader" view (`figs/firstorder_order_axis.png`) found
country's separability degrading with interaction order while reconstruction
stayed cheap: order-1 separable; order-2 a fixed component fails (0.52) but rank-2
per-input recombination works (0.96); order-3 the first-order reader goes "blind"
(`<r,L>`=0.53) and needs ~32 nonlinear units. The steelman asks whether those
claims survive the real solver and honest de-confounding.

---

## Task 1 — run the actual APD/SPD solver · **REVISED: no decomposition-axis effect (saturation withdrawn)**

Each readout (`layers[6:]`, `L→64→ReLU→8`) was wrapped as an ApolloResearch `spd`
target + SPD twin (each weight = Σ_C A_cB_c) and decomposed with the real
`optimize()` (param-match + topk-recon + act-recon + Schatten minimality), inputs
= real L activations. `figs/spd_components.png`.

| order | country AUC target→SPD | recon err | components: country / mean-linear | comps dominated by country | cross-seed MMCS |
|---|---|---|---|---|---|
| 1 linear | 1.000→1.000 | 0.2% | 8 / 8  (1.0×) | 13 / 40 | 0.16 |
| 2 gated  | 0.994→0.993 | 0.7% | 8 / 8  (1.0×) | 5 / 40  | 0.21 |
| 3 cubic  | 0.979→0.949 | 0.8% | **6 / 3  (2.0×)** | **38 / 40** | 0.74 |

- The solver **reaches faithfulness on all three readouts** — this is about how the
  computation is decomposed, not feasibility. (Closes the fallback gap.)
- **The C=40 snapshot is metric-fragile — and the metric that looked dramatic is
  confounded.** It *looks* like inflation (country is the argmax of 38/40
  components, "2×" the serves of a linear feature), but "dominant = argmax over 8
  outputs" is confounded by readout composition, and a confound-free re-test
  (below) shows **no saturation and no inflation** on the decomposition axis. The
  held claim is stated under the sweep.
- **MMCS wrinkle (honest):** order-3 components are *more* cross-seed reproducible
  (0.74) than order-1/2 (0.16–0.21). This is **not** "higher order = less stable";
  it reflects (a) order-3's dedicated phase bank and (b) over-decomposition at
  orders 1–2 (C=40 ≫ ~8 mechanisms → interchangeable fragments). We do **not**
  claim a stability degradation.

### Budget sweep — and a confound that withdraws the saturation claim

We first measured a "country-dominant" component count (argmax output = country)
and found it tracked the budget at order-3 (38→79→120 of C, "95–100%") vs ~15% for
the gate, and built a "redundancy" story (dominant ÷ recon-95) on top of it. **That
result is withdrawn.** The `dominant = argmax-over-8-outputs` metric is confounded
by **readout composition**: the pinwheel readout is *mostly country by
construction* (it is almost all phase-bank), so nearly every component affects
country more than any other feature — even components with *negligible* country
effect (hence dominant=120 while recon-95=4: ~115 components are redundant *for
country* but still argmax-country). The polynomial control "confirming"
phase-specificity (e2b dominant≈0) was the same artifact in reverse: e2b is a
*balanced* head where country is ≈1/8 of the readout. So that comparison measured
readout composition, not geometry.

**Confound-free re-test (`experiments/e1f_country_only.py`, `figs/spd_country_only.png`).**
We decomposed the **country-only readout** (64→64→1, a single output, so "dominant"
cannot even be defined) and measured a threshold-free spread metric: the
participation ratio **PR** of the per-component causal country effect (effective
number of components carrying country). Does PR track the budget (saturation) or
plateau (bounded)?

| effective components PR | C=40 | C=80 | C=120 | faithful? |
|---|---|---|---|---|
| order-2 gate | 7.1 | 11.9 | 12.5 | ✓ (0.99) |
| order-3 **phase** `sin3θ` | 10.8 | 17.4 | 16.5 | ✗ (0.90 → 0.68) |
| order-3 **polynomial** `a³−3ab²` | 7.7 | 7.6 | 9.0 | ✓ (0.98) |

- **No saturation.** PR **plateaus far below the budget ceiling** for all three —
  PR/C *declines* in every case (gate 0.18→0.10, phase 0.27→0.14, poly 0.19→0.07).
  Country resolves into a bounded set everywhere; recon-95 is ~2–7. The
  "saturation" was entirely the cross-feature/readout-composition artifact above.
- **A modest, bounded dimensionality difference remains:** the phase code spans
  somewhat more components (PR ~11–17) than the polynomial (~8) or gate (~12) — but
  bounded, not budget-tracking, and partly confounded by the phase faithfulness
  ceiling.
- **The phase code uniquely resists faithful high-budget decomposition** (country
  AUC 0.68 at C=80 vs ~0.98 for gate/polynomial) — a real, separate phenomenon
  (also seen earlier and unmovable by steps or minimality), which limits how
  cleanly the high-C phase PR can be read.

**Held claim (Task 1), rewritten to what survives:**

> Run with the real APD solver, the country readout is decomposed **faithfully**
> at every order (at moderate budget), and the feature **resolves into a small,
> bounded set of components at every order and geometry** (recon-95 ~2–7; the
> effective component count plateaus far below the budget — it does **not** track
> it). There is **no saturation and no rank inflation** on the decomposition axis.
> A *modest, bounded* dimensionality difference remains (the phase code spans
> somewhat more components than a polynomial of the same degree), and the phase
> code uniquely resists faithful high-budget decomposition. The geometry-specific
> effect with a clean signal is therefore on the **attribution axis** (Task 2:
> first-order attribution is blind to phase codes, recoverable for polynomials of
> the same degree), **not** on the decomposition axis.

The earlier dominant-count / redundancy figure (`figs/spd_budget_sweep.png`) is
retained only as the **confounded metric, shown for completeness**.

## Task 2 — kill the by-construction confound · **the key correction**

Ordinary dense Heads trained from scratch (no decoupling) on a cubic country
target:

| model | country AUC | first-order `<r,L>` | reader align | nonlinear units @AUC.95 | dedication |
|---|---|---|---|---|---|
| `e2`  θ from emb-PCA, `sign(sin3θ)`     | **0.795** (SGD can't fit) | 0.79 | 0.89 | — | 0.43 |
| `e2b` homogeneous cubic `a³−3ab²`       | **0.979** | **0.979 (NOT blind)** | 0.77 | **24** | 0.71 |
| (ref) `order3_pinwheel` `sign(sin3θ)`, constructed | 0.979 | **0.53 (blind)** | 0.13 | 32 | dedicated |

**This is where the attack bites.** Two *genuinely cubic* (degree-3) country codes
with matched accuracy (0.979) give **opposite** first-order behavior:
`sign(sin3θ)` is blind (`<r,L>`=0.53); the homogeneous polynomial `a³−3ab²` is
**not** (`<r,L>`=0.979). So:

- **The cubic spreads across more elementary ReLU units** (24–32 first-order
  *hidden units* to reconstruct vs ~6 for a linear/gated feature), in the
  non-dedicated e2b model too, not just the constructed pinwheel. *Granularity
  note:* this is the **hidden-unit** view; at the **SPD-component** level (Task 1)
  the cubic still reconstructs from ~6 components, so this is not an intrinsic
  SPD-component cost — it is the same "the cubic does not condense into bounded
  stable structure" phenomenon seen as budget saturation in Task 1, not a higher
  reconstruction price.
- **First-order "blindness" is NOT a generic consequence of interaction order.**
  It is a property of the encoding *geometry*: a bounded/periodic phase code has a
  tangential gradient (`x·∇` ⊥ level sets), so first-order attribution misses it;
  a homogeneous polynomial of the same degree satisfies `x·∇f = d·f` (Euler) and
  is fully first-order-recoverable. The original "order-3 ⇒ first-order-blind"
  conflated the phase encoding with interaction order. **Claim rewritten** (below).
- **Bound:** ordinary SGD did not reach a clean `sign(sin3θ)` code at all (0.795).
  The phase regime, where blindness lives, is reachable mainly by construction.

## Task 3 — minimality/simplicity tradeoff · **tension NOT observed → framing dropped**

SPD at Schatten (minimality) coeff ∈ {0.1, 1, 10} on order-2/3 readouts
(`figs/spd_minimality_tradeoff.png`):

| | schatten 0.1 → 10 |
|---|---|
| order-2 country AUC | 0.995 → 0.994 (flat) ; components 9 → 8 |
| order-3 country AUC | 0.87 → 0.89 (flat) ; components 6 → 2 |

Across a 100× range of minimality weight, **faithfulness is essentially flat** and
component count varies modestly (if anything country *compresses* under stronger
minimality with no accuracy loss). We assert no minimality-vs-simplicity tension —
**we drop that framing**, as the brief instructs when no tension appears.

## Task 4 — gap is not a metric artifact · **SURVIVED**

Re-derived the order-2 dissociation with **no gradients and no PCA**: atomic fixed
component = best single hidden unit; reconstruction = top-K causal hidden units.

| order | single best unit | fixed probe | causal recon (6 units) | gap |
|---|---|---|---|---|
| 1 | 0.925 | 1.000 | 0.997 | 0.07 |
| 2 | 0.596 | 0.520 | **0.962** | **0.37** |
| 3 | 0.563 | 0.551 | 0.531 | −0.03 |

The order-2 dissociation (a single fixed component fails; a few causally-selected
components succeed) reproduces with a gradient-free, PCA-free construction
(gap 0.37 vs original 0.44). Three independent instruments now agree.

## Task 5 — blind prediction on new structures · **MIXED (predictions pre-registered in code)**

| structure (target AUC) | prediction | measured | verdict |
|---|---|---|---|
| order-2 XOR `number⊕color` (0.99) | fixed probe ≈ chance; gap 0.3–0.45; few units; conditional restore HIGH | gap **0.32**, units **4**, reader-align 0.08, first-order 0.99 (locally linear), probe **0.87**, within-cell align **0.62** | **partial — weaker than predicted** |
| order-4 `sign(sin4θ)` (**0.565**) | more blind than order-3 | model **failed to train** (AUC 0.565) | **untestable** |

**The order-2 prediction's qualitative shape appeared but weaker than expected.**
The dissociation direction was right (a single fixed component is far worse than a
few recombined ones; the reader rotates; the logit is locally linear), but the
*magnitudes* missed: we predicted the fixed probe ≈ chance and a near-full
conditional restore, and got a probe of 0.87 and only a partial within-cell
restore (0.62) — i.e. this held-out gate is **softer / more linearly-leaky** than
the original `food⊕sentiment` gate. So this is a *predicted dissociation that
showed up attenuated*, not a clean confirmation. The order-4 prediction could not
be tested at all, because a clean order-4 phase code does not emerge from ordinary
training (AUC 0.565) — which separately corroborates the Task-2 bound.

---

## Divergences from the plan (verified against the code)

1. **The `spd` package is real and installed** (`ApolloResearch/apd`, env `apd`,
   py3.11). We wrote a minimal custom target/SPD-twin for the plain 2-layer
   readout, used a TMS task-config stub to hit the generic loss branches, and
   passed our own target/dataloader (the dummy pretrained path is never read).
2. **`model_m2_xor.pt` excluded** (composite gated construction, not a standard Head).
3. **Component counts depend on C, batch-topk=10, Schatten p=0.9 and thresholds —
   and the `dominant = argmax-over-8` metric is *confounded* by readout
   composition.** The clean, budget-robust signals are *recon-95* (~2–7, flat) and
   the *country-only* participation ratio (plateaus far below C). The earlier
   "2× / 38-of-40 / saturation / redundancy" framing was confounded and is
   **withdrawn** (see the budget-sweep subsection).
4. **SGD cannot reach a clean periodic high-order code from scratch** (order-3
   `sin3θ`: 0.795; order-4 `sin4θ`: 0.565). Reported, not hidden — it bounds the
   phase-regime claims.

## Bottom-line claim, rewritten to what survived

- **On the decomposition axis there is no clean geometry effect.** Run with the
  real APD solver, the country readout decomposes **faithfully** at every order and
  the feature **resolves into a small, bounded set of components everywhere**
  (recon-95 ~2–7; the country-only effective-component count plateaus far below the
  budget). **No saturation, no rank inflation.** A *modest, bounded* dimensionality
  difference remains (phase spans somewhat more components than a same-degree
  polynomial), and the phase code uniquely resists faithful high-budget
  decomposition. *(saturation/redundancy claim withdrawn — it was a confounded
  cross-feature metric)*
- **The clean geometry-specific result is on the attribution axis:** first-order
  attribution is blind to *phase/periodic* codes (gradient tangential) but recovers
  a homogeneous polynomial of the same degree. *(survives — the load-bearing claim)*
- **The order-2 gated feature is not carried by one stable component but is
  separable conditional on the features it interacts with** — robust to a
  gradient-free, PCA-free metric and (attenuated) to a new feature pair. *(survives)*
- **None of this is impossibility.** Faithful decomposition exists at every order;
  country reconstructs from a handful of components regardless of geometry.

See `STEELMAN_MEMO.md` for the per-task survive/weaken/strengthen ledger and
`LIMITATIONS.md` for scope.
