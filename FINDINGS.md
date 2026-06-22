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

## Task 1 — run the actual APD/SPD solver · **STRENGTHENED (rank inflation)**

Each readout (`layers[6:]`, `L→64→ReLU→8`) was wrapped as an ApolloResearch `spd`
target + SPD twin (each weight = Σ_C A_cB_c) and decomposed with the real
`optimize()` (param-match + topk-recon + act-recon + Schatten minimality), inputs
= real L activations. `figs/spd_components.png`.

| order | country AUC target→SPD | recon err | components: country / mean-linear | comps dominated by country | cross-seed MMCS |
|---|---|---|---|---|---|
| 1 linear | 1.000→1.000 | 0.2% | 8 / 8  (1.0×) | 13 / 40 | 0.16 |
| 2 gated  | 0.994→0.993 | 0.7% | 8 / 8  (1.0×) | 5 / 40  | 0.21 |
| 3 cubic  | 0.979→0.949 | 0.8% | **6 / 3  (2.0×)** | **38 / 40** | 0.74 |

- The solver **reaches faithfulness on all three readouts** — this is about cost,
  not feasibility. (Closes the "we used a parameter-Jacobian fallback" gap.)
- **Rank inflation is real:** at order-3 country needs 2× the components of a
  linear feature and dominates 38/40 components; at orders 1–2 it decomposes like
  a linear feature. The bottom-row effect heatmaps show country's effect spreading
  across many components only at order-3.
- **MMCS wrinkle (honest):** order-3 components are *more* cross-seed reproducible
  (0.74) than order-1/2 (0.16–0.21). This is **not** "higher order = less stable";
  it reflects (a) order-3's dedicated phase bank and (b) over-decomposition at
  orders 1–2 (C=40 ≫ ~8 mechanisms → interchangeable fragments). We do **not**
  claim a stability degradation.

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

- **Rank inflation is architecture-independent and survives** — it appears in the
  *non-dedicated, from-scratch* clean-cubic model too (24 units vs ~6 for a linear
  feature), not just the constructed pinwheel.
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
| order-2 XOR `number⊕color` (0.99) | gap 0.3–0.45; few units; conditional-restored | gap **0.32**, units **4**, reader-align 0.08, first-order 0.99 (locally linear), within-cell align **0.62** | **mostly right** — dissociation structure confirmed; "fixed probe ≈ chance / full conditional restore" was too strong (probe 0.87, within-cell 0.62 → *softer* gate) |
| order-4 `sign(sin4θ)` (**0.565**) | more blind than order-3 | model **failed to train** (AUC 0.565) | **untestable** — but consistent with Task 2: SGD does not reach high-order phase codes |

The order-2 prediction broadly held (the gated dissociation generalizes to a new
feature pair). The order-4 prediction could not be tested because a clean order-4
phase code does not emerge from ordinary training — itself corroborating the
Task-2 bound.

---

## Divergences from the plan (verified against the code)

1. **The `spd` package is real and installed** (`ApolloResearch/apd`, env `apd`,
   py3.11). We wrote a minimal custom target/SPD-twin for the plain 2-layer
   readout, used a TMS task-config stub to hit the generic loss branches, and
   passed our own target/dataloader (the dummy pretrained path is never read).
2. **`model_m2_xor.pt` excluded** (composite gated construction, not a standard Head).
3. **Component counts depend on C=40, batch-topk=10, Schatten p=0.9 and the
   alive/serves thresholds.** The robust signals are the *within-model*
   country-vs-linear ratio and the *trend across order*, not absolute counts.
4. **SGD cannot reach a clean periodic high-order code from scratch** (order-3
   `sin3θ`: 0.795; order-4 `sin4θ`: 0.565). Reported, not hidden — it bounds the
   phase-regime claims.

## Bottom-line claim, rewritten to what survived

- **Interaction/computational complexity inflates the *cost* of an
  attribution-based parameter decomposition** (more components for the complex
  feature) — robust with the real APD/SPD solver and in a non-dedicated,
  from-scratch model. *(survived, strengthened)*
- **The order-2 gated feature is not carried by one stable component but is
  separable conditional on the features it interacts with** — robust to a
  gradient-free, PCA-free metric and to a new feature pair. *(survived)*
- **First-order attribution is *not* blind to high-order features in general; it
  is blind specifically to *phase/periodic* codes** (gradient tangential), and
  recovers a homogeneous polynomial code of the same degree. *(claim corrected)*
- **None of this is impossibility.** Faithful decomposition exists at every order;
  what rises is the component budget. *(unchanged)*

See `STEELMAN_MEMO.md` for the per-task survive/weaken/strengthen ledger and
`LIMITATIONS.md` for scope.
