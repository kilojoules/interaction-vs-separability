# FINDINGS — interaction-vs-separability

Does a feature's **encoding geometry** affect how cleanly a small readout decomposes?
Toy scale, ground truth known. Country is encoded several ways at the analyzed layer
**L**, and we ask what survives honest, adversarial scrutiny. This document states
each claim at the strength a **multi-agent adversarial audit** (`AUDIT.md`) licensed;
the revision history (and what we withdrew) is in the last section, because the
component-count story in particular took four tries to get right.

Encodings (country): order-1 linear; order-2 **gate** `food⊕sentiment` (released
`model.pt`); order-3 **phase** `sign(sin 3θ)` (constructed `model_pinwheel.pt`);
order-3 **polynomial** `a³−3ab²` (from-scratch `e2b`); order-3 **gated/XOR**
`number⊕color⊕person` (from-scratch). Decomposition target = readout `net.layers[6:]`
(`L(64)→64→ReLU→8 logits`). The real solver is Apollo's `spd`/APD `optimize()`.

## Headline (audited)

1. **Attribution-axis blindness — SURVIVES-WITH-CAVEAT.** First-order attribution
   `<r,L> = ∂z/∂L · L` is blind to the **phase** code — its first-order reconstruction
   reaches country AUC only **0.53** (vs the model's 0.98) — but *recovers* the
   polynomial (**0.98**) and gated/XOR (**0.99**) codes of the *same degree*. The ~0.45
   gap in first-order recoverability is robust across seeds and ≫ any sampling CI. Caveat: the
   recoverable controls are ones the trained network **linearizes in L**, and the
   only blind model is **dedicated by construction** — so it is "phase blind while
   degree-3 codes the readout linearizes are not," not a clean geometry-vs-everything
   result.
2. **Faithfulness↔parsimony tension — phase is the unique outlier (decomposition axis).**
   *Two sides.* (a) Under the **real APD minimality objective** (top-k + Schatten),
   only the phase code loses faithfulness: country AUC **0.90→0.68** vs gate/poly/XOR
   **0.98–0.99**. (b) **Without** minimality (pure reconstruction), phase reaches
   faithfulness (~0.96–0.98) only via a **high-rank, many-component interpolation**:
   reconstruction-spread area **0.33** vs gate 0.14, poly 0.05, xor 0.02. Phase is the
   outlier on *every* decomposition metric; the fine ordering among the non-phase
   codes is metric-dependent. This **replaces** an earlier (withdrawn) "saturation"
   claim and an even earlier (withdrawn) "no effect" claim — see "What we withdrew."

Both phase-specific signals share one root: a **smooth periodic** function resists
*both* sparse linear decomposition *and* first-order attribution.

---

## Attribution-axis blindness · SURVIVES-WITH-CAVEAT

`<r,L>` is the per-input directional derivative of the country logit, dotted with the
input — what a first-order (gradient-attribution) method sees. Three genuinely
third-order codes, matched accuracy:

| order-3 code | country AUC | `<r,L>` (blind?) | fixed linear probe @L | probe @raw-emb |
|---|---|---|---|---|
| **phase** `sin3θ` | 0.98 | **0.53 — BLIND** | 0.55 | — |
| polynomial `a³−3ab²` | 0.98 | 0.98 — recovered | **0.95** | 0.69 |
| gated/XOR `a⊕b⊕c` | 0.99 | 0.99 — recovered | **0.91** | 0.53 |

- **The contrast is real but not "geometry vs everything."** The polynomial/XOR are
  first-order-recoverable *because the trained network linearizes them in L* (fixed
  probe 0.95 / 0.91) — they are genuinely nonlinear in the raw embedding (0.69 / 0.53, audit probe) but the readout's input represents them ~linearly. So the codes are matched
  on *degree* but **not** on L-nonlinearity, which is the variable that actually drives
  `<r,L>` recoverability. Honest statement: *phase is blind while degree-3 codes the
  network linearizes in L are not.*
- **Construction caveat.** The only blind model is the hand-built, dedicated
  `model_pinwheel.pt`. SGD never produced a clean from-scratch phase code (the one
  attempt reached AUC 0.795 and was **not** blind). So "phase geometry" co-varies with
  "dedicated construction"; the genuinely confound-free part is the *degree control*
  (poly and XOR are from-scratch, same degree, both recoverable).
- **Mechanism.** Within a ReLU region `z = <r,L> + c`; for phase the offset `c` carries
  essentially all the signal (`std(<r,L>)/std(z) ≈ 6.9`, `corr(<r,L>, z) ≈ 0.035`), so
  a gradient-based reader discards the discriminative part. (Euler's `x·∇f = d·f` for
  homogeneous polynomials is a useful analogy for *why the polynomial is recoverable*,
  but the operative mechanism for blindness is offset cancellation, not Euler.)

## Decomposition-axis: faithfulness↔parsimony tension

The real APD/SPD solver (`optimize()`, A·B components) on the country readout.

**(a) Under the real minimality objective (top-k + Schatten), only phase loses faithfulness.**

| code | C=40 | C=80 | C=120 |  (full-model country AUC under top-k+Schatten) |
|---|---|---|---|---|
| gate | 0.994 | 0.992 | 0.994 | faithful |
| poly | 0.979 | 0.978 | 0.979 | faithful |
| XOR | 0.988 | 0.988 | 0.986 | faithful |
| **phase** | **0.903** | **0.676** | **0.689** | **degrades with budget** |

Phase is the only code that cannot be both **minimal** and **faithful**.

**(b) Without minimality (pure reconstruction), phase reaches faithfulness only via a
many-component, high-rank interpolation.** The recon-curve (country AUC vs fraction of
components kept, greedy by causal effect) crawls for phase and jumps for the others
(`figs/spd_pareto.png`); the area below each code's own ceiling. **All four cells in
each row are from one matched-config run** (C=40, 10k steps, lr=1e-3, pure-recon):

| code | faithfulness | recon_rel | reconstruction-spread **area** | recon-95 (shared / isolated) |
|---|---|---|---|---|
| **phase** `sin3θ` | 0.977 | 0.030 | **0.334** | 34 / **13** |
| gate `food⊕sent` | 0.994 | 0.001 | 0.139 | 17 / 1 |
| poly `a³−3ab²` | 0.979 | 0.000 | 0.045 | 7 / 7 |
| XOR `a⊕b⊕c` | 0.988 | 0.001 | 0.021 | 6 / 4 |

**Caveats (all from the audit, all material):**
- **Not at matched faithfulness.** Even at its best, phase has `recon_rel ≈ 0.03` vs the
  others' `≤ 0.001`; phase's large area is *inseparable* from its poorer
  reconstructability. We report `recon_rel` alongside every area.
- **Pure reconstruction has no minimality objective** (`topk=none, schatten=none`), so
  the recon-curve is a *reconstruction-spread of an unconstrained factorization*, not an
  APD minimality frontier. The actual-minimality result is side (a).
- **The area is robust** (0.32–0.36 across training steps and a second seed); the
  specific value and "phase ≫ all by 5×" are not — and an earlier figure cherry-picked
  the *least*-faithful phase run (fixed).
- **The component count is metric-dependent.** Shared-layer ablation inflates it (the
  64×64 `mlp_in` is 98.5% of params and is shared across outputs); isolating the country
  projection drops phase 34→13, gate 17→1. We lead with the *area* and side (a), not a
  precise count. Directionally **phase is the outlier on all of them.**
- **PR (participation ratio) is unreliable here** and is *not* used — it swings 6–36 for
  phase across step counts and disagrees with recon-95.
- **Graded, single phase model.** Phase ≫ {gate, poly, XOR}; the non-phase ordering is
  metric-dependent (gate is 2nd by area but lowest by isolated count). The phase result
  rests on **n=1** constructed model (see scope).

## Order-2 gated separability · SURVIVES

The order-2 `food⊕sentiment` feature is not carried by one stable component but is
**separable conditional on the features it interacts with**. This reproduces with a
**gradient-free, PCA-free** metric (best single hidden unit 0.60 vs a few causal units
0.96; gap 0.37 vs the original 0.44) and (attenuated) on a held-out gate. (Tasks 4/5.)

## Faithfulness ceiling · SURVIVES-WITH-CAVEAT (reframed: difficulty, not a wall)

An earlier claim called the phase faithfulness ceiling "unmovable." **That is wrong.**
A 21-config stress sweep reaches phase AUC **~0.96–0.977** (pure reconstruction, C=40,
lr=1e-3) — essentially the polynomial's 0.979. The "best 0.977" is an *undertrained*
peak (2k steps); converged is ~0.957–0.961. What persists is a real **phase-specific
difficulty**, not a wall: it needs the parsimony-free recipe, is **seed-unstable**
(lr=3e-3 seeds: 0.85 / 0.76 / 0.64), lr-fragile (1e-3 → 0.96; 1e-2 → 0.65), and even at
its best has `recon_rel` ~10× the others. That difficulty *is* the faithfulness-parsimony
tension above.

---

## What we withdrew (the component-count saga — stated plainly)

The component-count story was wrong three times before it was right; we keep this
explicit because it is the main cautionary lesson:

1. **"2× rank inflation / 38-of-40"** — metric-fragile (mixed two different metrics).
2. **"Budget saturation" (dominant-count tracks the budget)** — **confounded by readout
   composition** (the phase readout is country-heavy *by construction*, so nearly every
   component is nominally country-dominant). The "e2b confirms phase-specificity"
   sub-result was the same artifact in reverse and is withdrawn too.
3. **"No decomposition-axis effect / recon-95 ~2–7 everywhere"** — measured on an
   **unfaithful** phase decomposition (AUC 0.90; recon-95 is normalized to each run's own
   ceiling, so a degenerate low-ceiling solve trivially hits 95% of itself with few
   components).

The surviving, audited result is the faithfulness↔parsimony tension above, measured
against faithfulness directly and under the real minimality objective.

## Methodology & divergences (verified against the code)

- **Real solver:** Apollo `optimize()` (`spd` package, `ApolloResearch/apd`,
  `attribution_type="gradient"`); a minimal custom target/SPD-twin for the plain
  readout + a TMS task-config stub to hit the generic loss branches. Component effects
  read by **causal ablation** (`set_subnet_to_zero`), not gradients. The parameter-Jacobian
  "reader" fallback (`src/separability.py`) is used **only** for the first-order
  (attribution-axis) metrics.
- **`model_m2_xor.pt` excluded** (composite gated construction, not a standard `Head`).
- **Faithfulness is feature-specific.** The 8-output aggregate `recon_rel` (~0.8% for the
  constructed-pinwheel multi-output readout) is *not* the country faithfulness — the
  country logit is ~2% of total output variance, and its single-output `recon_rel` for
  that readout is ~0.40. (Separately, the country-only phase *stress* runs reach
  `recon_rel` 0.03–0.25.) Cite the country-specific number, not the aggregate.
- **SGD cannot reach a clean periodic high-order code from scratch** (order-3 `sin3θ`:
  0.795; order-4 `sin4θ`: 0.565). This bounds all phase-regime claims to the construction.

## Scope (see also LIMITATIONS.md)

Toy scale, ground truth known. The **phase model is n=1 and dedicated by construction**;
geometry-vs-construction cannot be fully separated. Most numbers are single-seed (areas
checked on 2). The decomposition-axis claim is about a 64→64→8 readout, not a real LM.
None of this is impossibility — every code is faithfully decomposable; what is
phase-specific is the *cost/tension*, and (for first-order attribution) the *blindness*.

See `AUDIT.md` for the full adversarial review and `STEELMAN_MEMO.md` for the per-claim ledger.
