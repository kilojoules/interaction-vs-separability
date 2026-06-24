# Steelman memo — what survived (post-audit)

One line per claim, at the strength a multi-agent adversarial audit (`AUDIT.md`)
licensed. This is the memo the proposal is written from.

| claim | verdict | one-line |
|---|---|---|
| **Attribution-axis blindness** (phase `<r,L>` 0.53 vs poly 0.98 / xor 0.99) | **SURVIVES-WITH-CAVEAT** | Gap robust across seeds, ≫ any CI. But the recoverable controls are ones the readout *linearizes in L* (fixed probe 0.95/0.91), and the only blind model is dedicated-by-construction — so it's "phase blind while linearized degree-3 codes are not," not confound-free. |
| **Faithfulness↔parsimony tension** (phase is the decomposition-axis outlier) | **HOLDS, REVISED** | Two-sided and confound-free *directionally*: under the real APD minimality objective only phase loses faithfulness (0.68–0.90 vs 0.98–0.99); without minimality phase reaches faithfulness only via a high-rank interpolation (area 0.33 vs ≤0.14). NOT at matched faithfulness; count is metric-dependent; single phase model. |
| **Phase faithfulness ceiling** | **SURVIVES-WITH-CAVEAT** | A real phase-specific *difficulty*, not a "wall": reaches ~0.96–0.977 with a careful, parsimony-free, seed-fragile recipe (lr=3e-3 seeds 0.85/0.76/0.64). Earlier "unmovable" was wrong. |
| **Order-2 gated = separable only conditional on interacting features** | **SURVIVES** | Reproduced gradient-free / PCA-free (gap 0.37); attenuated on a held-out gate. |
| ~~Rank inflation (2×/38-of-40)~~ → ~~Budget saturation~~ → ~~"no effect, recon-95 ~6"~~ | **ALL WITHDRAWN** | Metric-fragile → readout-composition-confounded → measured on unfaithful phase. Replaced by the faithfulness-parsimony tension. |

## What the audit changed

- **The decomposition-axis headline was misframed three times; the audit forced the
  fourth.** Component-*count* metrics on a shared multi-output readout are treacherous
  (readout-composition confound; shared-full-rank-layer inflation; normalization to an
  unfaithful ceiling; PR instability). The audit's load-bearing catch: the Pareto headline
  was **not** "at matched faithfulness" and the figure pinned the *least*-faithful phase
  run. Fixed: lead with (i) under-minimality faithfulness and (ii) the area, report
  `recon_rel` and the isolated count, never a single hand-picked number.
- **Blindness survives but "confound-free" did not.** Disclosed the fixed-probe column and
  the linearization/construction confounds; demoted Euler to an analogy.
- **The ceiling is fragility, not a wall.**

## Final bounded claim (proposal seed)

> A feature's encoding *geometry* affects how cleanly its readout decomposes. A **smooth
> periodic (phase) code** is the unique outlier on two axes: (1) it is **invisible to
> first-order attribution** (`<r,L>` 0.53 vs 0.98–0.99 for polynomial/gated codes the
> readout linearizes in L), and (2) it exhibits a **faithfulness↔parsimony tension** —
> under the real APD minimality objective it alone loses reconstruction faithfulness
> (0.68–0.90 vs 0.98–0.99), and without minimality it reaches faithfulness only through a
> high-rank, many-component interpolation (reconstruction-spread area ~0.33 vs ≤0.14). Both
> stem from a smooth nonlinearity resisting *sparse linear decomposition* and *first-order
> gradients* alike. This is a statement about cost/visibility, **not** impossibility, and it
> is demonstrated on a single, construction-dependent phase model at toy scale.

## What would move this from toy to real

- **The clean, scalable screen is attribution-axis blindness** (no budget sweep, no
  confounds survived) — does first-order attribution recover the feature?
- **Do NOT use multi-output component-count / "saturation" metrics** — confounded twice. If
  a decomposition-axis cost is wanted, use (i) faithfulness *under the real minimality
  objective*, and (ii) the reconstruction-spread area with `recon_rel` reported — never a
  raw count.
- **Prerequisite: a non-constructed phase code.** SGD never produced one (0.795 at order-3,
  0.565 at order-4); geometry-vs-construction can't be separated until one exists.
