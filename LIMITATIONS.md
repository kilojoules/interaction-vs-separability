# Limitations — what this does and does not show

**Toy scale, by design.** A frozen MiniLM encoder + a tiny 8-output MLP. The point is
*controlled* contrast with **known ground truth**, not scale. Nothing here demonstrates
behavior in real LLMs; cross-scale transfer is a hypothesis, not a result.

**The phase result is n=1 and dedicated *by construction*.** The only phase model is the
hand-built `model_pinwheel.pt` (decoupled, country-dedicated block-diagonal phase bank).
SGD never produced a clean from-scratch phase code (order-3 `sin3θ` plateaus at AUC 0.795
and is *not* blind; order-4 `sin4θ` reaches 0.565). So **"phase geometry" co-varies with
"dedicated construction"** and the two cannot be fully separated. The genuinely
confound-free part is the *degree control*: the polynomial `a³−3ab²` and gated/XOR
`a⊕b⊕c` are from-scratch, same degree, and both first-order-recoverable and
faithfully-and-parsimoniously decomposable.

**The blindness "controls" are matched on degree, not on L-nonlinearity.** The polynomial
and XOR are first-order-recoverable largely because the trained readout **linearizes them
in L** (fixed linear probe 0.95 / 0.91 vs phase 0.55). They are genuinely nonlinear in the
raw embedding (0.69 / 0.53), so they are not trivial — but the honest claim is "phase is
blind while degree-3 codes the readout linearizes in L are not," not a clean
geometry-vs-everything separation. The mechanism is ReLU-region offset cancellation
(within a cell `z = <r,L> + c`, the offset carries the signal); Euler's identity is an
analogy for the polynomial, not the operative mechanism.

**The decomposition-axis result is NOT at matched faithfulness, and the component count is
metric-dependent.** Even at its best the phase decomposition has `recon_rel ≈ 0.03` vs the
others' `≤ 0.001`, so its large reconstruction-spread area is *inseparable* from its
poorer reconstructability. The "minimality" recon-curve is run in pure-reconstruction mode
(no minimality objective); the actual-minimality result is the separate observation that
only phase loses faithfulness under top-k+Schatten. The raw component *count* is inflated
by the shared full-rank `mlp_in` (isolating the country projection drops phase 34→13, gate
17→1), so we lead with the area and the under-minimality faithfulness, not a count. Phase
is the outlier on every metric; the fine ordering among non-phase codes is not robust.

**The faithfulness "ceiling" is fragility, not a wall.** Phase reaches AUC ~0.96–0.977
under a careful, parsimony-free recipe, but is seed-unstable (lr=3e-3 seeds 0.85/0.77/0.64)
and lr-fragile. An earlier "unmovable ceiling" claim was wrong and is withdrawn.

**Mostly single-seed; AUC noise is real.** Most numbers are single-seed (areas checked on
2). AUC on 1500 test points has a ~±0.005 Hanley–McNeil SE, so the *large-margin* claims
(blindness 0.45 gap; area 3–25×) are safe but the *faithfulness* gaps (e.g. 0.961 vs 0.979)
are within ~2× the CI and swamped by seed variance — do not lean on them.

**Three withdrawn decomposition-axis framings.** "Rank inflation" (metric-fragile), "budget
saturation" (readout-composition-confounded), and "no effect / recon-95 ~6 everywhere"
(measured on an unfaithful phase decomposition) were each asserted and then withdrawn. See
FINDINGS "What we withdrew" and `AUDIT.md`. The lesson — component-count metrics on a
shared multi-output readout are treacherous — is itself a finding.

**Not impossibility.** Every code is faithfully decomposable; what is phase-specific is the
*cost/tension* and (for first-order attribution) the *blindness*. We never claim no
decomposition exists.
