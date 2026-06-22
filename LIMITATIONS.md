# Limitations — what this does and does not show

**Toy scale, by design.** Everything here is a frozen MiniLM encoder + a tiny MLP
on 8 binary features. The point is *controlled* contrast with **known ground
truth**, not scale. Nothing here demonstrates behavior in real LLMs.

**The order-3 model is dedicated *by construction*.** `order3_pinwheel.pt` was
built with a decoupled, country-dedicated phase bank (block-diagonal). So its
*component dedication* and its high cross-seed MMCS are partly construction
artifacts, not emergent properties. Task 2 partially de-confounds this: a
non-dedicated, from-scratch homogeneous-cubic model (`e2b`, AUC 0.979) reproduces
**rank inflation** (24 nonlinear units vs ~6 for a linear feature), so inflation
is architecture-independent. But **first-order blindness does NOT transfer**: the
homogeneous cubic is fully first-order-recoverable (`<r,L>`=0.979), while the
constructed `sin3θ` phase code is blind (0.53). Blindness is therefore a property
of the *phase/periodic encoding geometry*, not of interaction order or of any
particular architecture — and the phase regime did not emerge from ordinary SGD
at all (order-3 `sin3θ` plateaued at 0.795, order-4 at 0.565). See FINDINGS
Task 2 for numbers.

**"Reconstruction" of the first-order reader uses privileged information.** The
rank-K reader-subspace reconstruction is computed *with* the per-input model
gradient, so it measures reconstruction fidelity of the reader cloud, not a
deployable probe. The causal hidden-unit reconstruction (Task 4) and the real SPD
solver (Task 1) do not have this caveat and corroborate the same dissociation.

**Single checkpoints per order (plus seed-1 for SPD stability).** The order-2/3
target models are individual checkpoints, not ensembles; subsample bootstraps and
(for SPD) seed-1 reruns stand in for retrain variation.

**The claim is about a method family, not impossibility.** We show that a
*first-order / linear* parameter decomposition cannot carry country in one stable
component, and that the real APD/SPD solver needs more components at higher order.
We do **not** claim no faithful decomposition exists: a nonlinear, higher-rank
decomposition reconstructs country at every order. "Non-separable" here always
means "not parsimoniously separable by first-order attribution at these budgets,"
never "no decomposition exists."

**Generalization to trained (non-toy) models is proposed, not shown.** We give a
mechanism (first-order attribution is blind to signal stored in ReLU-region
offsets; interaction inflates component count) that *should* transfer, but we have
not tested it outside this toy. Treat the cross-scale claim as a hypothesis.

**SPD hyperparameters are not exhaustively tuned.** We use the resid-MLP recipe
(batch-topk, Schatten minimality, param-match + topk-recon + act-recon) with a
modest sweep of the minimality coefficient (Task 3). Component counts depend on
the alive-threshold and the topk budget; we report the thresholds and show the
sweep so the trend (not a single number) carries the claim.
