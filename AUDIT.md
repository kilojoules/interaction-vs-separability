# AUDIT — interaction-vs-separability

Adversarial review of the toy-scale interpretability study at HEAD `6cf4ec1`. Every
finding below was reproduced against the actual checkpoints, `results/*.json`, and
instrument source — not the prose. Refuted/overstated claims have been dropped or
downgraded.

## 1. Executive summary

The study has **one genuinely robust load-bearing result and one load-bearing result
that is currently misframed**. *Attribution-axis blindness* — first-order
`<r,L>` recovers the polynomial (0.98) and XOR (0.99) order-3 codes but is blind to the
phase code (0.53) — **survives**: the ~0.45 gap is stable across all seeds I re-ran
(poly 0.973–0.979, xor 0.987–0.988, phase 0.534) and is far beyond any sampling CI. It
needs two honesty caveats, not a retraction. The *faithfulness-parsimony Pareto* headline
(phase area 0.363 ≫ gate 0.107 > poly 0.030 > xor 0.014) reproduces exactly and its
qualitative ordering is robust, **but the single most important problem is that it is
NOT measured "at matched faithfulness" as claimed** (M1, C3, S3): the canonical phase run
(`st=20000`, seed 0) has 25% output reconstruction error (`recon_rel=0.254`) and country
AUC only 0.961, while poly/xor/gate reconstruct to `recon_rel ≤ 0.009` at near-target AUC.
The figure script *hardcodes* that worst, non-converged phase run. Compounding this, the
parsimony comparison is run in `pure-reconstruction` mode with **no minimality objective at
all** (`topk=none, schatten=none`), so calling `recon-95` a minimality property is
internally inconsistent (M2). Finally, **every committed doc still tells the superseded
"no decomposition-axis effect, recon-95 ~2-7 everywhere" story** and never mentions the new
Pareto headline (DOC-1, DOC-2) — the author flagged this in the commit message ("Docs NOT
yet updated"). The science is salvageable; the framing and the docs are not yet publishable.

## 2. Findings by severity

### CRITICAL

**[M1 / C3 / S3 — merged] The Pareto headline is NOT "at matched faithfulness."**
- *Claim:* "at matched faithfulness phase needs recon-95 ~33-37 of 40 components" (Key Claim 2).
- *Problem:* The canonical phase point is the *least* faithful available. At the headline
  config (`C40, st20000, lr1e-3, sd0`) phase `recon_rel=0.254` and full SPD country
  AUC=0.961 vs target 0.979, while gate/poly/xor are essentially exact
  (`recon_rel` 0.00026 / 0.00016 / 0.0088; full AUC == target to 3 dp). The curves are each
  self-normalized to their own full endpoint (`e7` L102-106; `e8` L44), so phase's curve is
  anchored to a 25%-unfaithful reconstruction. The comparison is at *badly mismatched*
  faithfulness — the exact confound Key Claim 4 says invalidated an earlier interim claim.
- *Evidence:* `results/stress/{phase,poly,xor,gate}_C40_st20000_lr0.001_sd0_..._co.json`;
  `experiments/e8_pareto_figure.py` `find()` hardcodes `steps==20000` (L29).
- *Fix:* Drop "at matched faithfulness." Report `recon_rel` alongside every Pareto area.
  Either compare only at matched `recon_rel` (which phase cannot meet — its best is 0.0523,
  ~5× xor, ~300× poly/gate), or state explicitly that phase's large area is inseparable from
  its inability to be reconstructed faithfully. Note: re-anchoring to a *faithful* phase run
  (`st5000`, `recon_rel=0.052`) gives area 0.341 and still dominates, so the **ordering
  survives** — only the "matched faithfulness" label and the cherry-picked `0.363` do not.

**[DOC-1 — partly merges C2] The load-bearing decomposition-axis prose contradicts the newest data.**
- *Claim:* "country resolves into a small bounded set (recon-95 ~2-7) at every order and
  geometry; no saturation, no rank inflation" (README L26-30/L69/L125-129; FINDINGS
  L78/L93/L221; STEELMAN L9/L28/L43-46).
- *Problem:* At matched (faithful) config phase needs `recon_k95=37/40` (AUC 0.961) and gate
  `15/40` (0.994), vs poly 6 / xor 3. The docs' phase "~5" comes from the *unfaithful*
  `country_only` run (`recon_k95=5` at AUC **0.903**, target 0.979). `recon_k95` is normalized
  to each run's own full AUC (`spd_analyze.py` L100; `e1f` L93; `e7` L103), so a degenerate
  low-ceiling solve trivially hits 95% of itself with few components. Every committed doc
  still asserts the superseded "no effect" story.
- *Evidence:* `results/stress/phase_..._co.json` `recon_k95=37`; old
  `results/country_only/order3_pinwheel_C40_sd0.json` `recon_k95=5 @ auc 0.903`; commit
  `6cf4ec1` message: "Docs NOT yet updated - pending adversarial audit."
- *Fix:* Rewrite the headline across README/FINDINGS/STEELMAN: the decomposition-axis result
  is now a phase-specific faithfulness-parsimony tension (phase recon-95 ~33-37 *at
  faithfulness* vs poly/xor 3-10), NOT "no effect / ~2-7 everywhere." Stop citing the AUC~0.90
  phase recon-95 as if faithful.

### MAJOR

**[M2] The "minimality" comparison is trained with no minimality objective.**
- *Claim:* `recon-95` / Pareto area is an APD/SPD *parsimony* property.
- *Problem:* The Pareto runs use `topk=none, schatten=none, unit_norm=off` — the loss is
  `param_match + out_recon + act_recon` with zero sparsity/minimality pressure
  (`e7` L73-77; `spd/run_spd.py` skips terms with `None` coeffs). For the **same** phase
  model/config, turning minimality ON (`tk10 sc1.0`) gives `recon_k95=1`; the published
  full-APD `country_only` phase runs give 2-5. So `recon-95` is governed by the
  presence/absence of the minimality term, not the geometry. Calling an unconstrained
  low-rank factorization a "minimality↔faithfulness frontier" (per `e8`) is inconsistent.
- *Evidence:* config keys in `results/stress/*_scnone_tknone_*_co.json`; `e7` L73-77;
  `e8_pareto_figure.py` axis label "parsimony →".
- *Fix:* Either run the Pareto under the real APD objective (topk+Schatten) for all four
  codes, or rename it "reconstruction spread of an unconstrained factorization" and state
  explicitly that pure-reconstruction is not APD. The surviving steelman: phase cannot be
  *both* faithful and low-recon-95, while poly/xor/gate achieve both — keep that.

**[M4 / S2 — merged] The phase headline run is hand-picked, non-converged, and PR is not estimable.**
- *Claim:* phase `pareto_area=0.363`, `PR_effective` and `recon-95` as stable headline numbers.
- *Problem:* Across `st {2000,5000,10000,20000,40000}` at fixed config, phase `recon_rel` =
  0.058/0.052/0.097/**0.254**/0.170 (more training → *worse*) and `PR_effective` =
  34.4/34.8/36.0/11.4/6.0 — a 6× swing. `e8` hardcodes the single worst-`recon_rel` step.
  PR and recon-95 then tell opposite stories (`st40000`: recon-95=37 but PR=6), and the
  PR-direction reverses by config: `country_only` gives phase 10.8 > poly 7.7 > xor 6.7, but
  pure-recon at `st40000` gives phase 6 ≪ poly 22 < xor 27. PR is measuring degeneracy of an
  unconverged solve, not dimensionality.
- *Evidence:* `results/stress/phase_C40_st{...}_lr0.001_sd0_..._co.json`;
  `results/country_only/` vs `e8`. `e8_pareto_figure.py` L29.
- *Fix:* Drop/heavily caveat the PR sub-claim; lead with recon-95-at-matched-faithfulness.
  Do not headline a hand-picked step. Note: `pareto_area` itself stays 0.318–0.363 across the
  sweep, so the *ordering* survives — it is PR and the specific `0.363` that do not.

**[M5] Component count partly measures an arbitrary partition of a shared full-rank layer.**
- *Claim:* "how many components does country need" on the country-only readout.
- *Problem:* Each `mlp_in` component is a full-rank 64×64 matrix (`m=64`) but each `mlp_out`
  component is rank-1 (`m=1`); `param_match` is 98.5% about the shared 64×64 first layer
  (4096 params) vs 1.5% the country 64→1 projection (64 params). Ablating a component drops
  both layers, so recon-95 mixes the arbitrary split of the shared layer into the count.
  Isolating the country projection: phase recon-95 drops 35→11, poly 6→6, xor 3→3 — i.e.
  ~24 of phase's 35 "components" exist only because ablation also destroys a slice of the
  shared layer. The headline overstates intrinsic country dimensionality ~3×.
- *Evidence:* `src/spd_readout.py` `ReadoutSPD/LinearComponent` (`mlp_in` A `(1,40,64,64)`,
  `mlp_out` A `(1,40,64,1)`).
- *Fix:* Constrain `mlp_in` rank (`m` small), or measure component-count on `mlp_out`/act-space.
  At minimum caveat. The directional phase≫poly/xor gap survives isolation (11 vs 6 vs 3), so
  this is a magnitude/validity caveat, not a refutation.

**[C1] The non-blind "controls" are unmatched on what drives first-order recoverability.**
- *Claim:* first-order recovers same-degree poly/xor → blindness is geometry-specific, not degree.
- *Problem:* poly/xor are recoverable because the trained network *linearized* them in `L`: a
  static linear probe on `L` gives 0.946 (cubic) / 0.913 (xor) vs 0.520 (gate) / 0.551 (phase).
  model-minus-fixed-probe gap is +0.033 (cubic) / +0.075 (xor) vs +0.474 (gate) / +0.428 (phase).
  So the controls are matched on *degree* but unmatched on `L`-nonlinearity — the variable
  that actually drives `<r,L>` recoverability. The Task-2 table omits the fixed-probe column,
  even though analogous Task-5 *does* disclose it (0.87).
- *Evidence:* `src/separability.py country_blindness_report`; `results/task2/clean_cubic.json`
  fixed=0.946; `results/task6/order3_xor.json` fixed=0.913; FINDINGS L113-116 omit the column.
- *Fix:* Add the fixed-probe column to the Task-2 table and soften: "phase is blind while
  degree-3 codes the network linearizes in `L` are not." (Targets are genuinely nonlinear in
  the raw embedding — cubic probe 0.69, xor 0.53 — so the codes are not trivial; the
  linearization happens in the readout input.) Blindness contrast survives; "confound-free"
  framing does not.

**[blind-1] Phase-blindness is confounded with dedicated construction.**
- *Claim:* blindness is "clean, confound-free, geometry-specific" (STEELMAN L36).
- *Problem:* The only blind model is the hand-built dedicated block-diagonal `model_pinwheel.pt`.
  The one from-scratch phase attempt reaches AUC 0.795 and is NOT blind (model 0.795 vs
  `<r,L>` 0.794). So "phase geometry" co-varies perfectly with "dedicated construction"; SGD
  never produced a clean from-scratch phase code (0.795 at order-3, 0.565 at order-4).
- *Evidence:* `country_blindness_report`: pinwheel `<r,L>`=0.534; e2b=0.979; xor=0.988; e2
  from-scratch phase model 0.795/`<r,L>`0.794. LIMITATIONS L20-21 already concedes SGD never
  produced the regime; STEELMAN L36 still says "confound-free."
- *Fix:* Soften STEELMAN L36: phase-blindness is observed only in the dedicated construction;
  geometry-vs-construction cannot be fully separated. Keep the **degree control** (poly/xor
  same-degree, from-scratch, both recoverable) as the genuinely confound-free part.

**[DOC-2] The current top-level Pareto result is entirely missing from the docs, and the docs assert its opposite.**
- *Problem:* No doc mentions pareto/parsimony/`spd_pareto`/recon-curve. Worse, STEELMAN L11
  lists the minimality/simplicity tradeoff as "DROPPED (no tension found)" and FINDINGS
  L162-165 says "we assert no minimality-vs-simplicity tension" — the new headline IS that
  tension. `figs/spd_pareto.png` and `e7/e8` are committed but unreferenced.
- *Fix:* Add the Pareto result (figure, area numbers, recon-95-at-matched-faithfulness table)
  as the decomposition-axis finding, replacing the "no tension / no inflation" prose.

**[DOC-3] LIMITATIONS presents withdrawn "budget saturation" as live.**
- *Problem:* LIMITATIONS L14-16 still says "reconstructs from ~5-6 components at every order,
  and the order-3 signature is budget *saturation*, not a higher count." FINDINGS L21/L40/L225,
  README L131, STEELMAN L20-22 all explicitly *withdrew* saturation as a confounded
  cross-feature/readout-composition metric. Git confirms LIMITATIONS was last touched by the
  commit that *introduced* saturation; the withdrawal commit skipped it.
- *Fix:* Delete the sentence; align with the withdrawn-saturation language and the new
  recon-95 ~33-37 phase result.

**[DOC-4] STEELMAN/README/FINDINGS present a disavowed PR residual as settled.**
- *Problem:* "phase PR ~11-17 vs poly ~8" (STEELMAN L9; README ~L128; FINDINGS L80-83) rests
  on a metric the authors' own commit calls "noisy/unreliable → use recon-95," and in the new
  pure-recon regime poly PR=21.7, xor PR=25.2 both *exceed* phase 11.4.
- *Fix:* Drop or sharply qualify; base any dimensionality claim on
  recon-95-at-matched-faithfulness (phase 33-37 vs poly/xor 3-10), not PR.

**[DOC-5] The "faithfulness ceiling" is framed as an unmovable wall; data say optimization/seed fragility.**
- *Problem:* FINDINGS L84-87 calls it "UNMOVABLE by steps or minimality," but the stress sweep
  reaches phase AUC 0.977 (`st2000`) ≈ poly 0.979; the commit itself concludes "Ceiling was an
  optimization/parsimony artifact, NOT a representability wall." Phase is also seed/lr-unstable
  (lr=3e-3 seeds: 0.847/0.765/0.644) — no doc mentions this. A genuine phase-specific
  *difficulty* persists (best phase still recon_rel 5.8% / recon-95 ~34 vs poly recon-95 10),
  but it is the Pareto tension, not a wall.
- *Fix:* Reframe as optimization/seed fragility (reaches ~0.975-0.977 only under narrow config;
  seed-unstable 0.64-0.85 at higher lr); add the seed caveat to LIMITATIONS; do not erase the
  parsimony difficulty.

**[ceiling-3] "Faithful at every order, 0.8% recon err" hides the country ceiling.**
- *Claim:* FINDINGS Task-1 cubic "0.8% recon err / faithful at every order."
- *Problem:* `recon_rel=0.0082` is a full 8-output aggregate where the country logit is only
  2.0% of total output variance (var 28.0 of ~1396). Per-column, the country logit alone has
  `recon_rel=0.40` (98.2% of total reconstruction MSE) — the 0.8% headline is ~50× too
  optimistic for the feature of interest. The country AUC drop 0.979→0.949 is in the adjacent
  column but not propagated.
- *Evidence:* `results/spd/order3_pinwheel_C40_sc1.0_sd0.json`; `src/spd_analyze.py` L63.
- *Fix:* Qualify the "recon err" column as a non-country-dominated aggregate; cite the
  single-output country `recon_rel` (0.05-0.25) as the relevant faithfulness number.

### MINOR

- **[pareto-3]** Order-2 gate (area 0.107) exceeds both order-3 controls (poly 0.030, xor
  0.014). Reframe "phase-specific" as a *graded* effect: phase ≫ gate > poly,xor. (Lives only
  in commit `6cf4ec1`, not yet in prose.)
- **[pareto-4 / S1]** Pareto/ceiling headlines are single-seed at the reported lr. The one
  extra phase seed (sd1) gives area 0.318 / recon-95 33 / AUC 0.974 — ordering robust,
  component-level metrics (PR 11.4→32.2) not. `e8 find()` is seed-blind and returns a
  glob-ordered seed. Report seed mean±spread; separate lr-sensitivity from seed-instability.
- **[blind-2]** The Euler/"tangential gradient" mechanism is a loose heuristic; the rigorous
  mechanism is ReLU-region offset cancellation: within a ReLU cell `z = <r,L> + c`; for phase
  `std(<r,L>)/std(z)=6.9`, `corr(<r,L>,z)=0.035`, and the offset `c` carries all signal
  (the discriminative AUC 0.53 lives entirely in the part `<r,L>` discards). Promote LIMITATIONS
  L43 to the headline; demote Euler to an analogy; note `std-ratio≈1` is the empirical
  recoverable/blind discriminator.
- **[ceiling-1]** The "best phase ~0.975" is an *undertrained peak* (`st2000`); converged phase
  is ~0.957-0.961 and *degrades* with training. Report the converged value or flag early-stopping.
- **[ceiling-2 / ceiling-5]** AUC and `recon_rel` separate phase very differently; report the
  ceiling on the `recon_rel` axis (phase best 0.052 vs poly 0.0002, ~350×) and exclude `multi=1`
  runs from any cross-code `recon_rel` comparison (different denominator — same dilution as
  ceiling-3).
- **[C4]** Order-3 XOR's registered "fixed probe ≈ chance" prediction failed (measured 0.913);
  Task-5 honestly flags the analogous order-2 miss but Task-2 does not. Apply the same honesty;
  XOR-3 is ~91% linearly available in `L`, a weak example of an "irreducible" interaction.
- **[S4]** No CIs/error bars anywhere in the prose. The Hanley–McNeil SE at AUC 0.961
  (n1=746,n0=754) is 0.0051, so the phase-vs-poly *faithfulness* gap (0.018) is only ~2× the CI
  and swamped by seed variance — that secondary claim is not safely beyond noise. (The
  large-margin claims — blindness 0.45 gap, Pareto 3-23× — are not at risk.)
- **[DOC-6]** FINDINGS Task-1 table bolds the withdrawn `6/3 (2.0×)` and `38/40` numbers; the
  "~6 SPD components" baseline (L136) is undercut by matched-faithfulness data (phase 33-37,
  gate 15). Mark inline as withdrawn; reconcile the baseline.

*(Refuted/overstated and dropped from severity ranking: ceiling-4 and C5 — premise factually
wrong, phase DOES have a matched-config seed rerun; blind-3 — best-of-3 selection is on AUC not
`<r,L>`, all seeds confirm the gap; pareto-1 — confirmation, not a defect.)*

## 3. Docs checklist before publishing

The doc-vs-evidence contradictions are the most urgent issue — the repo currently asserts the
negation of its own headline.

- [ ] **README/FINDINGS/STEELMAN:** replace "no decomposition-axis effect / recon-95 ~2-7
      everywhere / no inflation" with the faithfulness-parsimony Pareto headline (phase recon-95
      33-37 *at faithfulness* vs poly/xor 3-10). (DOC-1, DOC-2, C2)
- [ ] **Add the Pareto result + `figs/spd_pareto.png`** and delete the "no minimality tension /
      DROPPED" statements (STEELMAN L11, FINDINGS L162-165). (DOC-2)
- [ ] **LIMITATIONS L14-16:** delete the live "budget saturation / ~5-6 components" sentence.
      (DOC-3)
- [ ] **Drop the "phase PR ~11-17 vs poly ~8" residual** (STEELMAN L9 etc.); the authors'
      commit already disavows PR. (DOC-4)
- [ ] **Reframe the "faithfulness ceiling"** from "unmovable wall" to optimization/seed
      fragility + persistent parsimony difficulty; add the seed-instability caveat. (DOC-5)
- [ ] **State explicitly** that the Pareto comparison is NOT at matched faithfulness (phase
      `recon_rel=0.25` vs others ≤0.009), is in pure-reconstruction (no minimality objective),
      and is single-seed at a hardcoded step. (M1, M2, M4)
- [ ] **Soften "confound-free"** (STEELMAN L36): phase-blindness only in the dedicated
      construction; keep the degree control as the clean part. (blind-1)
- [ ] **Add the fixed-probe column** to the Task-2 table (cubic 0.946, xor 0.913). (C1)
- [ ] **Qualify Task-1 "0.8% recon err"** as a non-country aggregate; cite country-only
      `recon_rel` 0.05-0.25. (ceiling-3); un-bold the withdrawn `2.0×`/`38/40`. (DOC-6)

## 4. Verdict per load-bearing claim

- **Claim 1 — Attribution-axis blindness (phase 0.53 vs poly 0.98 / xor 0.99):**
  **SURVIVES-WITH-CAVEAT.** Gap robust across seeds and ≫ any CI; but drop "confound-free"
  (blind-1), add the fixed-probe disclosure (C1), and fix the mechanism prose (blind-2).
- **Claim 2 — Faithfulness-parsimony Pareto (phase area 0.363 ≫ gate ≫ poly,xor):**
  **NEEDS-REVISION.** Qualitative ordering survives (robust to seed/step/normalization), but
  the headline as stated is false: not "matched faithfulness," not under a minimality objective,
  and the canonical run is a hardcoded, cherry-picked, 25%-unfaithful single seed. Re-measure
  honestly and reframe before publishing. (M1, M2, M4, M5)
- **Claim 3 — Phase faithfulness ceiling (~0.975, unique):** **SURVIVES-WITH-CAVEAT.** A real
  phase-specific difficulty exists at matched config (converged ~0.957-0.961, recon_rel ≫ others),
  but "unmovable wall" and "~0.975" are wrong — report converged values and the seed/lr fragility.
  (ceiling-1, ceiling-2, DOC-5)
- **"Bounded scope / no decomposition-axis effect, recon-95 ~2-7 everywhere":**
  **WITHDRAW.** Contradicted by the newest committed data and by the authors' own commit; the
  small phase recon-95 was measured on an unfaithful (AUC 0.90) decomposition. (DOC-1, C2)
