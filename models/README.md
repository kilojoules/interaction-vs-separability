# Checkpoints

All share the `Head` architecture (384→64→64→64→64→8, ReLU). The analyzed layer
is `L = layers[:6](emb)`; the decomposition target is the readout `layers[6:]`.

| file | order | country encoding | provenance |
|---|---|---|---|
| `order1_vanilla.pt` | 1 | linear (learned normally) | `train_vanilla.py` (Adam, 80 ep, BCE on real labels) |
| `order2_model.pt` | 2 | gated sign-flip on `food⊕sentiment` | the released BlueDot puzzle model (`bluedot-tais-puzzle/model.pt`) |
| `order3_pinwheel.pt` | 3 | phase parity `sign(sin 3θ)`, decoupled/**dedicated** construction | BlueDot pinwheel build (scripts 35h–35j: decoupled channels + warm-started phase-bank head) |
| `nondedicated_order3_sd0.pt` | 3* | cubic target via top-2 PCA θ, **dense from-scratch** training | `../experiments/e2_train_nondedicated.py` |
| `nondedicated_clean_cubic.pt` | 3* | cubic `a³−3ab²` of two linear margins, dense from-scratch | `../experiments/e2b_clean_cubic.py` |
| `order2_xor_number_color.pt` | 2* | `country := number ⊕ color` | `../experiments/e5_blind_prediction.py` |
| `order4_sin4theta.pt` | 4* | `sign(sin 4θ)` | `../experiments/e5_blind_prediction.py` |

`*` = built for the steelman (Tasks 2 and 5).

`order2_model.pt` and `order3_pinwheel.pt` are reproduced by the BlueDot puzzle
repo, not retrained here (the pinwheel build is a multi-stage construction; see
that repo's scripts 35h–35j and the `bluedot-puzzle-status` notes). The order-1
control is fully reproducible via `train_vanilla.py`.
