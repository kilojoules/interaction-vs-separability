"""Add an order-3 GATED/Boolean encoding: country = a XOR b XOR c (3-way parity).

Fills the design: we had order-3 PHASE (sin3θ) and order-3 POLYNOMIAL (a³−3ab²);
this is order-3 GATED. It is an irreducible third-order interaction (the degree-3
monomial x1·x2·x3 in +/-1 coding), but piecewise-linear (within each of the 8 gate
cells country is linear). Tests degree-vs-geometry: if this order-3 code is NOT
first-order-blind (unlike phase), blindness is a phase-geometry property, not a
consequence of high order.

Predictions committed BEFORE measuring:
  - trains to country AUC > 0.85 (parity of 3 linearly-available features)
  - first-order <r,L> NOT blind (>0.9): gated => locally linear, reader rotates
    between the 8 cells rather than tangentially
  - fixed linear probe ~ chance (parity not linearly separable)
  - hidden units to AUC>=.95: more than order-2 XOR (4), predict ~8-16
  - within-(a,b,c)-cell reader alignment HIGH (conditioning restores separability)

Run (base env): USE_TF=0 python experiments/e6_order3_xor.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import separability as S
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")
A, Bf, Cf = 0, 2, 6   # number, color, person  (three strong linear features)


def train(seed=0, epochs=250):
    tr = np.load(BDC/"cache/train.npz"); te = np.load(BDC/"cache/test.npz")
    emb_tr = tr["emb"].astype(np.float32); emb_te = te["emb"].astype(np.float32)
    y_tr = tr["labels"].copy(); y_te = te["labels"].copy()
    c_tr = (y_tr[:, A] ^ y_tr[:, Bf] ^ y_tr[:, Cf]).astype(np.int64)
    c_te = (y_te[:, A] ^ y_te[:, Bf] ^ y_te[:, Cf]).astype(np.int64)
    y_tr[:, S.CI] = c_tr; y_te[:, S.CI] = c_te
    torch.manual_seed(seed); np.random.seed(seed)
    net = S.Head(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(emb_tr); Y = torch.from_numpy(y_tr).float()
    rng = np.random.default_rng(seed); n = len(X)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, 256):
            idx = perm[i:i+256]; opt.zero_grad(); lf(net(X[idx]), Y[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        L_te = net.layers[:6](torch.from_numpy(emb_te)).numpy().astype(np.float64)
    return net, L_te, y_te, c_te


def within_cell_alignment(net, L, y):
    R = S.readers(net, L)[S.CI]
    cell = 4*y[:, A] + 2*y[:, Bf] + y[:, Cf]    # 8 cells
    al = [S.reader_alignment(R[cell == c]) for c in range(8) if (cell == c).sum() > 25]
    return float(np.mean(al)), len(al)


def main(seed=0):
    net, L, y, c_te = train(seed)
    rep = S.country_blindness_report(net, L, y)
    wca, ncells = within_cell_alignment(net, L, y)
    out = {"encoding": "order3_xor (number^color^person)", "seed": seed,
           "country_auc": rep["country_model_auc"], "report": rep,
           "within_cell_alignment": wca, "n_cells": ncells}
    od = ROOT/"results"/"task6"; od.mkdir(parents=True, exist_ok=True)
    (od/"order3_xor.json").write_text(json.dumps(out, indent=2))
    torch.save(net.state_dict(), ROOT/"models"/"order3_xor.pt")
    print(f"[order-3 XOR  number^color^person]  (base rate {c_te.mean():.2f})")
    print(f"  country model AUC          {rep['country_model_auc']:.3f}")
    print(f"  FIRST-ORDER <r,L> full     {rep['first_order_recon']['full']:.3f}  (blind if << model AUC)")
    print(f"  reader alignment (marginal){rep['reader_alignment']:.3f}")
    print(f"  fixed linear probe AUC     {rep['fixed_reader_auc']:.3f}")
    print(f"  nonlinear units for AUC.95 {rep['nonlinear_units_for_auc95']}")
    print(f"  within-cell alignment      {wca:.3f}  (over {ncells} of 8 gate cells)")


if __name__ == "__main__":
    main()
