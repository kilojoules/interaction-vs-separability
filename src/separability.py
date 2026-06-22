"""First-order separability metrics for a readout (ported & cleaned from the
original analysis).  These are the cheap parameter-Jacobian instruments; the
real APD/SPD decomposition lives in spd_readout.py / spd_analyze.py.

Cut: L = net.layers[:6](emb) (64-d); readout = net.layers[6:].
"""
from __future__ import annotations
import json
import numpy as np
import torch
import torch.nn as nn

FEATS = ["number", "question", "color", "food", "sentiment", "country", "person", "body_part"]
CI = 5


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
    def forward(self, x): return self.layers(x)


def load(path):
    net = Head(); net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    net.eval(); return net


def readers(net, L):
    """Per-input reader r_f(x) = d z_f / d L for all 8 outputs.  (8, n, 64)."""
    Lt = torch.tensor(L, dtype=torch.float64, requires_grad=True)
    W1 = net.layers[6].weight.double(); b1 = net.layers[6].bias.double()
    W2 = net.layers[8].weight.double(); b2 = net.layers[8].bias.double()
    R = np.zeros((8, L.shape[0], 64))
    for f in range(8):
        if Lt.grad is not None: Lt.grad.zero_()
        z = (torch.relu(Lt @ W1.T + b1) @ W2.T + b2)[:, f].sum()
        z.backward(retain_graph=True)
        R[f] = Lt.grad.detach().numpy()
    return R


def reader_alignment(R_f):
    """Mean pairwise cosine of per-input unit readers (1=one fixed reader)."""
    Rn = R_f / (np.linalg.norm(R_f, axis=1, keepdims=True) + 1e-12)
    s = Rn.sum(0); n = R_f.shape[0]
    return float((s @ s - n) / (n * (n - 1)))


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels).astype(int)
    order = s.argsort(); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    a = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))


def fixed_reader_auc(L, y, folds=5, seed=0):
    """Best single STATIC linear component: 5-fold CV logistic probe AUC."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed); n = len(y)
    if y.sum() == 0 or y.sum() == n: return float("nan")
    idx = rng.permutation(n); aucs = []
    for k in range(folds):
        te = idx[k::folds]; trm = np.ones(n, bool); trm[te] = False
        lr = LogisticRegression(max_iter=2000).fit(L[trm], y[trm])
        aucs.append(auc(lr.decision_function(L[te]), y[te]))
    return float(np.mean(aucs))


def first_order_recon_auc(R_f, L, y, K_list=(1, 2, 4, 64)):
    """Reconstruction with top-K reader components + true per-input loadings
    (privileged): p=<r_hat,L>.  full = use the exact per-input reader."""
    full = np.einsum("nd,nd->n", R_f, L)
    out = {"full": auc(full, y)}
    m = R_f.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(R_f - m, full_matrices=False)
    for K in K_list:
        V = Vt[:K]; Rhat = m + (R_f - m) @ V.T @ V
        out[f"K{K}"] = auc(np.einsum("nd,nd->n", Rhat, L), y)
    return out


def nonlinear_unit_reconstruction(net, L, y, f=CI, K_list=None):
    """Faithful nonlinear component count: keep top-K hidden units (with ReLU),
    recompute logit, AUC.  Returns {K:auc}, and K to reach AUC>=0.95."""
    with torch.no_grad():
        W1 = net.layers[6].weight.double(); b1 = net.layers[6].bias.double()
        W2 = net.layers[8].weight.detach().numpy(); b2 = float(net.layers[8].bias[f])
        h = torch.relu(torch.tensor(L) @ W1.T + b1).numpy()
    contrib = np.abs(W2[f]) * h.std(0)
    order = np.argsort(contrib)[::-1]
    if K_list is None: K_list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    aucs, k95 = {}, None
    for K in K_list:
        sel = order[:K]; z = h[:, sel] @ W2[f, sel] + b2
        a = auc(z, y); aucs[K] = a
        if k95 is None and a >= 0.95: k95 = K
    return aucs, k95


def country_blindness_report(net, L, y):
    """One-call summary of the order-3 claims for a model's country readout."""
    R = readers(net, L); Rc = R[CI]; yc = y[:, CI]
    fo = first_order_recon_auc(Rc, L, yc)
    units, k95 = nonlinear_unit_reconstruction(net, L, yc)
    with torch.no_grad():
        z = net.layers[6:](torch.tensor(L, dtype=torch.float32)).numpy()[:, CI]
    return {
        "country_model_auc": auc(z, yc),
        "reader_alignment": reader_alignment(Rc),
        "fixed_reader_auc": fixed_reader_auc(L, yc),
        "first_order_recon": fo,
        "nonlinear_unit_auc": units,
        "nonlinear_units_for_auc95": k95,
    }
