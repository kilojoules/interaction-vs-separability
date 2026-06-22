"""Analysis of a trained SPD decomposition of a readout.

All measures are computed on the FULL trained SPD model (every component present)
and on causal ablations of single components -- no gradients, no PCA. The central
object is the (C x 8) causal effect matrix: how much zeroing component c changes
each output logit. From it we read off, for each feature, how many components it
needs and how monosemantic they are -- the APD-native notion of separability.
"""
from __future__ import annotations
import numpy as np
import torch

FEATS = ["number", "question", "color", "food", "sentiment", "country", "person", "body_part"]
CI = 5


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels).astype(int)
    order = s.argsort(); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    a = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))


def component_param_vectors(spd):
    """(C, D): each component flattened across both layers' A@B weight."""
    cin = spd.mlp_in.component_weights.detach()      # (n_inst, C, d_in, d_out)
    cout = spd.mlp_out.component_weights.detach()
    cin = cin[0] if cin.ndim == 4 else cin           # (C, ...)
    cout = cout[0] if cout.ndim == 4 else cout
    C = cin.shape[0]
    return torch.cat([cin.reshape(C, -1), cout.reshape(C, -1)], dim=1).numpy()


def component_norms(spd):
    v = component_param_vectors(spd)
    return np.linalg.norm(v, axis=1)


def ablation_effect_matrix(spd, X, batch=2000):
    """(C, 8) RMS change in each output logit when zeroing each component.
    Pure causal intervention via the SPDModel's own set_subnet_to_zero."""
    spd.eval()
    with torch.no_grad():
        full = spd(X[:batch]).squeeze(1).numpy()      # (n,8)
    C = spd.C
    eff = np.zeros((C, 8))
    for c in range(C):
        stored = spd.set_subnet_to_zero(c, has_instance_dim=True)
        with torch.no_grad():
            abl = spd(X[:batch]).squeeze(1).numpy()
        spd.restore_subnet(c, stored, has_instance_dim=True)
        eff[c] = np.sqrt(((full - abl) ** 2).mean(axis=0))
    return eff


def faithfulness(spd, target, X, y):
    spd.eval(); target.eval()
    with torch.no_grad():
        t = target(X).squeeze(1).numpy(); s = spd(X).squeeze(1).numpy()
    mse = float(((t - s) ** 2).mean())
    rel = float(((t - s) ** 2).mean() / (t.var() + 1e-9))
    per = {FEATS[g]: {"target": auc(t[:, g], y[:, g]), "spd_full": auc(s[:, g], y[:, g])}
           for g in range(8)}
    return {"recon_mse": mse, "recon_rel": rel, "per_feature_auc": per}


def mmcs(va, va2):
    """Mean max cosine similarity between two component sets (both directions)."""
    A = va / (np.linalg.norm(va, axis=1, keepdims=True) + 1e-12)
    B = va2 / (np.linalg.norm(va2, axis=1, keepdims=True) + 1e-12)
    S = np.abs(A @ B.T)
    return float(0.5 * (S.max(axis=1).mean() + S.max(axis=0).mean()))


def summarize_components(eff, norms, alive_frac=0.02, share=0.1):
    """From the (C,8) effect matrix + norms, derive separability descriptors.

    alive: component norm above alive_frac of the max norm.
    A component 'serves' feature g if its effect on g exceeds `share` of that
    feature's largest single-component effect (i.e. it is a non-trivial
    contributor to g). n_components[g] = number of alive components serving g.
    monosemanticity[c] = top-feature effect share of component c's total effect.
    """
    C = eff.shape[0]
    alive = norms > alive_frac * norms.max()
    feat_max = eff.max(axis=0) + 1e-12                       # per-feature largest effect
    serves = (eff > share * feat_max[None, :]) & alive[:, None]
    n_components = {FEATS[g]: int(serves[:, g].sum()) for g in range(8)}
    mono = []
    for c in range(C):
        if not alive[c] or eff[c].sum() < 1e-9:
            continue
        mono.append(float(eff[c].max() / eff[c].sum()))
    # components whose dominant output is country
    dom = eff.argmax(axis=1)
    country_dom = int(((dom == CI) & alive).sum())
    return {
        "C_total": C, "C_alive": int(alive.sum()),
        "n_components_per_feature": n_components,
        "n_country_components": n_components["country"],
        "n_country_dominant_components": country_dom,
        "mean_monosemanticity_alive": float(np.mean(mono)) if mono else float("nan"),
        "effect_matrix": eff.tolist(),
        "component_norms": norms.tolist(),
    }
