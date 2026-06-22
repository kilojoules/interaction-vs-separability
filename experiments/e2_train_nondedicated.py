"""Task 2 -- kill the by-construction confound.

The original order-3 model encodes country with a decoupled, country-DEDICATED
phase bank (block-diagonal construction).  Here we train an ORDINARY dense Head
from scratch (random init, Adam, no architectural isolation) on a cubic country
target, then re-measure the two architecture-independent claims:
  (1) first-order reconstruction blindness  (<reader,L> AUC ~ chance)
  (2) rank inflation (nonlinear component count vs a linear feature)
and additionally check whether the learned country units are DEDICATED or
entangled with other features (the thing the construction handed us for free).

country target = sign(sin 3*theta), theta = angle of emb in its top-2 PCA plane.
The 7 other features keep their real labels.  If SGD fits this with a dense head
and blindness+inflation persist, the claims are architecture-independent.

Run (base env):  USE_TF=0 python experiments/e2_train_nondedicated.py --seed 0
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import separability as S
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")


def cubic_country_labels(emb_tr, emb_te):
    """theta from top-2 PCA directions of train emb; country = sign(sin 3 theta)."""
    mu = emb_tr.mean(0); X = emb_tr - mu
    U, Svals, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:2].T                                  # (384,2) top-2 PCA dirs
    def lab(emb):
        p = (emb - mu) @ P
        theta = np.arctan2(p[:, 1], p[:, 0])
        return (np.sin(3 * theta) > 0).astype(np.int64)
    return lab(emb_tr), lab(emb_te), {"mu": mu.tolist(), "P": P.tolist()}


def train(seed=0, epochs=200, lr=1e-3, wd=0.0):
    tr = np.load(BDC / "cache/train.npz"); te = np.load(BDC / "cache/test.npz")
    emb_tr = tr["emb"].astype(np.float32); emb_te = te["emb"].astype(np.float32)
    y_tr = tr["labels"].copy(); y_te = te["labels"].copy()
    c_tr, c_te, meta = cubic_country_labels(emb_tr, emb_te)
    y_tr[:, S.CI] = c_tr; y_te[:, S.CI] = c_te      # replace country with cubic target

    torch.manual_seed(seed); np.random.seed(seed)
    net = S.Head()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(emb_tr); Y = torch.from_numpy(y_tr).float()
    rng = np.random.default_rng(seed); n = len(X)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, 256):
            idx = perm[i:i + 256]; opt.zero_grad(); lf(net(X[idx]), Y[idx]).backward(); opt.step()
    net.eval()

    with torch.no_grad():
        logits_te = net(torch.from_numpy(emb_te)).numpy()
        L_te = net.layers[:6](torch.from_numpy(emb_te)).numpy().astype(np.float64)
    aucs = {S.FEATS[g]: S.auc(logits_te[:, g], y_te[:, g]) for g in range(8)}
    return net, L_te, y_te, aucs, meta


def dedication(net, L, y, n_units=12):
    """Are the top country units also used by other features? Compare their
    country weight to their mean other-feature weight (1 = fully dedicated)."""
    with torch.no_grad():
        W1 = net.layers[6].weight.double(); b1 = net.layers[6].bias.double()
        W2 = net.layers[8].weight.detach().numpy()
        h = torch.relu(torch.tensor(L) @ W1.T + b1).numpy()
    contrib = np.abs(W2[S.CI]) * h.std(0)
    top = np.argsort(contrib)[::-1][:n_units]
    cw = np.abs(W2[S.CI, top])                              # country weight of those units
    ow = np.abs(np.delete(W2, S.CI, axis=0)[:, top]).mean(0)  # mean other-feature weight
    ded = float((cw / (cw + ow + 1e-9)).mean())            # ->1 dedicated, ->0.5 shared
    return {"country_unit_dedication": ded, "top_country_units": top.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200); a = ap.parse_args()
    net, L_te, y_te, aucs, meta = train(seed=a.seed, epochs=a.epochs)
    print(f"[nondedicated seed {a.seed}] per-feature AUC:")
    for f, v in aucs.items(): print(f"    {f:<11} {v:.3f}")
    rep = S.country_blindness_report(net, L_te, y_te)
    ded = dedication(net, L_te, y_te)
    rep.update(ded)
    outd = ROOT / "results" / "task2"; outd.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ROOT / "models" / f"nondedicated_order3_sd{a.seed}.pt")
    (outd / f"seed{a.seed}.json").write_text(json.dumps(
        {"per_feature_auc": aucs, "report": rep, "pca_meta_shape": "P=(384,2)"}, indent=2))
    print(f"  country model AUC {rep['country_model_auc']:.3f} | reader_align {rep['reader_alignment']:.3f}")
    print(f"  FIRST-ORDER blindness: <r,L> full AUC {rep['first_order_recon']['full']:.3f} (chance~0.5)")
    print(f"  fixed-reader AUC {rep['fixed_reader_auc']:.3f}")
    print(f"  nonlinear units for AUC>=.95: {rep['nonlinear_units_for_auc95']}")
    print(f"  country-unit dedication {ded['country_unit_dedication']:.3f} (1=dedicated, .5=shared)")
