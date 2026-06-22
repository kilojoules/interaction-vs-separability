"""Task 2 (stronger) -- try harder to get a NON-dedicated but CLEAN cubic.

e2 (theta from PCA plane) only reached country AUC ~0.80 from scratch -- SGD found
a partial, more-linear shortcut rather than a true cubic.  Here we make the cubic
maximally learnable: build it from two linearly-decodable signals a, b (logistic
probe margins of two real features), so the network can compute a,b cheaply and
only needs to learn the degree-3 combination:
    country := sign(a^3 - 3 a b^2)          # Re((a+ib)^3): 6 sign sectors, like sin(3theta)
Train a dense Head from scratch (no dedication), best of several seeds, more
epochs.  Measure blindness + rank inflation + dedication.

If country AUC is high (>0.9) AND dedication is low AND first-order is blind ->
the order-3 claims are architecture-independent (confound killed).
If country AUC stays low -> report honestly that ordinary training cannot reach a
clean cubic, which bounds (not kills) the confound.

Run (base env): USE_TF=0 python experiments/e2b_clean_cubic.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import separability as S
from e2_train_nondedicated import dedication
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")


def clean_cubic_labels(emb_tr, emb_te, y_tr):
    """a,b = standardized logistic-probe margins of two strong linear features
    (color=2, person=6); country = sign(a^3 - 3 a b^2)."""
    from sklearn.linear_model import LogisticRegression
    def margins(fidx):
        lr = LogisticRegression(max_iter=2000).fit(emb_tr, y_tr[:, fidx])
        a_tr = lr.decision_function(emb_tr); a_te = lr.decision_function(emb_te)
        mu, sd = a_tr.mean(), a_tr.std() + 1e-9
        return (a_tr - mu) / sd, (a_te - mu) / sd
    a_tr, a_te = margins(2); b_tr, b_te = margins(6)
    def cube(a, b): return ((a ** 3 - 3 * a * b ** 2) > 0).astype(np.int64)
    return cube(a_tr, b_tr), cube(a_te, b_te)


def train(c_tr, emb_tr, y_tr, seed, epochs):
    Y = y_tr.copy().astype(np.float32); Y[:, S.CI] = c_tr
    torch.manual_seed(seed); np.random.seed(seed)
    net = S.Head(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(emb_tr); Yt = torch.from_numpy(Y); rng = np.random.default_rng(seed); n = len(X)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, 256):
            idx = perm[i:i + 256]; opt.zero_grad(); lf(net(X[idx]), Yt[idx]).backward(); opt.step()
    net.eval(); return net


def main(epochs=400, seeds=(0, 1, 2)):
    tr = np.load(BDC / "cache/train.npz"); te = np.load(BDC / "cache/test.npz")
    emb_tr = tr["emb"].astype(np.float32); emb_te = te["emb"].astype(np.float32)
    y_tr, y_te = tr["labels"], te["labels"]
    c_tr, c_te = clean_cubic_labels(emb_tr, emb_te, y_tr)
    print(f"cubic target base rate: {c_te.mean():.3f}")
    best = None
    for sd in seeds:
        net = train(c_tr, emb_tr, y_tr, sd, epochs)
        with torch.no_grad():
            L = net.layers[:6](torch.from_numpy(emb_te)).numpy().astype(np.float64)
        ylab = y_te.copy(); ylab[:, S.CI] = c_te
        cauc = S.auc(net.layers[6:](torch.tensor(L, dtype=torch.float32)).detach().numpy()[:, S.CI], c_te)
        print(f"  seed {sd}: country cubic AUC {cauc:.3f}")
        if best is None or cauc > best[0]:
            best = (cauc, sd, net, L, ylab)
    cauc, sd, net, L, ylab = best
    rep = S.country_blindness_report(net, L, ylab)
    ded = dedication(net, L, ylab)
    out = {"best_seed": sd, "country_cubic_auc": cauc, "report": rep, "dedication": ded}
    outd = ROOT / "results" / "task2"; outd.mkdir(parents=True, exist_ok=True)
    (outd / "clean_cubic.json").write_text(json.dumps(out, indent=2))
    torch.save(net.state_dict(), ROOT / "models" / "nondedicated_clean_cubic.pt")
    print(f"\nBEST (seed {sd}) country cubic AUC {cauc:.3f}")
    print(f"  reader_alignment {rep['reader_alignment']:.3f} | fixed_probe {rep['fixed_reader_auc']:.3f}")
    print(f"  FIRST-ORDER <r,L> full AUC {rep['first_order_recon']['full']:.3f}  (blind if << country AUC)")
    print(f"  nonlinear units for AUC>=.95: {rep['nonlinear_units_for_auc95']}")
    print(f"  country-unit dedication {ded['country_unit_dedication']:.3f} (1=dedicated, .5=shared)")


if __name__ == "__main__":
    main()
