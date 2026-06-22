"""Task 5 -- the honest null: predict a NEW interaction structure before measuring.

If the interaction-order ladder is real (not curve-fit to three points), we
should be able to PREDICT the separability of structures we never tuned on.
We commit the predictions below in code, then train and measure two new dense,
from-scratch models:

  A) order-2 XOR gate on a NEW feature pair:  country := number XOR color.
     Predict: marginally non-separable (single unit/probe ~ chance, gap_nongrad
     ~0.3-0.4, few nonlinear units ~<=8), BUT separable conditional on the
     interacting pair (within-(number,color)-cell alignment high).
  B) order-4 cubic-plane code:  country := sign(sin 4*theta).
     Predict: MORE blind than order-3 -- first-order <r,L> AUC ~ chance,
     reader alignment < order-3 (<0.13), nonlinear units for AUC>=.95 >= 32,
     and NOT restored by conditioning on other features (self-interaction).

Run (base env): USE_TF=0 python experiments/e5_blind_prediction.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import separability as S
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")

PREDICTIONS = {
    "order2_xor_number_color": {
        "single_best_unit_auc": "<=0.7 (marginally non-separable)",
        "fixed_probe_auc": "~0.5 (chance)",
        "gap_nongrad_K6": "0.30-0.45",
        "nonlinear_units_for_auc95": "<=8 (few)",
        "within_cell_alignment": "HIGH (>0.8) -> conditional separability restored",
        "first_order_recon_full": ">0.9 (locally linear; reader rotates between 2 gate states)",
    },
    "order4_sin4theta": {
        "first_order_recon_full": "~0.5 (BLIND, <= order-3's 0.53)",
        "reader_alignment": "<0.13 (below order-3)",
        "nonlinear_units_for_auc95": ">=32 (>= order-3)",
        "within_cell_alignment": "stays LOW (self-interaction, not restored by conditioning)",
    },
}


def pca_plane(emb_tr):
    mu = emb_tr.mean(0); X = emb_tr - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return mu, Vt[:2].T


def make_labels(kind, emb_tr, emb_te, y_tr, y_te):
    if kind == "order4_sin4theta":
        mu, P = pca_plane(emb_tr)
        def lab(emb):
            p = (emb - mu) @ P; th = np.arctan2(p[:, 1], p[:, 0])
            return (np.sin(4 * th) > 0).astype(np.int64)
        return lab(emb_tr), lab(emb_te)
    if kind == "order2_xor_number_color":
        # number = feat 0, color = feat 2
        return (y_tr[:, 0] ^ y_tr[:, 2]), (y_te[:, 0] ^ y_te[:, 2])
    raise ValueError(kind)


def train_model(c_tr, c_te, emb_tr, y_tr, seed=0, epochs=200):
    Y = y_tr.copy().astype(np.float32); Y[:, S.CI] = c_tr
    torch.manual_seed(seed); np.random.seed(seed)
    net = S.Head(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(emb_tr); Yt = torch.from_numpy(Y)
    rng = np.random.default_rng(seed); n = len(X)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, 256):
            idx = perm[i:i + 256]; opt.zero_grad(); lf(net(X[idx]), Yt[idx]).backward(); opt.step()
    net.eval(); return net


def within_cell_alignment(net, L, yc, cellfeat_a, cellfeat_b, y):
    R = S.readers(net, L)[S.CI]
    cell = 2 * y[:, cellfeat_a] + y[:, cellfeat_b]
    al = [S.reader_alignment(R[cell == c]) for c in range(4) if (cell == c).sum() > 30]
    return float(np.mean(al))


def main():
    tr = np.load(BDC / "cache/train.npz"); te = np.load(BDC / "cache/test.npz")
    emb_tr = tr["emb"].astype(np.float32); emb_te = te["emb"].astype(np.float32)
    y_tr, y_te = tr["labels"], te["labels"]
    out = ROOT / "results" / "task5"; out.mkdir(parents=True, exist_ok=True)
    results = {"predictions": PREDICTIONS, "measured": {}}

    for kind, (ca, cb) in [("order2_xor_number_color", (0, 2)), ("order4_sin4theta", (3, 4))]:
        c_tr, c_te = make_labels(kind, emb_tr, emb_te, y_tr, y_te)
        net = train_model(c_tr, c_te, emb_tr, y_tr)
        with torch.no_grad():
            L = net.layers[:6](torch.from_numpy(emb_te)).numpy().astype(np.float64)
        yc = y_te[:, S.CI].copy(); yc[:] = c_te
        ylab = y_te.copy(); ylab[:, S.CI] = c_te
        rep = S.country_blindness_report(net, L, ylab)
        # non-grad single best unit + gap
        with torch.no_grad():
            W1 = net.layers[6].weight.double(); b1 = net.layers[6].bias.double()
            h = torch.relu(torch.tensor(L) @ W1.T + b1).numpy()
        sbu = max(S.auc(h[:, i], c_te) for i in range(64))
        gap = float(rep["nonlinear_unit_auc"][6] - sbu)
        wca = within_cell_alignment(net, L, c_te, ca, cb, ylab)
        torch.save(net.state_dict(), ROOT / "models" / f"{kind}.pt")
        m = {
            "country_target_auc": rep["country_model_auc"],
            "single_best_unit_auc": sbu, "fixed_probe_auc": rep["fixed_reader_auc"],
            "gap_nongrad_K6": gap, "nonlinear_units_for_auc95": rep["nonlinear_units_for_auc95"],
            "reader_alignment": rep["reader_alignment"],
            "first_order_recon_full": rep["first_order_recon"]["full"],
            "within_cell_alignment": wca,
        }
        results["measured"][kind] = m
        print(f"\n[{kind}]  (target fit AUC {m['country_target_auc']:.3f})")
        for k, v in m.items():
            if k == "country_target_auc": continue
            print(f"    {k:<28} {v:.3f}" if isinstance(v, (int, float)) else f"    {k:<28} {v}")
    (out / "blind_prediction.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out/'blind_prediction.json'}")


if __name__ == "__main__":
    main()
