"""Task 4 -- confirm the separability gap is not a metric artifact.

The original gap = (reader-PCA rank-2 reconstruction) - (best fixed reader).
Both use model GRADIENTS and PCA.  Here we re-derive the same dissociation with
metrics that use NEITHER:

  separability (atomic, no fit/grad/PCA): best SINGLE hidden unit for country.
  separability (best fixed linear, no model-grad): logistic probe at L.
  reconstruction (causal, no grad/PCA): top-K hidden UNITS (ReLU kept), ablated
      in/out by the model itself -> AUC.  K units recombine per-input via ReLU.

gap_nongrad = unit_recon(small K) - single_best_unit.  If this matches the
original gradient/PCA gap across interaction order, the dissociation is robust.
Run (base env): USE_TF=0 python experiments/e4_nongradient_gap.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import separability as S
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")

MODELS = {
    "order1_vanilla":  BDC / "model_vanilla.pt",
    "order2_model":    BDC / "model.pt",
    "order3_pinwheel": BDC / "model_pinwheel.pt",
}


def hidden_acts(net, L):
    with torch.no_grad():
        W1 = net.layers[6].weight.double(); b1 = net.layers[6].bias.double()
        return torch.relu(torch.tensor(L) @ W1.T + b1).numpy()


def single_best_unit_auc(h, y):
    """Best single hidden unit as a country classifier -- the most atomic fixed
    component: no fitting, no gradient, no PCA."""
    return max(S.auc(h[:, i], y) for i in range(h.shape[1]))


def main():
    te = np.load(BDC / "cache/test.npz"); emb = te["emb"].astype(np.float32); y = te["labels"]
    rows = {}
    for key, ckpt in MODELS.items():
        net = S.load(ckpt)
        with torch.no_grad():
            L = net.layers[:6](torch.from_numpy(emb)).numpy().astype(np.float64)
        yc = y[:, S.CI]
        h = hidden_acts(net, L)
        # non-gradient separability
        sbu = single_best_unit_auc(h, yc)
        probe = S.fixed_reader_auc(L, yc)
        # causal non-gradient reconstruction (top-K units)
        units, k95 = S.nonlinear_unit_reconstruction(net, L, yc)
        # gradient/PCA originals for comparison
        R = S.readers(net, L)[S.CI]
        align = S.reader_alignment(R)
        fo = S.first_order_recon_auc(R, L, yc)
        # linear-feature control (single best unit + probe) on feature "number"
        ln = single_best_unit_auc(h, y[:, 0]); lp = S.fixed_reader_auc(L, y[:, 0])
        rows[key] = {
            "country": {
                "single_best_unit_auc": sbu,            # non-grad/PCA separability (atomic)
                "fixed_probe_auc": probe,               # non-model-grad separability
                "unit_recon_K2": units[2], "unit_recon_K6": units[6], "unit_recon_K64": units[64],
                "units_for_auc95": k95,
                "gap_nongrad_K6": float(units[6] - sbu),
                "reader_pca_recon_full": fo["full"],    # original gradient+PCA reconstruction
                "reader_alignment": align,
            },
            "linear_control_number": {
                "single_best_unit_auc": ln, "fixed_probe_auc": lp,
            },
        }
        c = rows[key]["country"]
        print(f"[{key}] country: single_unit {sbu:.3f} | probe {probe:.3f} | "
              f"unit_recon K2 {units[2]:.3f} K6 {units[6]:.3f} | gap_nongrad(K6) {c['gap_nongrad_K6']:.3f}")
    out = ROOT / "results" / "task4"; out.mkdir(parents=True, exist_ok=True)
    (out / "nongradient_gap.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out/'nongradient_gap.json'}")


if __name__ == "__main__":
    main()
