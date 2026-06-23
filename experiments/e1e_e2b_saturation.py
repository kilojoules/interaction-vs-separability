"""Decisive test: does the NON-dedicated HOMOGENEOUS-cubic model (e2b) saturate
the SPD budget like the phase pinwheel, or resolve like the gate?

This disentangles "saturation is phase-geometry-specific" from "saturation is
order-3/cubic in general" and from the by-construction (country-heavy readout)
confound -- e2b is a balanced dense head trained from scratch.

Confound note: the dominant-count *fraction* is partly set by country's share of
the readout (~1/8 for a balanced head, ~all for the constructed pinwheel). The
confound-resistant signal is the REDUNDANCY ratio dominant/recon-95 (country-
dominant components beyond the ~6 genuinely needed): pinwheel ~13x, gate ~2x.

Run (apd env): .../envs/apd/bin/python experiments/e1e_e2b_saturation.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import spd_readout as R, spd_analyze as A
from spd.run_spd import Config, TMSTaskConfig, optimize
from spd.utils import set_seed
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")
CKPT = ROOT / "models" / "nondedicated_clean_cubic.pt"


def run(C, seed=0, steps=10000):
    set_seed(seed)
    tr = np.load(BDC / "cache/train.npz"); te = np.load(BDC / "cache/test.npz")
    L_tr, _ = R.compute_L(CKPT, tr["emb"]); L_te, _ = R.compute_L(CKPT, te["emb"])
    c_te = np.load(ROOT / "results" / "task2" / "cubic_labels.npz")["c_te"]  # precomputed (base env)
    y_te = te["labels"].copy(); y_te[:, A.CI] = c_te          # country col = cubic target
    target = R.load_readout_target(CKPT)
    spd = R.make_spd_from_target(target, C=C, m=None)
    X = torch.tensor(L_tr, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(X, torch.zeros(len(X))), batch_size=256, shuffle=True)
    tc = TMSTaskConfig(feature_probability=0.05, train_bias=False, bias_val=0.0, pretrained_model_path="d")
    cfg = Config(batch_size=256, steps=steps, print_freq=steps, image_freq=None, save_freq=None,
                 lr=1e-3, C=C, m=None, topk=10.0, batch_topk=True, unit_norm_matrices=True,
                 param_match_coeff=1.0, topk_recon_coeff=1.0, act_recon_coeff=1.0,
                 schatten_coeff=1.0, schatten_pnorm=0.9, attribution_type="gradient", task_config=tc)
    optimize(model=spd, config=cfg, device="cpu", dataloader=dl, target_model=target,
             param_names=["mlp_in", "mlp_out"], out_dir=None, plot_results_fn=None)
    X_te = torch.tensor(L_te, dtype=torch.float32).unsqueeze(1)
    rep = A.budget_report(spd, target, X_te, y_te)
    rep["redundancy"] = rep["dominant_country"] / max(rep["recon_k95"], 1)
    rep["dominant_frac"] = rep["dominant_country"] / C
    out = ROOT / "results" / "budget"; out.mkdir(parents=True, exist_ok=True)
    (out / f"e2b_cubic_C{C}_sd{seed}.json").write_text(json.dumps(rep, indent=2))
    return rep


def main():
    print(f"{'model':<22}{'C':>4}{'cAUC':>7}{'rel':>7}{'recon95':>8}{'dom':>6}{'dom/C':>7}{'redund(dom/r95)':>16}")
    def ref(key):  # load existing pinwheel/gate budget points
        for C in (40, 80, 120):
            p = ROOT / "results" / "budget" / f"{key}_C{C}_sd0.json"
            if p.exists():
                d = json.loads(p.read_text())
                r = d["dominant_country"] / max(d["recon_k95"], 1)
                print(f"{key:<22}{C:>4}{d['country_auc_spd']:>7.2f}{d['recon_rel']:>7.3f}"
                      f"{d['recon_k95']:>8}{d['dominant_country']:>6}{d['dominant_country']/C:>7.2f}{r:>16.1f}")
    ref("order2_model"); ref("order3_pinwheel")
    print("  --- e2b: non-dedicated homogeneous cubic a^3-3ab^2 ---")
    for C in (40, 80, 120):
        d = run(C)
        r = d["redundancy"]
        print(f"{'e2b_cubic':<22}{C:>4}{d['country_auc_spd']:>7.2f}{d['recon_rel']:>7.3f}"
              f"{d['recon_k95']:>8}{d['dominant_country']:>6}{d['dominant_frac']:>7.2f}{r:>16.1f}")


if __name__ == "__main__":
    main()
