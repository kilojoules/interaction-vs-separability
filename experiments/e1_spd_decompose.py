"""Task 1 -- run the real APD/SPD algorithm on a readout and analyze it.

Usage:
  python experiments/e1_spd_decompose.py MODEL_KEY CKPT [--C 40 --topk 10 \
      --schatten 1.0 --seed 0 --steps 8000]

Writes results/spd/<MODEL_KEY>_C<C>_sc<schatten>_sd<seed>.json and the SPD
state_dict alongside it.  Run with the apd env python.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import spd_readout as R
import spd_analyze as A
from spd.run_spd import Config, TMSTaskConfig, optimize
from spd.utils import set_seed

BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")


def run_one(model_key, ckpt, C=40, m=None, topk=10.0, schatten=1.0, seed=0, steps=8000,
            lr=1e-3, batch_size=256, out_root=ROOT / "results" / "spd", save_model=True):
    set_seed(seed)
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    tag = f"{model_key}_C{C}_sc{schatten}_sd{seed}"

    tr = np.load(BDC / "cache/train.npz"); te = np.load(BDC / "cache/test.npz")
    L_tr, _ = R.compute_L(ckpt, tr["emb"]); L_te, logits_te = R.compute_L(ckpt, te["emb"])
    y_te = te["labels"]

    target = R.load_readout_target(ckpt)
    # sanity: target readout on L must reproduce the head's logits
    with torch.no_grad():
        chk = target(torch.tensor(L_te, dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()
    assert np.abs(chk - logits_te).max() < 1e-3, f"target mismatch {np.abs(chk-logits_te).max()}"

    spd = R.make_spd_from_target(target, C=C, m=m)
    X_tr = torch.tensor(L_tr, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(X_tr, torch.zeros(len(X_tr))), batch_size=batch_size, shuffle=True)

    tc = TMSTaskConfig(feature_probability=0.05, train_bias=False, bias_val=0.0,
                       pretrained_model_path="dummy")
    cfg = Config(batch_size=batch_size, steps=steps, print_freq=max(steps // 4, 1),
                 image_freq=None, save_freq=None, lr=lr, C=C, m=m, topk=topk, batch_topk=True,
                 unit_norm_matrices=True, param_match_coeff=1.0, topk_recon_coeff=1.0,
                 act_recon_coeff=1.0, schatten_coeff=schatten, schatten_pnorm=0.9,
                 attribution_type="gradient", task_config=tc)
    optimize(model=spd, config=cfg, device="cpu", dataloader=dl, target_model=target,
             param_names=["mlp_in", "mlp_out"], out_dir=None, plot_results_fn=None)

    # ----- analysis (all on held-out test L) -----
    X_te = torch.tensor(L_te, dtype=torch.float32).unsqueeze(1)
    faith = A.faithfulness(spd, target, X_te, y_te)
    eff = A.ablation_effect_matrix(spd, X_te)
    norms = A.component_norms(spd)
    summ = A.summarize_components(eff, norms)
    result = {
        "model_key": model_key, "ckpt": str(ckpt), "config": {
            "C": C, "m": m, "topk": topk, "schatten": schatten, "seed": seed, "steps": steps},
        "faithfulness": faith, "summary": summ,
    }
    (out_root / f"{tag}.json").write_text(json.dumps(result, indent=2))
    if save_model:
        torch.save(spd.state_dict(), out_root / f"{tag}.pth")
    cdom = summ["n_country_dominant_components"]
    print(f"[{tag}] faithful country AUC tgt {faith['per_feature_auc']['country']['target']:.3f} "
          f"spd {faith['per_feature_auc']['country']['spd_full']:.3f} | recon_rel {faith['recon_rel']:.3f}")
    print(f"  C_alive {summ['C_alive']}/{C} | n_country_components {summ['n_country_components']} "
          f"(dominant {cdom}) | mono {summ['mean_monosemanticity_alive']:.3f}")
    print(f"  components/feature: {summ['n_components_per_feature']}")
    return result


REGISTRY = {
    "order1_vanilla":  BDC / "model_vanilla.pt",
    "order2_model":    BDC / "model.pt",
    "order3_pinwheel": BDC / "model_pinwheel.pt",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key")
    ap.add_argument("ckpt", nargs="?", default=None)
    ap.add_argument("--C", type=int, default=40); ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--topk", type=float, default=10.0); ap.add_argument("--schatten", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default=str(ROOT / "results" / "spd"))
    a = ap.parse_args()
    ckpt = a.ckpt or REGISTRY[a.model_key]
    run_one(a.model_key, ckpt, C=a.C, m=a.m, topk=a.topk, schatten=a.schatten, seed=a.seed,
            steps=a.steps, out_root=a.out)
