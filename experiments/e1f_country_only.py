"""Confound-free check: decompose the COUNTRY-ONLY readout (64 -> 64 -> 1).

With a single output there is no cross-feature "dominant" metric and no readout-
composition confound. We ask directly: as the component budget C grows, does the
country computation spread across the budget (saturation) or concentrate into a
bounded set (resolves)?  Threshold-free metric: participation ratio of the
per-component causal country effect, PR = (sum e)^2 / sum e^2  (effective number
of components carrying country).  Also recon-95 (components to reconstruct the
country logit to 95% of full AUC) and faithfulness.

Run (apd env): .../envs/apd/bin/python experiments/e1f_country_only.py MODEL_KEY C
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import spd_readout as R, spd_analyze as A
from spd.run_spd import Config, TMSTaskConfig, optimize
from spd.utils import set_seed
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")
REG = {"order2_model": BDC/"model.pt", "order3_pinwheel": BDC/"model_pinwheel.pt",
       "e2b_cubic": ROOT/"models"/"nondedicated_clean_cubic.pt"}


def country_target(ckpt):
    """Build a 64->64->1 readout that outputs only the country logit (idx 5)."""
    head = R.Head(); head.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)); head.eval()
    W6, b6 = head.layers[6].weight.detach(), head.layers[6].bias.detach()
    W8, b8 = head.layers[8].weight.detach(), head.layers[8].bias.detach()
    tgt = R.ReadoutTarget(d_out=1)
    with torch.no_grad():
        tgt.mlp_in.weight[:] = W6.T.unsqueeze(0)              # (1,64,64)
        tgt.mlp_out.weight[:] = W8[A.CI:A.CI+1].T.unsqueeze(0)  # (1,64,1)
        tgt.bias1[:] = b6.unsqueeze(0); tgt.bias2[:] = b8[A.CI:A.CI+1].unsqueeze(0)
    return tgt


def country_effect(spd, X, batch=2000):
    spd.eval()
    with torch.no_grad():
        full = spd(X[:batch]).squeeze(-1).squeeze(-1).numpy()
    e = np.zeros(spd.C)
    for c in range(spd.C):
        st = spd.set_subnet_to_zero(c, has_instance_dim=True)
        with torch.no_grad():
            abl = spd(X[:batch]).squeeze(-1).squeeze(-1).numpy()
        spd.restore_subnet(c, st, has_instance_dim=True)
        e[c] = np.sqrt(((full - abl) ** 2).mean())
    return e, full


def run(model_key, C, seed=0, steps=10000):
    set_seed(seed)
    ckpt = REG[model_key]
    tr = np.load(BDC/"cache/train.npz"); te = np.load(BDC/"cache/test.npz")
    L_tr, _ = R.compute_L(ckpt, tr["emb"]); L_te, _ = R.compute_L(ckpt, te["emb"])
    if model_key == "e2b_cubic":
        yc = np.load(ROOT/"results"/"task2"/"cubic_labels.npz")["c_te"]
    else:
        yc = te["labels"][:, A.CI]
    target = country_target(ckpt)
    spd = R.ReadoutSPD(C=C, m=None, d_out=1, n_instances=1)
    with torch.no_grad():
        spd.bias1[:] = target.bias1.clone(); spd.bias2[:] = target.bias2.clone()
    spd.bias1.requires_grad = False; spd.bias2.requires_grad = False
    X = torch.tensor(L_tr, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(X, torch.zeros(len(X))), batch_size=256, shuffle=True)
    tc = TMSTaskConfig(feature_probability=0.05, train_bias=False, bias_val=0.0, pretrained_model_path="d")
    cfg = Config(batch_size=256, steps=steps, print_freq=steps, image_freq=None, save_freq=None,
                 lr=1e-3, C=C, m=None, topk=10.0, batch_topk=True, unit_norm_matrices=True,
                 param_match_coeff=1.0, topk_recon_coeff=1.0, act_recon_coeff=1.0,
                 schatten_coeff=1.0, schatten_pnorm=0.9, attribution_type="gradient", task_config=tc)
    optimize(model=spd, config=cfg, device="cpu", dataloader=dl, target_model=target,
             param_names=["mlp_in", "mlp_out"], out_dir=None, plot_results_fn=None)

    Xte = torch.tensor(L_te, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        t = target(Xte).squeeze(-1).squeeze(-1).numpy()
    e, full = country_effect(spd, Xte)
    PR = float((e.sum() ** 2) / (e ** 2).sum())                # effective # components
    # recon-95: greedily add components by effect, country AUC vs full
    order = np.argsort(-e); aucs = []
    for k in range(1, C + 1):
        mask = torch.zeros(Xte.shape[0], 1, C, dtype=torch.bool); mask[:, :, order[:k]] = True
        with torch.no_grad():
            z = spd(Xte, topk_mask=mask).squeeze(-1).squeeze(-1).numpy()
        aucs.append(A.auc(z, yc))
    fullauc = aucs[-1]; k95 = next((k for k, a in enumerate(aucs, 1) if a >= 0.95 * fullauc), C)
    rep = {"model_key": model_key, "C": C, "seed": seed,
           "country_auc_target": A.auc(t, yc), "country_auc_spd": A.auc(full, yc),
           "PR_effective_components": PR, "PR_frac": PR / C, "recon_k95": int(k95),
           "n_eff_above_5pct": int((e > 0.05 * e.max()).sum())}
    out = ROOT/"results"/"country_only"; out.mkdir(parents=True, exist_ok=True)
    (out/f"{model_key}_C{C}_sd{seed}.json").write_text(json.dumps(rep, indent=2))
    print(f"[{model_key} C{C}] tgtAUC {rep['country_auc_target']:.3f} spdAUC {rep['country_auc_spd']:.3f} "
          f"| PR {PR:.1f} (PR/C {PR/C:.2f}) | recon95 {k95} | n>5% {rep['n_eff_above_5pct']}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model_key"); ap.add_argument("C", type=int)
    a = ap.parse_args(); run(a.model_key, a.C)
