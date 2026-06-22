"""Budget sweep analysis: load a saved SPD .pth at a given C and compute the
uniform budget_report (faithfulness + serves/dominant/recon-95 component counts).
Used to test whether country's component count plateaus or saturates with budget.

Usage (apd env):  python experiments/e1b_budget.py MODEL_KEY C SEED
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import spd_readout as R, spd_analyze as A
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")
REG = {"order1_vanilla": BDC/"model_vanilla.pt", "order2_model": BDC/"model.pt",
       "order3_pinwheel": BDC/"model_pinwheel.pt"}


def analyze(model_key, C, seed, sc=1.0):
    pth = ROOT/"results"/"spd"/f"{model_key}_C{C}_sc{sc}_sd{seed}.pth"
    if not pth.exists():
        print(f"MISSING {pth}"); return None
    ckpt = REG[model_key]
    te = np.load(BDC/"cache/test.npz")
    L_te, _ = R.compute_L(ckpt, te["emb"]); y = te["labels"]
    target = R.load_readout_target(ckpt)
    spd = R.ReadoutSPD(C=C, m=None); spd.load_state_dict(torch.load(pth, map_location="cpu")); spd.eval()
    X = torch.tensor(L_te, dtype=torch.float32).unsqueeze(1)
    rep = A.budget_report(spd, target, X, y)
    rep.update({"model_key": model_key, "seed": seed})
    out = ROOT/"results"/"budget"; out.mkdir(parents=True, exist_ok=True)
    (out/f"{model_key}_C{C}_sd{seed}.json").write_text(json.dumps(rep, indent=2))
    print(f"[{model_key} C{C} sd{seed}] country AUC tgt {rep['country_auc_target']:.3f} "
          f"spd {rep['country_auc_spd']:.3f} recon_rel {rep['recon_rel']:.3f} | "
          f"serves {rep['serves_country']}(lin {rep['serves_linear_mean']:.1f}) "
          f"dom {rep['dominant_country']} recon95 {rep['recon_k95']}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key"); ap.add_argument("C", type=int); ap.add_argument("seed", type=int)
    a = ap.parse_args(); analyze(a.model_key, a.C, a.seed)
