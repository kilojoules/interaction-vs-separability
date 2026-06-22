"""Task 3 -- minimality/simplicity tradeoff curve from the schatten sweep.

For each model we ran SPD at several schatten (minimality) coefficients.  We plot
faithfulness (country AUC / recon) vs number of country components, tracing the
tension: stronger minimality -> fewer components but (eventually) worse faithfulness.
If the curve is flat (faithfulness independent of #components down to the true
mechanism count) there is no tension and we say so.

Run (apd env): .../envs/apd/bin/python experiments/e3_sweep_figure.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SPD = ROOT / "results" / "spd_sweep"
SCH = [0.1, 1.0, 10.0]
MODELS = [("order2_model", "#e08a1e", "order-2 gated"), ("order3_pinwheel", "#6a1b9a", "order-3 cubic")]


def load(key, sc, steps_tag="C40"):
    # sweep runs use steps=6000; tag is key_C40_sc<sc>_sd0
    p = SPD / f"{key}_{steps_tag}_sc{sc}_sd0.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    rows = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, col, lab in MODELS:
        pts = []
        for sc in SCH:
            d = load(key, sc)
            if d is None: continue
            pts.append((sc, d["faithfulness"]["per_feature_auc"]["country"]["spd_full"],
                        d["faithfulness"]["recon_rel"], d["summary"]["n_country_components"],
                        d["summary"]["C_alive"], d["summary"]["mean_monosemanticity_alive"]))
        if not pts: continue
        pts.sort()
        sc, cauc, rel, ncc, calive, mono = map(np.array, zip(*pts))
        rows[key] = {"schatten": sc.tolist(), "country_auc": cauc.tolist(),
                     "recon_rel": rel.tolist(), "n_country_components": ncc.tolist(),
                     "C_alive": calive.tolist(), "monosemanticity": mono.tolist()}
        ax1.plot(sc, ncc, "o-", color=col, label=lab)
        ax2.plot(ncc, cauc, "o-", color=col, label=lab)
        for s, n in zip(sc, ncc): ax1.annotate(f"sc={s}", (s, n), fontsize=7)
    ax1.set_xscale("log"); ax1.set_xlabel("schatten (minimality) coeff")
    ax1.set_ylabel("# country components"); ax1.set_title("Minimality pressure -> fewer components")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25, which="both")
    ax2.set_xlabel("# country components"); ax2.set_ylabel("country AUC (SPD full)")
    ax2.set_title("Tradeoff: faithfulness vs component count"); ax2.set_ylim(0.45, 1.03)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "figs" / "spd_minimality_tradeoff.png", dpi=140, bbox_inches="tight")
    (SPD / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
    print("saved figs/spd_minimality_tradeoff.png"); print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
