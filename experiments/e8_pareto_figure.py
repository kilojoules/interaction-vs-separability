"""Pareto-area test: is the minimality<->faithfulness trade-off largest for phase?

For each code's faithful country-only decomposition, the recon-curve (country AUC
vs # components kept, greedy by causal effect) IS the minimality-faithfulness
frontier. The area between it and the ideal (faithful at few components) = the
dominated Pareto region = how much faithfulness you sacrifice to be parsimonious.
Hypothesis: largest for phase.

Run (apd env or base): python experiments/e8_pareto_figure.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "results" / "stress"
LAB = {"phase": ("order-3 phase  (sin 3θ)", "#6a1b9a"),
       "poly": ("order-3 polynomial  (a³−3ab²)", "#1a7a4a"),
       "xor": ("order-3 gated/XOR  (a⊕b⊕c)", "#2e86c1"),
       "gate": ("order-2 gate  (food⊕sent)", "#e08a1e")}


def find(model):
    """The pure-recon C=40 st=20000 lr=1e-3 run for a code (the controlled Pareto config)."""
    best = None
    for p in S.glob("*.json"):
        d = json.loads(p.read_text()); c = d["config"]
        if (c.get("model") == model and c["C"] == 40 and c["steps"] == 20000 and c["lr"] == 1e-3
                and c["schatten"] == "none" and c["topk"] == "none" and not c["multi"]
                and "recon_curve" in d):
            best = d
    return best


def main():
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    print(f"{'code':<28}{'faithAUC':>9}{'recon95':>9}{'PR':>7}{'pareto_area':>13}")
    rows = []
    for m, (lab, col) in LAB.items():
        d = find(m)
        if d is None:
            print(f"{lab:<28}   (not run yet)"); continue
        curve = np.array(d["recon_curve"]); full = curve[-1]; k = np.arange(1, len(curve) + 1)
        area = d.get("pareto_area", float(np.mean(full - curve) / (full + 1e-9)))
        rows.append((m, lab, d["country_auc_spd"], d["recon_k95"], d.get("PR_effective_components"), area))
        ax.plot(k / len(curve), curve, "-", color=col, lw=2, label=f"{lab}  (area={area:.3f})")
        ax.fill_between(k / len(curve), curve, full, color=col, alpha=0.10)
    for m, lab, fa, r95, pr, area in sorted(rows, key=lambda r: -r[5]):
        print(f"{lab:<28}{fa:>9.3f}{r95:>9}{(pr or 0):>7.1f}{area:>13.3f}")
    ax.set_xlabel("fraction of components kept  (parsimony →)")
    ax.set_ylabel("country AUC reached  (faithfulness)")
    ax.set_title("Minimality↔faithfulness Pareto frontier per code\n"
                 "(shaded = dominated area; larger = worse trade-off)")
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "figs" / "spd_pareto.png", dpi=140, bbox_inches="tight")
    if rows:
        rank = sorted(rows, key=lambda r: -r[5])
        print(f"\nLargest Pareto area: {rank[0][1]} ({rank[0][5]:.3f}). "
              f"Hypothesis (phase largest): {'CONFIRMED' if rank[0][0]=='phase' else 'NOT confirmed'}")
    print("saved figs/spd_pareto.png")


if __name__ == "__main__":
    main()
