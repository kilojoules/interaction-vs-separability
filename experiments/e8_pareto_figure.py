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
    """MATCHED-CONFIG run (C=40, 10k steps, lr=1e-3, pure-recon) for every code, so that
    area / recon_rel / shared / isolated in one row all come from the SAME run (verify-rewrite
    B1). Phase is faithful here (recon_rel 0.030); area is robust across steps (0.32-0.36)."""
    for p in S.glob("*.json"):
        d = json.loads(p.read_text()); c = d["config"]
        if (c.get("model") == model and c["C"] == 40 and c["steps"] == 10000 and c["lr"] == 1e-3
                and c["schatten"] == "none" and c["topk"] == "none" and not c["multi"]
                and "recon_curve" in d and "recon_k95_isolated" in d):
            return d
    return None


def main():
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    print(f"{'code':<28}{'faithAUC':>9}{'recon_rel':>10}{'recon95':>9}{'iso95':>7}{'area':>8}")
    rows = []
    for m, (lab, col) in LAB.items():
        d = find(m)
        if d is None:
            print(f"{lab:<28}   (not run yet)"); continue
        curve = np.array(d["recon_curve"]); full = curve[-1]; k = np.arange(1, len(curve) + 1)
        area = d.get("pareto_area", float(np.mean(full - curve) / (full + 1e-9)))
        rr = d["recon_rel"]; iso = d.get("recon_k95_isolated")
        rows.append((m, lab, d["country_auc_spd"], rr, d["recon_k95"], iso, area))
        ax.plot(k / len(curve), curve, "-", color=col, lw=2,
                label=f"{lab}  area={area:.2f}, recon_rel={rr:.3f}")
        ax.fill_between(k / len(curve), curve, full, color=col, alpha=0.10)
    for m, lab, fa, rr, r95, iso, area in sorted(rows, key=lambda r: -r[6]):
        print(f"{lab:<28}{fa:>9.3f}{rr:>10.3f}{r95:>9}{str(iso):>7}{area:>8.3f}")
    ax.set_xlabel("fraction of components kept  (greedy, by causal country effect →)")
    ax.set_ylabel("country AUC reached")
    ax.set_title("Reconstruction-spread frontier per code (pure-reconstruction factorization)\n"
                 "shaded = area below own ceiling.  NOTE: not at matched faithfulness — see recon_rel.")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "figs" / "spd_pareto.png", dpi=140, bbox_inches="tight")
    if rows:
        rank = sorted(rows, key=lambda r: -r[6])
        print(f"\nLargest area: {rank[0][1]} ({rank[0][6]:.3f}). "
              f"phase-largest: {'YES' if rank[0][0]=='phase' else 'NO'}  "
              f"(ordering robust to step/seed; absolute value & 'matched faithfulness' are NOT)")
    print("saved figs/spd_pareto.png")


if __name__ == "__main__":
    main()
