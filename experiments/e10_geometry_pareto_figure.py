"""Per-encoding 2-panel figure: LEFT = active-subspace geometry (φ₂ vs φ₁ colored
by country, from e9); RIGHT = faithfulness↔minimality frontier (country AUC vs #
components kept), one CONTOUR per Schatten 'simplicity' level (incl. λ=0 pure-recon).

Run (apd env or base): python experiments/e10_geometry_pareto_figure.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "results" / "geometry"; S = ROOT / "results" / "stress"
CODES = [("linear", "order-1 linear"), ("gate", "order-2 gate  food⊕sent"),
         ("phase", "order-3 PHASE  sin 3θ"), ("poly", "order-3 poly  a³−3ab²"),
         ("xor", "order-3 gated/XOR  a⊕b⊕c")]
LAMBDAS = [("none", "λ=0 (pure recon)", "#1a7a4a"), ("0.1", "λ=0.1", "#2e86c1"),
           ("1.0", "λ=1", "#e08a1e"), ("10.0", "λ=10", "#c0392b")]


def curve(code, sc):
    tk = "none" if sc == "none" else "10"
    for p in S.glob(f"{code}_C40_st10000_lr0.001_sd0_*_sc{sc}_tk{tk}_*_co.json"):
        d = json.loads(p.read_text())
        if "recon_curve" in d:
            return np.array(d["recon_curve"]), d["country_auc_spd"], d["country_auc_target"]
    return None, None, None


def main():
    n = len(CODES)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.7 * n))
    for r, (code, lab) in enumerate(CODES):
        axg, axp = axes[r, 0], axes[r, 1]
        # LEFT: geometry
        gp = G / f"{code}.npz"
        if gp.exists():
            d = np.load(gp); proj = d["proj"]; y = d["label"]; vf = d["var_frac"]
            proj = proj - proj.mean(0)                 # center so angular structure shows
            sd = proj.std(0).mean(); lim = 3.0 * sd
            for cls, col in [(0, "#2c7fb8"), (1, "#d7301f")]:
                m = y == cls
                axg.scatter(proj[m, 0], proj[m, 1], s=5, c=col, alpha=0.5, linewidths=0,
                            label=f"country={cls}")
            axg.set_title(f"{lab}\nactive subspace (var {vf[0]:.2f}/{vf[1]:.2f})", fontsize=9)
            axg.set_xlabel("φ₁", fontsize=8); axg.set_ylabel("φ₂", fontsize=8)
            axg.set_xticks([]); axg.set_yticks([])
            axg.set_aspect("equal"); axg.set_xlim(-lim, lim); axg.set_ylim(-lim, lim)
            if r == 0: axg.legend(fontsize=6.5, markerscale=2, loc="upper right")
        # RIGHT: Pareto contours (absolute country AUC vs # components — NOT normalized)
        any_c = False; ceil = None; ncomp = 40
        for sc, slab, col in LAMBDAS:
            c, full, tgt = curve(code, sc)
            if c is None: continue
            any_c = True; ceil = tgt; ncomp = len(c)
            axp.plot(np.arange(1, len(c) + 1), c, "-", color=col, lw=1.8,
                     label=f"{slab}  → {full:.2f}")
        if any_c:
            axp.axhline(ceil, ls="--", color="0.35", lw=1.0, label=f"model ceiling {ceil:.2f}")
            axp.set_ylim(0.45, 1.02); axp.set_xlim(0, ncomp)
            axp.legend(fontsize=6.2, loc="lower right")
        else:
            axp.text(0.5, 0.5, "country is 1-D\n(trivially decomposed)", ha="center", va="center",
                     fontsize=9, transform=axp.transAxes)
        axp.set_title("faithfulness ↔ minimality frontier", fontsize=9)
        axp.set_xlabel("# components kept (of 40, greedy by causal effect)", fontsize=8)
        axp.set_ylabel("country AUC (absolute)", fontsize=8); axp.grid(alpha=0.2)
    fig.suptitle("Encoding geometry (left) vs decomposition trade-off (right) — the phase code's "
                 "frontier alone collapses under minimality", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(ROOT / "figs" / "spd_geometry_pareto.png", dpi=135, bbox_inches="tight")
    print("saved figs/spd_geometry_pareto.png")


if __name__ == "__main__":
    main()
