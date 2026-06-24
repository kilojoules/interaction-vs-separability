"""Compare the country (φ₁,φ₂) plane found by the gradient ACTIVE SUBSPACE (e9) vs
SLICED INVERSE REGRESSION (e9b). Left col = active, right col = SIR; colored by country.
Annotates the principal angle between the two 2-D subspaces.

Run: python experiments/e9c_sir_vs_active.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "results" / "geometry"
CODES = [("linear", "order-1 linear"), ("gate", "order-2 gate"),
         ("phase", "order-3 PHASE  sin 3θ"), ("poly", "order-3 poly"),
         ("xor", "order-3 gated/XOR")]


def scat(ax, proj, y, title):
    proj = proj - proj.mean(0); lim = 3.0 * proj.std(0).mean()
    for cls, col in [(0, "#2c7fb8"), (1, "#d7301f")]:
        m = y == cls
        ax.scatter(proj[m, 0], proj[m, 1], s=5, c=col, alpha=0.5, linewidths=0)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("φ₁", fontsize=8); ax.set_ylabel("φ₂", fontsize=8)


def main():
    fig, axes = plt.subplots(len(CODES), 2, figsize=(6.2, 2.9 * len(CODES)))
    for r, (code, lab) in enumerate(CODES):
        a = np.load(G/f"{code}.npz"); s = np.load(G/f"{code}_sir.npz")
        ang = s["angle_to_active"]
        scat(axes[r, 0], a["proj"], a["label"], f"{lab}\nACTIVE SUBSPACE (gradient)")
        scat(axes[r, 1], s["proj"], s["label"],
             f"SLICED INVERSE REGRESSION\n∠ to active = {ang[0]:.0f}–{ang[1]:.0f}°")
    fig.suptitle("Country plane: gradient active subspace (left) vs sliced inverse regression "
                 "(right)\nSIR matches the active plane only for the 2-D phase carrier",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(ROOT/"figs"/"sir_vs_active.png", dpi=135, bbox_inches="tight")
    print("saved figs/sir_vs_active.png")


if __name__ == "__main__":
    main()
