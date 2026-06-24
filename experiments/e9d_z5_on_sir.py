"""Plot the country logit z_5 as a function of SIR's (φ₁, φ₂). Per code:
  col 1: the (φ₁,φ₂) plane colored by z_5  (the response surface)
  col 2: z_5 vs φ₁   (marginal)
  col 3: z_5 vs φ₂   (marginal)
For the phase model the surface is the pinwheel/spiral (z_5≈sin 3θ) and NEITHER marginal
is monotonic — z_5 depends on the angle, not on either SIR axis alone.

Run: python experiments/e9d_z5_on_sir.py
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "results" / "geometry"
CODES = [("linear", "order-1 linear"), ("gate", "order-2 gate"),
         ("phase", "order-3 PHASE  sin 3θ"), ("poly", "order-3 poly"),
         ("xor", "order-3 gated/XOR")]


def main():
    n = len(CODES)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 2.7 * n))
    for r, (code, lab) in enumerate(CODES):
        d = np.load(G/f"{code}_sir.npz")
        proj = d["proj"] - d["proj"].mean(0); z = d["z5"]
        lim = 3.0 * proj.std(0).mean()
        vmax = np.percentile(np.abs(z - z.mean()), 98); vc = z - z.mean()
        # col1: (φ1,φ2) colored by z5
        a = axes[r, 0]
        sc = a.scatter(proj[:, 0], proj[:, 1], c=vc, s=6, cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, alpha=0.8, linewidths=0)
        a.set_aspect("equal"); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel("SIR φ₁", fontsize=8)
        a.set_ylabel("SIR φ₂", fontsize=8); a.set_title(f"{lab}\nz₅ over (φ₁,φ₂)", fontsize=9)
        fig.colorbar(sc, ax=a, fraction=0.046, pad=0.03).set_label("z₅", fontsize=7)
        # col2/3: marginals
        for c, (ax, xi, name) in enumerate([(axes[r, 1], 0, "φ₁"), (axes[r, 2], 1, "φ₂")]):
            ax.scatter(proj[:, xi], z, s=4, c="#444", alpha=0.35, linewidths=0)
            ax.set_xlabel(f"SIR {name}", fontsize=8); ax.set_ylabel("z₅", fontsize=8)
            ax.set_title(f"z₅ vs {name}", fontsize=9); ax.grid(alpha=0.2)
    fig.suptitle("Country logit z₅ as a function of SIR's (φ₁, φ₂) — phase is a pinwheel in "
                 "the plane and non-monotonic in either axis alone", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(ROOT/"figs"/"z5_on_sir.png", dpi=135, bbox_inches="tight")
    print("saved figs/z5_on_sir.png")


if __name__ == "__main__":
    main()
