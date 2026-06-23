"""Aggregate the budget sweep + the polynomial-cubic control (e2b).

Question: is country's component count a plateau (genuine cost), does it saturate
the budget (never cleanly resolved), and is saturation specific to PHASE geometry
or to order-3/cubic in general?  The e2b control (homogeneous cubic, non-dedicated)
answers the last part.  Reads results/budget/<model>_C<C>_sd<seed>.json.

Run (apd env or base): python experiments/e1c_budget_summary.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "results" / "budget"
BUDGETS = [40, 80, 120]; SEEDS = [0, 1]
MODELS = [("order2_model", "order-2 gated", "#e08a1e"),
          ("order3_pinwheel", "order-3 phase  (sin 3θ)", "#6a1b9a"),
          ("e2b_cubic", "order-3 polynomial  (a³−3ab²)", "#1a7a4a")]


def load(m, C, s):
    p = B / f"{m}_C{C}_sd{s}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    print(f"{'model':<18}{'C':>4}{'sd':>3}{'cAUC':>7}{'rel':>7}{'recon95':>8}"
          f"{'dom':>6}{'dom/C':>7}{'redund':>8}")
    agg = {}
    for m, lab, _ in MODELS:
        for C in BUDGETS:
            rec95s, domc, redu, cauc, rel = [], [], [], [], []
            for s in SEEDS:
                d = load(m, C, s)
                if d is None: continue
                r = d["dominant_country"] / max(d["recon_k95"], 1)
                print(f"{m:<18}{C:>4}{s:>3}{d['country_auc_spd']:>7.2f}{d['recon_rel']:>7.3f}"
                      f"{d['recon_k95']:>8}{d['dominant_country']:>6}{d['dominant_country']/C:>7.2f}{r:>8.1f}")
                rec95s.append(d["recon_k95"]); domc.append(d["dominant_country"])
                redu.append(r); cauc.append(d["country_auc_spd"]); rel.append(d["recon_rel"])
            if rec95s:
                agg[(m, C)] = dict(recon95=rec95s, dom=domc, redund=redu, cauc=cauc, rel=rel)
    print()
    for m, lab, _ in MODELS:
        Cs = [C for C in BUDGETS if (m, C) in agg]
        if not Cs: continue
        red = [np.mean(agg[(m, C)]["redund"]) for C in Cs]
        print(f"{lab}: redundancy (dom/recon95) {[round(x,1) for x in red]} over C={Cs}")
    (B / "summary.json").write_text(json.dumps({f"{m}_C{C}": v for (m, C), v in agg.items()}, indent=2))

    # ---- 3-panel figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for m, lab, col in MODELS:
        Cs = [C for C in BUDGETS if (m, C) in agg]
        if not Cs: continue
        r95 = [np.mean(agg[(m, C)]["recon95"]) for C in Cs]
        dom = [np.mean(agg[(m, C)]["dom"]) for C in Cs]
        red = [np.mean(agg[(m, C)]["redund"]) for C in Cs]
        axes[0].plot(Cs, r95, "o-", color=col, lw=2, label=lab)
        axes[1].plot(Cs, dom, "o-", color=col, lw=2, label=lab)
        axes[2].plot(Cs, red, "o-", color=col, lw=2, label=lab)
    axes[0].set_title("Genuine need (recon-95)"); axes[0].set_ylabel("components to reconstruct @95%")
    axes[0].set_ylim(0, None)
    axes[1].plot(BUDGETS, BUDGETS, "k:", lw=1, label="ceiling (=C)")
    axes[1].set_title("Country-dominant count"); axes[1].set_ylabel("# components dominated by country")
    axes[2].set_title("Redundancy = dominant / recon-95"); axes[2].set_ylabel("country-dominant beyond genuine need")
    axes[2].axhline(1, color="k", ls=":", lw=1, alpha=0.5)
    for ax in axes:
        ax.set_xlabel("budget C (total components)"); ax.grid(alpha=0.25); ax.legend(fontsize=7.5)
    fig.suptitle("Saturation is specific to PHASE geometry: the phase cubic tracks the budget (redundancy 8–30×); "
                 "a polynomial cubic of the same degree resolves (≈0×), like the gate",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ROOT / "figs" / "spd_budget_sweep.png", dpi=140, bbox_inches="tight")
    print("saved figs/spd_budget_sweep.png")


if __name__ == "__main__":
    main()
