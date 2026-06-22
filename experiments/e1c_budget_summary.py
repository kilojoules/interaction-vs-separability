"""Aggregate the budget sweep: is country's component count a plateau (genuine
cost) or does it saturate the budget (never cleanly resolved)?  Reads
results/budget/<model>_C<C>_sd<seed>.json and prints a table + makes a figure.

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
MODELS = [("order2_model", "order-2 gated", "#e08a1e"), ("order3_pinwheel", "order-3 cubic", "#6a1b9a")]


def load(m, C, s):
    p = B / f"{m}_C{C}_sd{s}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    print(f"{'model':<16}{'C':>4}{'sd':>3}{'cAUC_spd':>9}{'recon_rel':>10}"
          f"{'recon95':>8}{'serves_c':>9}{'dom_c':>6}{'dom_lin':>8}")
    agg = {}
    for m, lab, _ in MODELS:
        for C in BUDGETS:
            rec95s, domc, doml, sv, cauc, rel = [], [], [], [], [], []
            for s in SEEDS:
                d = load(m, C, s)
                if d is None:
                    print(f"{m:<16}{C:>4}{s:>3}   MISSING"); continue
                print(f"{m:<16}{C:>4}{s:>3}{d['country_auc_spd']:>9.3f}{d['recon_rel']:>10.4f}"
                      f"{d['recon_k95']:>8}{d['serves_country']:>9}{d['dominant_country']:>6}"
                      f"{d['dominant_linear_sum']:>8}")
                rec95s.append(d["recon_k95"]); domc.append(d["dominant_country"])
                doml.append(d["dominant_linear_sum"]); sv.append(d["serves_country"])
                cauc.append(d["country_auc_spd"]); rel.append(d["recon_rel"])
            if rec95s:
                agg[(m, C)] = dict(recon95=rec95s, dom=domc, dom_lin=doml, serves=sv, cauc=cauc, rel=rel)
    print()
    # plateau vs climb verdict
    for m, lab, _ in MODELS:
        r = [np.mean(agg[(m, C)]["recon95"]) for C in BUDGETS if (m, C) in agg]
        dom = [np.mean(agg[(m, C)]["dom"]) for C in BUDGETS if (m, C) in agg]
        Cs = [C for C in BUDGETS if (m, C) in agg]
        if len(Cs) >= 2:
            print(f"{lab}: recon95 {[round(x,1) for x in r]} over C={Cs}  "
                  f"(plateau if ~flat)   |   dominant {[round(x,1) for x in dom]} vs ceilings {Cs}  "
                  f"(climb if tracks ceiling; frac={[round(d/C,2) for d,C in zip(dom,Cs)]})")
    (B / "summary.json").write_text(json.dumps(
        {f"{m}_C{C}": v for (m, C), v in agg.items()}, indent=2))

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for m, lab, col in MODELS:
        Cs = [C for C in BUDGETS if (m, C) in agg]
        for s in SEEDS:
            r = [load(m, C, s)["recon_k95"] for C in Cs if load(m, C, s)]
            ax1.plot(Cs, r, "o", color=col, alpha=0.5, ms=5)
        rm = [np.mean(agg[(m, C)]["recon95"]) for C in Cs]
        ax1.plot(Cs, rm, "o-", color=col, lw=2, label=lab)
        domm = [np.mean(agg[(m, C)]["dom"]) for C in Cs]
        ax2.plot(Cs, domm, "o-", color=col, lw=2, label=lab)
    ax1.set_xlabel("budget C (total components)"); ax1.set_ylabel("components to reconstruct country @95%")
    ax1.set_title("Genuine need (recon-95): plateau?"); ax1.set_ylim(0, None); ax1.grid(alpha=0.25); ax1.legend(fontsize=8)
    ax2.plot(BUDGETS, BUDGETS, "k:", lw=1, label="ceiling (=C)")
    ax2.set_xlabel("budget C (total components)"); ax2.set_ylabel("# components dominated by country")
    ax2.set_title("Country-dominant count: climb toward ceiling?"); ax2.grid(alpha=0.25); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figs" / "spd_budget_sweep.png", dpi=140, bbox_inches="tight")
    print(f"\nsaved figs/spd_budget_sweep.png")


if __name__ == "__main__":
    main()
