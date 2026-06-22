"""Aggregate the Task-1 SPD runs into the headline figure + summary.

Reads results/spd/<order>_C<C>_sc<sc>_sd<seed>.json (+ .pth for cross-seed MMCS)
and produces figs/spd_components.png and results/spd/summary.json.

Run (apd env): /Users/julianquick/miniconda3/envs/apd/bin/python experiments/e1_figure.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import spd_readout as R
import spd_analyze as A
SPD = ROOT / "results" / "spd"
FEATS = A.FEATS
ORDERS = [("order1_vanilla", "order-1\nlinear"), ("order2_model", "order-2\ngated"),
          ("order3_pinwheel", "order-3\ncubic")]


def load_json(key, C=40, sc=1.0, sd=0):
    p = SPD / f"{key}_C{C}_sc{sc}_sd{sd}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_spd_model(key, C=40, sc=1.0, sd=0):
    p = SPD / f"{key}_C{C}_sc{sc}_sd{sd}.pth"
    if not p.exists(): return None
    m = R.ReadoutSPD(C=C, m=None)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


def seed_mmcs(key, C=40, sc=1.0):
    a = load_spd_model(key, C, sc, 0); b = load_spd_model(key, C, sc, 1)
    if a is None or b is None: return None
    return A.mmcs(A.component_param_vectors(a), A.component_param_vectors(b))


def main():
    data = {k: load_json(k) for k, _ in ORDERS}
    if any(v is None for v in data.values()):
        missing = [k for k, v in data.items() if v is None]
        print(f"missing seed-0 results: {missing}"); return
    summary = {}
    for k, _ in ORDERS:
        d = data[k]; s = d["summary"]; f = d["faithfulness"]
        summary[k] = {
            "country_auc_target": f["per_feature_auc"]["country"]["target"],
            "country_auc_spd": f["per_feature_auc"]["country"]["spd_full"],
            "recon_rel": f["recon_rel"],
            "C_alive": s["C_alive"],
            "n_country_components": s["n_country_components"],
            "n_country_dominant": s["n_country_dominant_components"],
            "monosemanticity": s["mean_monosemanticity_alive"],
            "seed_mmcs": seed_mmcs(k),
        }
        print(f"{k}: {summary[k]}")
    (SPD / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure ----
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.42, wspace=0.32)
    labels = [lab for _, lab in ORDERS]; x = np.arange(3)

    ax = fig.add_subplot(gs[0, 0])
    ncc = [summary[k]["n_country_components"] for k, _ in ORDERS]
    ax.bar(x, ncc, color=["#1a7a4a", "#e08a1e", "#6a1b9a"])
    for i, v in enumerate(ncc): ax.text(i, v + 0.1, str(v), ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("# SPD components serving country")
    ax.set_title("Rank inflation (APD components)")

    ax = fig.add_subplot(gs[0, 1])
    at = [summary[k]["country_auc_target"] for k, _ in ORDERS]
    asp = [summary[k]["country_auc_spd"] for k, _ in ORDERS]
    ax.plot(x, at, "ks-", label="target"); ax.plot(x, asp, "o--", color="#c0392b", label="SPD (full)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("country AUC"); ax.set_title("Faithfulness"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    mono = [summary[k]["monosemanticity"] for k, _ in ORDERS]
    mm = [summary[k]["seed_mmcs"] for k, _ in ORDERS]
    ax.plot(x, mono, "o-", color="#2e86c1", label="monosemanticity")
    if all(v is not None for v in mm):
        ax.plot(x, mm, "^--", color="#8e44ad", label="cross-seed MMCS")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylim(0, 1.05)
    ax.set_title("Component quality"); ax.legend(fontsize=8)

    # component x feature effect heatmaps
    for j, (k, lab) in enumerate(ORDERS):
        ax = fig.add_subplot(gs[1, j])
        eff = np.array(data[k]["summary"]["effect_matrix"])           # (C,8)
        norms = np.array(data[k]["summary"]["component_norms"])
        alive = norms > 0.02 * norms.max()
        E = eff[alive]
        order = np.argsort(-E.max(axis=1))
        im = ax.imshow(E[order].T, aspect="auto", cmap="magma",
                       vmax=np.percentile(eff, 99) + 1e-9)
        ax.set_yticks(range(8)); ax.set_yticklabels(FEATS, fontsize=7)
        ax.set_xlabel(f"{lab.strip().replace(chr(10),' ')}  ({alive.sum()} alive comps)", fontsize=8)
        ax.axhline(A.CI + 0.5, color="cyan", lw=0.6); ax.axhline(A.CI - 0.5, color="cyan", lw=0.6)
        if j == 0: ax.set_ylabel("output feature", fontsize=8)
    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046, label="ablation effect")
    fig.suptitle("Real APD/SPD on the readout: country needs more components as interaction order rises\n"
                 "(country row outlined; bottom = component x feature causal-effect matrices)", fontsize=11)
    fig.savefig(ROOT / "figs" / "spd_components.png", dpi=140, bbox_inches="tight")
    print(f"saved {ROOT/'figs'/'spd_components.png'}")


if __name__ == "__main__":
    main()
