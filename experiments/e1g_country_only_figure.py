"""Figure for the confound-free country-only decomposition: effective component
count (PR) vs budget, and faithfulness vs budget.  PR plateauing well below the
C ceiling = the feature resolves into a bounded set (no saturation)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "results" / "country_only"
BUD = [40, 80, 120]
MODELS = [("order2_model", "order-2 gate", "#e08a1e"),
          ("order3_pinwheel", "order-3 phase  (sin 3θ)", "#6a1b9a"),
          ("e2b_cubic", "order-3 polynomial  (a³−3ab²)", "#1a7a4a"),
          ("order3_xor", "order-3 gated/XOR  (a⊕b⊕c)", "#2e86c1")]


def get(m, C, k):
    p = B / f"{m}_C{C}_sd0.json"
    return json.loads(p.read_text())[k] if p.exists() else None


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
ax1.plot(BUD, BUD, "k:", lw=1, label="ceiling (=C, i.e. saturation)")
for m, lab, col in MODELS:
    pr = [get(m, C, "PR_effective_components") for C in BUD]
    au = [get(m, C, "country_auc_spd") for C in BUD]
    ax1.plot(BUD, pr, "o-", color=col, lw=2, label=lab)
    ax2.plot(BUD, au, "o-", color=col, lw=2, label=lab)
ax1.set_xlabel("budget C (total components)"); ax1.set_ylabel("effective # components carrying country (PR)")
ax1.set_title("Country-only decomposition: PR plateaus far below C\n(no saturation; bounded resolution)")
ax1.legend(fontsize=8); ax1.grid(alpha=0.25)
ax2.axhline(0.95, color="k", ls=":", lw=1, alpha=0.5)
ax2.set_xlabel("budget C (total components)"); ax2.set_ylabel("country AUC (SPD, full)")
ax2.set_title("Faithfulness vs budget\n(phase code resists faithful high-C decomposition)")
ax2.set_ylim(0.5, 1.02); ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "figs" / "spd_country_only.png", dpi=140, bbox_inches="tight")
print("saved figs/spd_country_only.png")
