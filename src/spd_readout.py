"""Run the REAL APD/SPD algorithm (ApolloResearch `spd` package) on the BlueDot
readout stack -- closing the "we used a parameter-Jacobian fallback" gap.

The target is the readout `net.layers[6:]` of a Head checkpoint:
    L(64) --mlp_in--> 64 --ReLU--> --mlp_out--> 8 logits.
We wrap it as an `spd` target model and an `spd` SPD twin (each weight decomposed
into C subnetwork components via A@B), then call `spd.run_spd.optimize` -- the
same loss (param-match + topk-recon + act-recon + Schatten minimality) used in
the APD paper. Inputs are the *real* L activations for that model.

Requires the `apd` conda env (python 3.11, package `spd`):
    /Users/julianquick/miniconda3/envs/apd/bin/python
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.models.components import Linear, LinearComponent


# ----- the readout as an spd target / SPD twin -----------------------------
class ReadoutTarget(HookedRootModule):
    """Plain 2-layer readout in spd's Linear (hook_pre/hook_post) form."""
    def __init__(self, d_in=64, d_hidden=64, d_out=8, n_instances=1):
        super().__init__()
        self.n_instances = n_instances
        self.mlp_in = Linear(d_in, d_hidden, n_instances=n_instances)
        self.mlp_out = Linear(d_hidden, d_out, n_instances=n_instances)
        self.bias1 = nn.Parameter(torch.zeros(n_instances, d_hidden))
        self.bias2 = nn.Parameter(torch.zeros(n_instances, d_out))
        self.setup()

    def forward(self, x, **_):
        h = torch.relu(self.mlp_in(x) + self.bias1)
        return self.mlp_out(h) + self.bias2


class ReadoutSPD(SPDModel):
    """Component-decomposed twin: each weight = sum_C A_c @ B_c."""
    def __init__(self, C, m=None, d_in=64, d_hidden=64, d_out=8, n_instances=1):
        super().__init__()
        self.n_instances = n_instances
        self.C = C
        self.mlp_in = LinearComponent(d_in, d_hidden, C=C, m=m, n_instances=n_instances)
        self.mlp_out = LinearComponent(d_hidden, d_out, C=C, m=m, n_instances=n_instances)
        self.bias1 = nn.Parameter(torch.zeros(n_instances, d_hidden))
        self.bias2 = nn.Parameter(torch.zeros(n_instances, d_out))
        self.setup()

    def forward(self, x, topk_mask=None):
        h = torch.relu(self.mlp_in(x, topk_mask=topk_mask) + self.bias1)
        return self.mlp_out(h, topk_mask=topk_mask) + self.bias2


# ----- loading a Head readout into the spd target --------------------------
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
    def forward(self, x): return self.layers(x)


def load_readout_target(ckpt_path, n_instances=1):
    """Build a ReadoutTarget holding layers[6] (mlp_in) and layers[8] (mlp_out).
    spd Linear stores weight as (n_inst, d_in, d_out); nn.Linear is (d_out, d_in),
    so we transpose."""
    head = Head()
    head.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    head.eval()
    W6, b6 = head.layers[6].weight.detach(), head.layers[6].bias.detach()   # (64,64),(64)
    W8, b8 = head.layers[8].weight.detach(), head.layers[8].bias.detach()   # (8,64),(8)
    tgt = ReadoutTarget(n_instances=n_instances)
    with torch.no_grad():
        tgt.mlp_in.weight[:] = W6.T.unsqueeze(0)        # (1,64,64) = (in,out)
        tgt.mlp_out.weight[:] = W8.T.unsqueeze(0)       # (1,64,8)
        tgt.bias1[:] = b6.unsqueeze(0)
        tgt.bias2[:] = b8.unsqueeze(0)
    return tgt


def compute_L(ckpt_path, emb):
    """L = layers[:6](emb): the 64-d activation that feeds the readout."""
    head = Head(); head.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    head.eval()
    with torch.no_grad():
        L = head.layers[:6](torch.tensor(emb, dtype=torch.float32)).numpy()
        logits = head.layers(torch.tensor(emb, dtype=torch.float32)).numpy()
    return L, logits


def make_spd_from_target(target, C, m=None):
    """Build an SPD twin and freeze its biases to the target's (only the A/B
    component matrices are trained, matching the resid_mlp recipe)."""
    spd = ReadoutSPD(C=C, m=m, n_instances=target.n_instances)
    with torch.no_grad():
        spd.bias1[:] = target.bias1.detach().clone()
        spd.bias2[:] = target.bias2.detach().clone()
    spd.bias1.requires_grad = False
    spd.bias2.requires_grad = False
    return spd


# ----- small dependency-free helpers ---------------------------------------
def auc(scores, labels):
    """Mann-Whitney AUC, orientation-free (max(a, 1-a))."""
    s = np.asarray(scores, float); y = np.asarray(labels).astype(int)
    order = s.argsort(); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    a = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))
