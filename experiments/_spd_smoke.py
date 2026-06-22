"""Smoke test: does optimize() run end-to-end on our readout and reach faithfulness?"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spd_readout import load_readout_target, make_spd_from_target, auc

from spd.run_spd import Config, TMSTaskConfig, optimize
from torch.utils.data import TensorDataset, DataLoader

BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")
ckpt = BDC / "model.pt"
tr = np.load(BDC / "cache/train.npz"); te = np.load(BDC / "cache/test.npz")
L_tr = tr["L"].astype(np.float32); L_te = te["L"].astype(np.float32); y_te = te["labels"]

target = load_readout_target(ckpt)
spd = make_spd_from_target(target, C=40, m=None)

# data: real L activations, shaped (N, n_instances=1, 64)
X = torch.tensor(L_tr).unsqueeze(1)
ds = TensorDataset(X, torch.zeros(len(X)))
dl = DataLoader(ds, batch_size=256, shuffle=True)

tc = TMSTaskConfig(feature_probability=0.05, train_bias=False, bias_val=0.0, pretrained_model_path="dummy")
cfg = Config(batch_size=256, steps=4000, print_freq=1000, image_freq=None, save_freq=None, lr=1e-3,
             C=40, m=None, topk=10.0, batch_topk=True, unit_norm_matrices=True,
             param_match_coeff=1.0, topk_recon_coeff=1.0, act_recon_coeff=1.0,
             schatten_coeff=1.0, schatten_pnorm=0.9, attribution_type="gradient", task_config=tc)

optimize(model=spd, config=cfg, device="cpu", dataloader=dl, target_model=target,
         param_names=["mlp_in", "mlp_out"], out_dir=None, plot_results_fn=None)

# faithfulness on held-out test L
with torch.no_grad():
    Xte = torch.tensor(L_te).unsqueeze(1)
    t_out = target(Xte).squeeze(1).numpy()      # (N,8)
    s_out = spd(Xte).squeeze(1).numpy()         # full SPD (all C), no topk
mse = float(((t_out - s_out) ** 2).mean())
print(f"\nSMOKE RESULTS")
print(f"  recon MSE (spd-full vs target): {mse:.4e}")
print(f"  country AUC  target {auc(t_out[:,5], y_te[:,5]):.3f}   spd-full {auc(s_out[:,5], y_te[:,5]):.3f}")
print(f"  number-feat AUC target {auc(t_out[:,0], y_te[:,0]):.3f}   spd-full {auc(s_out[:,0], y_te[:,0]):.3f}")
