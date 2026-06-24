"""Active-subspace geometry of each country encoding: project L onto the top-2
eigenvectors of the country-logit gradient covariance C = E[∂z/∂L ∂z/∂L^T] and
scatter, colored by the country label. This is the 'shadow plot' that shows WHY
phase is special (it shows the pinwheel sectors), next to the others.

Run (base env): USE_TF=0 python experiments/e9_geometry.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import separability as S
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")

# (key, ckpt, label_fn)
def real(y): return y[:, S.CI]
def cube(y): return np.load(ROOT/"results"/"task2"/"cubic_labels.npz")["c_te"]
def xor(y):  return (y[:, 0] ^ y[:, 2] ^ y[:, 6]).astype(np.int64)
MODELS = [
    ("linear",  BDC/"model_vanilla.pt",                  real),
    ("gate",    BDC/"model.pt",                          real),
    ("phase",   BDC/"model_pinwheel.pt",                 real),
    ("poly",    ROOT/"models"/"nondedicated_clean_cubic.pt", cube),
    ("xor",     ROOT/"models"/"order3_xor.pt",           xor),
]


def main():
    te = np.load(BDC/"cache/test.npz"); emb = te["emb"].astype(np.float32); y = te["labels"]
    out = ROOT/"results"/"geometry"; out.mkdir(parents=True, exist_ok=True)
    for key, ckpt, labfn in MODELS:
        net = S.load(ckpt)
        with torch.no_grad():
            L = net.layers[:6](torch.from_numpy(emb)).numpy().astype(np.float64)
        R = S.readers(net, L)[S.CI]                       # (N,64) per-input country reader
        C = R.T @ R / len(R)                              # gradient covariance (active subspace)
        ev, evec = np.linalg.eigh(C)
        phi = evec[:, ::-1][:, :2]                        # top-2 eigenvectors
        proj = L @ phi                                    # (N,2)
        lab = labfn(y)
        np.savez(out/f"{key}.npz", proj=proj, label=lab,
                 eig=ev[::-1][:8], var_frac=(ev[::-1]/ev.sum())[:8])
        cum = np.cumsum(ev[::-1]/ev.sum())
        print(f"{key:<7} top-3 var-frac {np.round((ev[::-1]/ev.sum())[:3],3)} | 99%-dim "
              f"{int(np.searchsorted(cum,0.99)+1)} | label base-rate {lab.mean():.2f}")


if __name__ == "__main__":
    main()
