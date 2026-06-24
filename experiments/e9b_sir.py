"""Find the country (U,V)=(φ₁,φ₂) directions with Sliced Inverse Regression (SIR)
instead of the gradient active subspace, and compare. SIR estimates the
effective-dimension-reduction (EDR) subspace from the INVERSE regression E[L | y]:
standardize L, slice the response, take the covariance of the slice means, eigendecompose.

We slice on the model's CONTINUOUS country logit (a binary response gives only H−1=1
direction). SIR has a known blind spot: it misses directions along which E[L | y] is
SYMMETRIC — the phase pinwheel's alternating sectors are the textbook case — so this is
also a test of whether the phase geometry defeats SIR the way it defeats first-order
attribution.

Run (base env): USE_TF=0 python experiments/e9b_sir.py
"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import separability as S
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")

def real(y): return y[:, S.CI]
def cube(y): return np.load(ROOT/"results"/"task2"/"cubic_labels.npz")["c_te"]
def xor(y):  return (y[:, 0] ^ y[:, 2] ^ y[:, 6]).astype(np.int64)
MODELS = [("linear", BDC/"model_vanilla.pt", real), ("gate", BDC/"model.pt", real),
          ("phase", BDC/"model_pinwheel.pt", real),
          ("poly", ROOT/"models"/"nondedicated_clean_cubic.pt", cube),
          ("xor", ROOT/"models"/"order3_xor.pt", xor)]


def sir(L, z, H=20, reg=1e-3):
    """SIR. L:(N,p), z:(N,) continuous response. Returns EDR dirs (p,k), eigvals, variates."""
    N, p = L.shape
    Lc = L - L.mean(0)
    Sig = np.cov(Lc, rowvar=False) + reg * np.eye(p)
    w, V = np.linalg.eigh(Sig)
    isqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T          # Σ^{-1/2}
    Z = Lc @ isqrt
    order = np.argsort(z); M = np.zeros((p, p))
    for sl in np.array_split(order, H):                  # equal-count slices
        mh = Z[sl].mean(0); M += (len(sl) / N) * np.outer(mh, mh)
    ev, evec = np.linalg.eigh(M)
    i = np.argsort(ev)[::-1]; ev, evec = ev[i], evec[:, i]
    beta = isqrt @ evec                                  # directions in original L space
    return beta, ev, Z @ evec                            # dirs, eigenvalues, SIR variates


def principal_angles(A, B):
    """Largest/smallest principal angle (deg) between the column spaces of A,B (p×2)."""
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))                      # [θ_min, θ_max]


def main():
    te = np.load(BDC/"cache/test.npz"); emb = te["emb"].astype(np.float32); y = te["labels"]
    out = ROOT/"results"/"geometry"; out.mkdir(parents=True, exist_ok=True)
    print(f"{'code':<7}{'SIR eig1':>9}{'eig2':>8}{'eig3':>8}  {'eig2/eig1':>9}  {'∠(SIR,active) deg':>18}")
    for key, ckpt, labfn in MODELS:
        net = S.load(ckpt)
        with torch.no_grad():
            L = net.layers[:6](torch.from_numpy(emb)).numpy().astype(np.float64)
            z = net.layers[6:](torch.from_numpy(L.astype(np.float32))).numpy()[:, S.CI]  # cont. logit
        beta, ev, var = sir(L, z, H=20)
        phi_sir = beta[:, :2]
        proj = L @ phi_sir                               # project onto SIR directions
        lab = labfn(y)
        # compare to the active subspace (from e9, recompute the gradient-cov top-2)
        R = S.readers(net, L)[S.CI]; Cact = R.T @ R / len(R)
        eva, eveca = np.linalg.eigh(Cact); phi_act = eveca[:, ::-1][:, :2]
        ang = principal_angles(phi_sir, phi_act)
        np.savez(out/f"{key}_sir.npz", proj=proj, label=lab, eig=ev[:8],
                 eig_ratio=(ev[:8]/ev[0]), angle_to_active=ang)
        print(f"{key:<7}{ev[0]:>9.3f}{ev[1]:>8.3f}{ev[2]:>8.3f}  {ev[1]/ev[0]:>9.2f}  "
              f"{ang[0]:>8.1f},{ang[1]:>7.1f}")


if __name__ == "__main__":
    main()
