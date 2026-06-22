"""Reproduce the order-1 control: a vanilla Head trained normally on the real 8
labels (country ends up linearly readable).  Run (base env):
    USE_TF=0 python models/train_vanilla.py
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from separability import Head
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")

def main(seed=0, epochs=80):
    tr = np.load(BDC / "cache/train.npz")
    X = torch.from_numpy(tr["emb"].astype(np.float32)); Y = torch.from_numpy(tr["labels"]).float()
    torch.manual_seed(seed); net = Head()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed); n = len(X)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, 256):
            idx = perm[i:i+256]; opt.zero_grad(); lf(net(X[idx]), Y[idx]).backward(); opt.step()
    net.eval()
    torch.save(net.state_dict(), Path(__file__).parent / "order1_vanilla.pt")
    print("saved order1_vanilla.pt")

if __name__ == "__main__":
    main()
