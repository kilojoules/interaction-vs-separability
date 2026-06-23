"""Stress-test the phase faithfulness ceiling.

The phase pinwheel country readout decomposed by the standard APD recipe plateaued
at country AUC ~0.68-0.90 (and got *worse* at higher C). We claimed this is a real
ceiling unmovable by steps/minimality. Throw everything at it: seeds, steps,
learning rate, C, initialization, unit-norm on/off, and a PURE-RECONSTRUCTION mode
(param-match + out-recon only, no topk / no schatten) which — with rank-64
components — *should* be able to represent any weight matrix. If even one config
reaches ~0.98, the ceiling is an artifact of the regularized objective, not the
representability of the phase code.

Faithfulness measured = country AUC of the FULL SPD model (all C components, no
topk) + output reconstruction error vs the target readout.

Run (apd env): .../bin/python experiments/e7_phase_faithfulness_stress.py --C 40 ...
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import spd_readout as R, spd_analyze as A
from e1f_country_only import country_target, country_effect
from spd.run_spd import Config, TMSTaskConfig, optimize
from spd.utils import set_seed
from spd.module_utils import init_param_
BDC = Path("/Users/julianquick/bdc/bluedot-tais-puzzle")
REG = {"phase": BDC/"model_pinwheel.pt", "poly": ROOT/"models"/"nondedicated_clean_cubic.pt",
       "xor": ROOT/"models"/"order3_xor.pt", "gate": BDC/"model.pt"}


def country_label(model, te):
    if model == "poly":
        return np.load(ROOT/"results"/"task2"/"cubic_labels.npz")["c_te"]
    if model == "xor":
        y = te["labels"]; return (y[:, 0] ^ y[:, 2] ^ y[:, 6]).astype(np.int64)
    return te["labels"][:, A.CI]   # phase / gate (model.pt) / pinwheel: real country


def none_or(v, cast=float):
    return None if str(v).lower() == "none" else cast(v)


def run(args):
    set_seed(args.seed)
    CKPT = REG[args.model]
    tr = np.load(BDC/"cache/train.npz"); te = np.load(BDC/"cache/test.npz")
    L_tr, _ = R.compute_L(CKPT, tr["emb"]); L_te, _ = R.compute_L(CKPT, te["emb"])
    yc = country_label(args.model, te)
    m = none_or(args.m, int)
    if args.multi:
        target = R.load_readout_target(CKPT); d_out = 8
    else:
        target = country_target(CKPT); d_out = 1
    spd = R.ReadoutSPD(C=args.C, m=m, d_out=d_out, n_instances=1)
    # re-initialize components with requested scale/type
    init_param_(spd.mlp_in.A, scale=args.init_scale, init_type=args.init_type)
    init_param_(spd.mlp_in.B, scale=args.init_scale, init_type=args.init_type)
    init_param_(spd.mlp_out.A, scale=args.init_scale, init_type=args.init_type)
    init_param_(spd.mlp_out.B, scale=args.init_scale, init_type=args.init_type)
    with torch.no_grad():
        spd.bias1[:] = target.bias1.clone(); spd.bias2[:] = target.bias2.clone()
    spd.bias1.requires_grad = False; spd.bias2.requires_grad = False

    X = torch.tensor(L_tr, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(X, torch.zeros(len(X))), batch_size=256, shuffle=True)
    topk = none_or(args.topk); schatten = none_or(args.schatten)
    tc = TMSTaskConfig(feature_probability=0.05, train_bias=False, bias_val=0.0, pretrained_model_path="d")
    cfg = Config(
        batch_size=256, steps=args.steps, print_freq=args.steps, image_freq=None, save_freq=None,
        lr=args.lr, C=args.C, m=m,
        topk=topk, batch_topk=(topk is not None), topk_recon_coeff=(1.0 if topk is not None else None),
        unit_norm_matrices=bool(args.unit_norm),
        param_match_coeff=args.pm, out_recon_coeff=(None if topk is not None else 1.0),
        act_recon_coeff=1.0,
        schatten_coeff=schatten, schatten_pnorm=(0.9 if schatten is not None else None),
        lr_schedule=args.sched, attribution_type="gradient", task_config=tc)
    optimize(model=spd, config=cfg, device="cpu", dataloader=dl, target_model=target,
             param_names=["mlp_in", "mlp_out"], out_dir=None, plot_results_fn=None)

    Xte = torch.tensor(L_te, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        t = target(Xte).squeeze(-1).reshape(len(yc), -1)
        s = spd(Xte).squeeze(-1).reshape(len(yc), -1)
        cidx = A.CI if args.multi else 0
        t_c = t[:, cidx].numpy(); s_c = s[:, cidx].numpy()
    recon_rel = float(((t.numpy() - s.numpy()) ** 2).mean() / (t.numpy().var() + 1e-9))
    rep = {"country_auc_target": A.auc(t_c, yc), "country_auc_spd": A.auc(s_c, yc),
           "recon_rel": recon_rel, "config": vars(args)}
    # faithfulness-vs-parsimony: effective # components carrying country (country-only only)
    if not args.multi:
        e, _ = country_effect(spd, Xte)
        rep["PR_effective_components"] = float((e.sum() ** 2) / ((e ** 2).sum() + 1e-12))
        rep["n_eff_above_5pct"] = int((e > 0.05 * e.max()).sum())
        order = np.argsort(-e); aucs = []
        for k in range(1, args.C + 1):
            mask = torch.zeros(Xte.shape[0], 1, args.C, dtype=torch.bool); mask[:, :, order[:k]] = True
            with torch.no_grad():
                z = spd(Xte, topk_mask=mask).squeeze(-1).squeeze(-1).numpy()
            aucs.append(A.auc(z, yc))
        full = aucs[-1]
        rep["recon_k95"] = int(next((k for k, a in enumerate(aucs, 1) if a >= 0.95 * full), args.C))
        rep["recon_curve"] = [float(a) for a in aucs]   # AUC vs #components kept (the Pareto frontier)
        # Pareto area = normalized faithfulness deficit when constrained to few components
        rep["pareto_area"] = float(np.mean([full - a for a in aucs]) / (full + 1e-9))
    tag = (f"{args.model}_C{args.C}_st{args.steps}_lr{args.lr}_sd{args.seed}_un{args.unit_norm}_is{args.init_scale}"
           f"_{args.init_type[:3]}_pm{args.pm}_sc{args.schatten}_tk{args.topk}_{args.sched}"
           f"_m{args.m}_{'multi' if args.multi else 'co'}")
    out = ROOT/"results"/"stress"; out.mkdir(parents=True, exist_ok=True)
    (out/f"{tag}.json").write_text(json.dumps(rep, indent=2))
    print(f"[{tag}] spdAUC {rep['country_auc_spd']:.3f}  reconRel {recon_rel:.4f}  (tgt {rep['country_auc_target']:.3f})")
    return rep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=40); p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sched", default="constant"); p.add_argument("--unit_norm", type=int, default=0)
    p.add_argument("--init_scale", type=float, default=1.0); p.add_argument("--init_type", default="xavier_normal")
    p.add_argument("--pm", type=float, default=1.0); p.add_argument("--schatten", default="none")
    p.add_argument("--topk", default="none"); p.add_argument("--m", default="none")
    p.add_argument("--multi", type=int, default=0)
    p.add_argument("--model", default="phase", choices=["phase", "poly", "xor", "gate"])
    run(p.parse_args())
