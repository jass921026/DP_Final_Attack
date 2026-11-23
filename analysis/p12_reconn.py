"""
檔案功能
--------
P12 + iTransformer 上以 ReCoNN（MLP 重建網路，Balle 2022 Sec. IV-B）取代 p12_rero_pilot 的線性核嶺迴歸。
讀 analysis_out/p12_rero/<tag>/{shadow,released}.npz（同一批影子/釋出模型），參數逐座標標準化後訓練 MLP
預測標準化的目標；影子集的後 20% 當留出集，保留留出 MSE 最低的 epoch。指標與線性版相同。
使用方式
--------
    python -m analysis.p12_reconn [--tag f500_s1000_t200_T100_lr0.03] [--hidden 1000 1000] [--epochs 200] [--subset all|<k>]
    --subset k：只取變異最大的 k 個座標（降維）。輸出 summary_reconn_<name>.json
"""
import argparse
import json
import os

import numpy as np
import torch

from util.registry import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f500_s1000_t200_T100_lr0.03"); ap.add_argument("--fixed", type=int, default=500)
    ap.add_argument("--shadows", type=int, default=1000); ap.add_argument("--targets", type=int, default=200)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1000, 1000]); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--subset", default="all"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    out = os.path.join("analysis_out", "p12_rero", a.tag)
    Ps, Pt = np.load(os.path.join(out, "shadow.npz"))["params"], np.load(os.path.join(out, "released.npz"))["params"]
    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    perm = np.random.default_rng(0).permutation(len(ds.x))             # 與 p12_rero_pilot 相同的切分
    i_fix, i_sh = perm[:a.fixed], perm[a.fixed:a.fixed + a.shadows]
    i_te = perm[a.fixed + a.shadows:a.fixed + a.shadows + a.targets]
    fx, sx, tx = ds.x[i_fix], ds.x[i_sh], ds.x[i_te]
    mu, sd = Ps.mean(0), Ps.std(0) + 1e-12
    Xs, Xt = ((Ps - mu) / sd).astype(np.float32), ((Pt - mu) / sd).astype(np.float32)
    if a.subset != "all":
        keep = np.argsort(-np.abs(np.corrcoef(Xs.T[:, :1].T, Xs.T)[0]) if False else -Ps.std(0))[:int(a.subset)]
        Xs, Xt = Xs[:, keep], Xt[:, keep]
    zmu = sx.mean(0); Ys = (sx - zmu).astype(np.float32)
    n = len(Xs); ntr = int(n * .8)
    torch.manual_seed(a.seed)
    phi = build({"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": a.hidden, "activation": "relu"}}, in_dim=Xs.shape[1], out_dim=Ys.shape[1]).to(a.device)
    opt = torch.optim.AdamW(phi.parameters(), lr=a.lr, weight_decay=a.wd)
    X, Y = torch.as_tensor(Xs, device=a.device), torch.as_tensor(Ys, device=a.device)
    best, best_state, hist = (1e9, 0), None, []
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(1, a.epochs + 1):
        phi.train(); idx = torch.randperm(ntr, generator=g)
        for s in range(0, ntr, a.bs):
            b = idx[s:s + a.bs].to(a.device)
            e = phi(X[b]) - Y[b]
            loss = e.abs().mean() + (e ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        phi.eval()
        with torch.no_grad():
            va = float(((phi(X[ntr:]) - Y[ntr:]) ** 2).mean())
        hist.append(va)
        if va < best[0]:
            best, best_state = (va, ep), {k: v.clone() for k, v in phi.state_dict().items()}
        if ep % 10 == 0:
            print(f"ep {ep} holdout mse {va:.4f} (best {best[0]:.4f}@{best[1]})  prior {float((Y[ntr:] ** 2).mean()):.4f}", flush=True)
    phi.load_state_dict(best_state); phi.eval()
    with torch.no_grad():
        zhat = phi(torch.as_tensor(Xt, device=a.device)).cpu().numpy() + zmu
    mse = ((zhat - tx) ** 2).mean(1); prior = ((zmu - tx) ** 2).mean(1)
    pool = np.concatenate([fx, sx])
    d = ((zhat ** 2).sum(1)[:, None] + (pool ** 2).sum(1)[None] - 2 * zhat @ pool.T)
    dt = ((zhat - tx) ** 2).sum(1)
    pct = ((d < dt[:, None]).sum(1) + 0.5 * (d == dt[:, None]).sum(1)) / len(pool)
    dn = ((zhat[np.roll(np.arange(len(tx)), 1)] - tx) ** 2).sum(1)
    pn = (d[np.roll(np.arange(len(tx)), 1)] < dn[:, None]).sum(1) / len(pool)
    res = {"args": vars(a), "best_epoch": best[1], "holdout_mse_best": best[0], "holdout_prior_mse": float((Y[ntr:] ** 2).mean()),
           "mse_attack": float(mse.mean()), "mse_prior_mean": float(prior.mean()), "mse_ratio_attack_over_prior": float(mse.mean() / prior.mean()),
           "rank_auc": float(1 - pct.mean()), "rank_auc_null": float(1 - pn.mean()), "top5pct": float((pct <= .05).mean()),
           "top5pct_null": float((pn <= .05).mean()), "frac_better_than_prior": float((mse < prior).mean())}
    name = f"{a.subset}_h{'-'.join(map(str, a.hidden))}_e{a.epochs}_lr{a.lr}_wd{a.wd}_seed{a.seed}"
    json.dump(res, open(os.path.join(out, f"summary_reconn_{name}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
