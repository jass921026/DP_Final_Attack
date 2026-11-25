"""
檔案功能
--------
P12 上 Balle 重建攻擊（ReRo）的可行性試驗（小規模）：
    D- = 500 筆固定資料；演算法 A = 確定性全批次 GD（共用初始化、無 dropout）；
    影子模型 A(D- ∪ {z_i})，z_i 來自對手的資料池；釋出模型 A(D- ∪ {z})，z 為測試目標。
    攻擊：參數標準化後以核（對偶）嶺迴歸預測標準化的 z（線性攻擊，比 ReCoNN 便宜，用來判斷「參數裡有沒有訊號」）。
    指標：MSE 與先驗（影子目標的平均）的 MSE 之比、目標在候選池中的排名分位（越小越好，無資訊約均勻）。
--model：原本只有 iTransformer；跟 analysis/p12_model_sweep.py 共用同一組架構（iTransformer/SUMMIT/MLP/GRU），
    SUMMIT 用 mode="values_mask"（D=13*82），其餘三個用 mode="values"（D=13*41）。輸出路徑的 tag 前面加
    `<model>_` 以免四個架構互相覆蓋。
使用方式
--------
    python -m analysis.p12_rero_pilot --model itransformer [--shadows 1000] [--targets 200] [--steps 100] [--lr 0.03] [--fixed 500] [--clip C --sigma S]
    輸出：analysis_out/p12_rero/<model>_<tag>/{shadow,released}.npz（可續跑）、summary.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.utils as U
from sklearn.metrics import roc_auc_score

from trainers.batched_gd import accuracy_of_flat_params, train_models_batched
from util.registry import build

MODEL_SPECS = {
    "itransformer": {"spec": {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}}, "mode": "values"},
    "summit": {"spec": {"module": "models.summit_cls", "type": "SUMMITClassifier", "kwargs": {}}, "mode": "values_mask"},
    "mlp": {"spec": {"module": "models.mlp", "type": "MLP", "kwargs": {"in_dim": 533, "out_dim": 2, "hidden": [256, 128], "activation": "relu"}}, "mode": "values"},
    "gru": {"spec": {"module": "models.gru_cls", "type": "GRUClassifier", "kwargs": {}}, "mode": "values"},
}
CE = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}


def get_data_shape(mode: str) -> int:
    return 13 * 41 * (2 if mode == "values_mask" else 1)


def train_chunks(spec, fx, fy, zx, zy, a, tag, out):
    dp = {"clip": a.clip, "sigma": a.sigma, "seed": a.dp_seed} if a.clip > 0 else None
    path = os.path.join(out, f"{tag}.npz")
    if os.path.exists(path):
        return np.load(path)["params"]
    parts, done = [], 0
    part = os.path.join(out, f"{tag}_partial.npy")
    if os.path.exists(part):
        parts = [np.load(part)]; done = len(parts[0])
    t0 = time.time()
    for s in range(done, len(zx), 100):
        P = train_models_batched(spec, 42, fx, fy, zx[s:s + 100], zy[s:s + 100], a.lr, 0.9, a.steps, build(CE), chunk_size=a.chunk, device=a.device, dp=dp)
        parts.append(P); np.save(part, np.concatenate(parts))
        print(f"{tag}: {s + len(P)}/{len(zx)}  {time.time() - t0:.0f}s", flush=True)
    P = np.concatenate(parts)
    np.savez(path, params=P)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="itransformer", choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--shadows", type=int, default=1000); ap.add_argument("--targets", type=int, default=200)
    ap.add_argument("--fixed", type=int, default=500); ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.03); ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--clip", type=float, default=0.0, help=">0 啟用 DP-GD（逐樣本裁剪 C）；sigma=雜訊倍數，0=只裁剪")
    ap.add_argument("--sigma", type=float, default=0.0); ap.add_argument("--dp-seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    SPEC, mode = MODEL_SPECS[a.model]["spec"], MODEL_SPECS[a.model]["mode"]
    tag = f"{a.model}_f{a.fixed}_s{a.shadows}_t{a.targets}_T{a.steps}_lr{a.lr}" + (f"_dpC{a.clip}_sig{a.sigma}" if a.clip > 0 else "")
    out = os.path.join("analysis_out", "p12_rero", tag); os.makedirs(out, exist_ok=True)
    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": mode, "flat": True}})
    perm = np.random.default_rng(0).permutation(len(ds.x))
    i_fix, i_sh = perm[:a.fixed], perm[a.fixed:a.fixed + a.shadows]
    i_te, i_ev = perm[a.fixed + a.shadows:a.fixed + a.shadows + a.targets], perm[-1000:]
    fx, fy, sx, sy, tx, ty = ds.x[i_fix], ds.y[i_fix], ds.x[i_sh], ds.y[i_sh], ds.x[i_te], ds.y[i_te]
    Ps = train_chunks(SPEC, fx, fy, sx, sy, a, "shadow", out)
    Pt = train_chunks(SPEC, fx, fy, tx, ty, a, "released", out)
    # 效用：釋出模型在評估集上的 AUROC
    m = build(SPEC); aucs = []
    for p in Pt[:10]:
        U.vector_to_parameters(torch.as_tensor(p), m.parameters()); m.eval()
        with torch.no_grad():
            aucs.append(roc_auc_score(ds.y[i_ev], torch.softmax(m(torch.as_tensor(ds.x[i_ev])), 1)[:, 1].numpy()))
    # [攻擊] 參數標準化 + 核嶺迴歸
    mu, sd = Ps.mean(0), Ps.std(0) + 1e-12
    Xs, Xt = (Ps - mu) / sd, (Pt - mu) / sd
    zmu = sx.mean(0)
    Ys = sx - zmu
    n = len(Xs); K = Xs @ Xs.T / Xs.shape[1]
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1, 10):                               # 以影子集內的留出（後 20%）選 lambda
        tr, va = np.arange(int(n * .8)), np.arange(int(n * .8), n)
        A = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Ys[tr])
        mse = ((K[np.ix_(va, tr)] @ A - Ys[va]) ** 2).mean()
        if best is None or mse < best[1]:
            best = (lam, mse)
    lam = best[0]
    A = np.linalg.solve(K + lam * np.eye(n), Ys)
    zhat = (Xt @ Xs.T / Xs.shape[1]) @ A + zmu
    mse = ((zhat - tx) ** 2).mean(1); prior = ((zmu - tx) ** 2).mean(1)
    pool = np.concatenate([fx, sx])
    d = ((zhat ** 2).sum(1)[:, None] + (pool ** 2).sum(1)[None] - 2 * zhat @ pool.T)
    dt = ((zhat - tx) ** 2).sum(1)
    pct = ((d < dt[:, None]).sum(1) + 0.5 * (d == dt[:, None]).sum(1)) / len(pool)
    perm2 = np.roll(np.arange(len(tx)), 1)
    dn = ((zhat[perm2] - tx) ** 2).sum(1)
    pn = ((d[perm2] < dn[:, None]).sum(1)) / len(pool)
    res = {"tag": tag, "lambda": lam, "released_auroc_mean": float(np.mean(aucs)), "mse_attack": float(mse.mean()), "mse_prior_mean": float(prior.mean()),
           "mse_ratio_attack_over_prior": float(mse.mean() / prior.mean()), "rank_auc": float(1 - pct.mean()), "rank_auc_null": float(1 - pn.mean()),
           "top5pct": float((pct <= .05).mean()), "top5pct_null": float((pn <= .05).mean()), "frac_better_than_prior": float((mse < prior).mean())}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
