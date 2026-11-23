"""
檔案功能
--------
P12 + iTransformer 上 imprint 攻擊的批次版本：延伸 p12_imprint_b1（batch=1）到任意批次大小 n，
跟 imprint_dp_clip1*（ImageNet，n=64）走同一套慣例：客戶端整批裁剪到範數 <= clip 再平均，雜訊
scale = sigma*clip/n（n 愈大，同一個 sigma 對應的絕對雜訊愈小，但 epsilon 本身與 n 無關，只看 sigma）。
伺服器端用 attacks.imprint.reconstruct（cumulative 差分 + select_hits）還原，再用「最接近的真實樣本」
配對算 MSE，並與 Balle 分析一致地算排名分數（1 - 目標在池中的百分位）。
使用方式
--------
    python -m analysis.p12_imprint_batch [--batches 1 4 16 64] [--bins 128] [--clip 1.0]
        [--sigmas 0 1e-4 1e-3 1e-2] [--out p12_imprint_batch]
"""
import argparse
import json
import os

import numpy as np
import torch

from attacks.imprint import ImprintBlock, add_gradient_noise, compute_empirical_bins, reconstruct
from trainers.fl_client import dp_clipped_gradient
from util.gdp import gdp_epsilon
from util.registry import build

ITR_SPEC = {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}}
CE_SPEC = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}
DATA_SHAPE = (13 * 41,)


def best_match_mse(rec: torch.Tensor, true_x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """功能：每個真實樣本各自找「所有重建中最接近的」（同 attacks.imprint.best_match_psnr 的作法，非互斥配對）；
    沒有任何重建（n_hits=0）時整批回退到先驗。"""
    if rec.shape[0] == 0:
        return (prior[None] - true_x).pow(2).mean(1)
    d = ((rec[:, None] - true_x[None]) ** 2).mean(-1)      # (n_rec, n_true)
    return d.min(0).values


def rank_metrics(rec, true_x, pool):
    """同 analysis/balle_rank.py 的排名分數：1 - 目標在池中的百分位（越大越準），對照組用錯位配對。"""
    d_pool = ((rec[:, None] - pool[None]) ** 2).sum(-1)              # (n_rec, n_pool)
    d_true = ((rec - true_x) ** 2).sum(-1)
    pct = ((d_pool < d_true[:, None]).sum(1).float() + 0.5 * (d_pool == d_true[:, None]).sum(1).float()) / pool.shape[0]
    perm = np.roll(np.arange(len(true_x)), 1)
    d_null = ((rec - true_x[perm]) ** 2).sum(-1)
    pct_null = ((d_pool < d_null[:, None]).sum(1).float()) / pool.shape[0]
    return float(1 - pct.mean()), float(1 - pct_null.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--bins", type=int, default=128); ap.add_argument("--linfunc", default="avg")
    ap.add_argument("--gain", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--thr-sd", type=float, default=3.0); ap.add_argument("--n-aux", type=int, default=1000)
    ap.add_argument("--n-rounds", type=int, default=5, help="每個 (batch,sigma) 重複幾輪不同的批次取樣")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm_all = rng.permutation(len(ds.x))
    i_aux, i_rank_pool, i_pool = perm_all[:a.n_aux], perm_all[a.n_aux:a.n_aux + 2000], perm_all[a.n_aux + 2000:]
    aux_x = torch.as_tensor(ds.x[i_aux], dtype=torch.float32)
    pool_x = torch.as_tensor(ds.x[i_rank_pool], dtype=torch.float32)       # 排名用的池子，與攻擊抽樣的 i_pool 不重疊
    prior = aux_x.mean(0)

    bins = compute_empirical_bins(aux_x, DATA_SHAPE, a.bins, linfunc=a.linfunc)
    torch.manual_seed(42)
    net = build(ITR_SPEC)
    block = ImprintBlock(DATA_SHAPE, a.bins, connection="linear", gain=a.gain, linfunc=a.linfunc, empirical_bins=bins)
    model = torch.nn.Sequential(block, net)
    model.eval()
    loss_fn = build(CE_SPEC)

    rows = []
    for n in a.batches:
        for sigma in a.sigmas:
            mse_all: list = []
            rank_all, rank_null_all, hits_all, clip_fracs = [], [], [], []
            for r in range(a.n_rounds):
                idx = rng.choice(i_pool, n, replace=False)
                x = torch.as_tensor(ds.x[idx], dtype=torch.float32)
                y = torch.as_tensor(ds.y[idx], dtype=torch.long)
                clean, g_norms = dp_clipped_gradient(model, loss_fn, x, y, a.clip)
                clip_fracs.append(float((g_norms > a.clip).float().mean()))
                scale = sigma * a.clip / n
                grads = add_gradient_noise(clean, scale, seed=a.seed * 1000 + r, distribution="gaussian") if sigma > 0 else clean
                thr = a.thr_sd * 2 * scale if sigma > 0 else 0.0
                rec, n_hits = reconstruct(grads, n, DATA_SHAPE, bias_threshold=thr, reduction="weight", structure=block.structure,
                                          clip_range=(-20.0, 20.0))
                hits_all.append(n_hits)
                per_sample = best_match_mse(rec, x, prior)        # 同 best_match_psnr 的作法：每個樣本各自找最接近的重建，非互斥配對
                mse_all.append(per_sample)
                best_match = rec[((rec[:, None] - x[None]) ** 2).mean(-1).argmin(0)] if rec.shape[0] else prior[None].repeat(n, 1)
                if rec.shape[0] >= 2:
                    ra, rn = rank_metrics(best_match, x, pool_x)     # 池子不含目標本身，避免污染排名
                    rank_all.append(ra); rank_null_all.append(rn)
            mse_cat = torch.cat(mse_all)
            prior_mse = float(((prior[None] - torch.as_tensor(ds.x[rng.choice(i_pool, 300, replace=False)], dtype=torch.float32)) ** 2).mean())
            eps = gdp_epsilon(1.0 / sigma, 1e-5) if sigma > 0 else float("inf")
            # mean 不穩健（少數碰撞/補零的重建可以把個別樣本的 MSE 拉得很大，見 attacks.imprint.best_match_psnr 的說明），
            # 所以同時報 median 與 frac_perfect（MSE < 0.05，量級取自「先驗 MSE 的 1/10」，不是像素攻擊的絕對閾值）
            row = {"n": n, "sigma": sigma, "epsilon": eps, "clip": a.clip, "frac_clipped": float(np.mean(clip_fracs)),
                   "mean_n_hits": float(np.mean(hits_all)), "mse_mean": float(mse_cat.mean()), "mse_median": float(mse_cat.median()),
                   "frac_perfect": float((mse_cat < 0.05).float().mean()), "prior_mse": prior_mse,
                   "mse_ratio_median": float(mse_cat.median() / prior_mse),
                   "rank_score": float(np.mean(rank_all)) if rank_all else None,
                   "rank_score_null": float(np.mean(rank_null_all)) if rank_null_all else None}
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
            rows.append(row)

    if a.out:
        d = os.path.join("runs", "results", a.out)
        os.makedirs(d, exist_ok=True)
        json.dump({"args": vars(a), "rows": rows}, open(os.path.join(d, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
