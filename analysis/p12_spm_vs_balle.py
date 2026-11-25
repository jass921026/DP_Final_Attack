"""
檔案功能
--------
把搬進 `attacks/prior_aware_spm.py` 的 Hayes et al. 2023 SPM 攻擊（prior-aware,單次全批次釋出,q=1,T=1）套到
P12 上,跟同一組 clip/sigma 下的 Balle weight-ReCoNN(`analysis/p12_rero_pilot.py`+`p12_reconn.py`)、Imprint
closed-form(`analysis/p12_model_sweep.py`)放在一起比——三個攻擊第一次在同一個資料集、同一組 sigma 清單下
並排。

跟 MNIST 重現（`analysis/mnist_spm_repro.py`）的差異：這裡不用官方的 ρ 正規化（那需要他們自己的理論界估計出
的 tr_Sigma,我們沒有 P12 版本),改用專案一路用下來的 `rank_metrics`（NN-oracle 候選池 + 百分位排名),才能
跟 Imprint/Balle 的數字直接放一起比。sigma 清單跟 clip 沿用專案標準(0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
clip=1.0),候選池/目標切分沿用 `p12_model_sweep.py` 的慣例(bin_aux+aux 當候選池、test 當目標,彼此不重疊)。

SPM 攻擊本身不訓練模型：候選池/目標都是 P12 病人的原始向量(標準化後),威脅模型是「如果一個機制誠實地把
(中心化、裁剪過的)目標病人向量 + 高斯雜訊直接釋出,攻擊者最多能從候選池猜多準」——跟 Imprint(插入惡意架構
取梯度結構)、Balle(訓練上千個 shadow 模型學參數→資料的映射)都不是同一種攻擊成本,是這次比較裡最輕量的
一個，見檔頭 `attacks/prior_aware_spm.py` 的說明。

使用方式
--------
    python -m analysis.p12_spm_vs_balle --sigmas 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2
"""
import argparse
import json
import os

import numpy as np
import torch

from attacks.prior_aware_spm import clip_candidates, gen_release_single, gen_release_sub, solve_sigma, spm_attack
from util.registry import build

DATA_SHAPE = (13 * 41,)


def rank_metrics(xhat: np.ndarray, x_true: np.ndarray, pool: np.ndarray):
    """功能：同 analysis/p12_wg_reconn.py 的 rank_metrics（NN-oracle 池子 + 百分位排名）,見該檔 docstring。"""
    d_pool = (xhat ** 2).sum(1, keepdims=True) + (pool ** 2).sum(1)[None, :] - 2 * xhat @ pool.T
    d_true = ((xhat - x_true) ** 2).sum(1)
    pct = ((d_pool < d_true[:, None]).sum(1) + 0.5 * (d_pool == d_true[:, None]).sum(1)) / len(pool)
    roll = np.roll(np.arange(len(x_true)), 1)
    d_null = ((xhat[roll] - x_true) ** 2).sum(1)
    pct_null = (d_pool[roll] < d_null[:, None]).sum(1) / len(pool)
    return float(1 - pct.mean()), float(1 - pct_null.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aux", type=int, default=1500); ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-bin-aux", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--steps", type=int, default=1, help=">1：額外跑 T 步 q=1 子抽樣釋出(跟 Balle 的 DP-GD T 步設定對等)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="p12_spm_vs_balle")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ds.x))
    i_bin_aux = perm[:a.n_bin_aux]
    i_aux = perm[a.n_bin_aux:a.n_bin_aux + a.n_aux]
    i_test = perm[a.n_bin_aux + a.n_aux:a.n_bin_aux + a.n_aux + a.n_test]
    cand_x = np.concatenate([ds.x[i_bin_aux], ds.x[i_aux]])          # 候選池（NN-oracle 用的同一個池）
    x_test = ds.x[i_test]

    mu = cand_x.mean(0).astype(np.float64)
    cand_u = torch.as_tensor((cand_x.astype(np.float64) - mu[None]).astype(np.float32), device=a.device)
    target_u = torch.as_tensor((x_test.astype(np.float64) - mu[None]).astype(np.float32), device=a.device)
    cand_c, cand_a_, cand_norm2 = clip_candidates(cand_u, a.clip)
    target_c, _, _ = clip_candidates(target_u, a.clip)
    mu_t = torch.as_tensor(mu, dtype=torch.float32)

    rows = []
    for sigma in a.sigmas:
        noise_std = max(sigma * a.clip, 1e-8)                       # sigma=0：極小雜訊地板，避免除以 0
        G = gen_release_single(target_c, noise_std, seed=1000 + int(sigma * 1e6))[:, None, :]
        u_mean, u_map = spm_attack(G, cand_c, cand_norm2, cand_a_, noise_std)
        xhat = (mu_t[None] + u_mean).detach().cpu().numpy()
        rank_score, rank_null = rank_metrics(xhat, x_test, cand_x)
        xhat_map = (mu_t[None] + u_map).detach().cpu().numpy()
        rank_map, _ = rank_metrics(xhat_map, x_test, cand_x)
        row = {"sigma": sigma, "spm_rank_score": rank_score, "spm_rank_null": rank_null, "spm_rank_map": rank_map}

        if a.steps > 1:
            # [T 步、q=1 的子抽樣釋出，participation 永遠是 1 —— 跟 Balle 的 DP-GD 全批次(每步都用到目標)、
            # 每步各加一次獨立雜訊的設定對等，同一個 raw sigma 直接比，不用再換算 eps]
            n = target_c.shape[0]
            GT = torch.stack([gen_release_sub(target_c[ti], noise_std, seed=2000 + ti, T=a.steps, q=1.0)
                              for ti in range(n)], 0)
            um_T, umap_T = spm_attack(GT, cand_c, cand_norm2, cand_a_, noise_std, q=1.0)
            xhat_T = (mu_t[None] + um_T).detach().cpu().numpy()
            rank_T, _ = rank_metrics(xhat_T, x_test, cand_x)
            row[f"spm_rank_score_T{a.steps}"] = rank_T

        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
        rows.append(row)

    d = os.path.join("runs", "results", a.out)
    os.makedirs(d, exist_ok=True)
    json.dump({"args": vars(a), "rows": rows}, open(os.path.join(d, f"result_seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
