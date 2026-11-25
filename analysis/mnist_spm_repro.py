"""
檔案功能
--------
用 `attacks/prior_aware_spm.py`（搬進本專案的 Hayes et al. 2023 SPM 攻擊）在 MNIST 上重現
`h-0-0/info_reconstruction_bound`（"From Pixels to Privacy-Sensitive Features"）repo 裡 `tab:attack_results` 的
MNIST pixel、q=1(全批次)那一列數字（ρ at gamma=0.72/2.89/8.01/72.13 應為 0.987/0.948/0.848/0.620），驗證移植
正確。切分跟常數直接沿用該 repo 已經算好、隨 repo 一起發布的檔案（不是我們自己重新派生,是引用同一份資料）：
    - 候選/目標切分：<repo>/results/attack/splits/mnist_seed42.npz（pool = MNIST train+test 串接,
      np.random.default_rng(42).permutation 切出 D_rest=1000/D_shadow=58000/D_targets=1000）
    - 正規化常數 tr_Sigma_raw：<repo>/results/bounds/pixel_mnist/entropy_bound.json（他們自己的理論界估計出來
      的量,拿來把 MSE 換算成跟論文同單位的 rho = sqrt(MSE) / (sqrt(tr_Sigma_raw) / 255)）
    - clip C = 0.3（跟 `scripts/tightness_spm.py` 的預設一致）

使用方式
--------
    python -m analysis.mnist_spm_repro --repo-dir ../hayes2023_reconstruction
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from torchvision import datasets

from attacks.prior_aware_spm import clip_candidates, gen_release_single, nm_of_gamma, spm_attack

GREY = 255.0
C = 0.3
GAMMAS = [0.72, 2.89, 8.01, 72.13]
EXPECTED_RHO = [0.987, 0.948, 0.848, 0.620]        # repo 的 tab:attack_results，MNIST pixel，q=1


def load_mnist_pool(data_dir: str) -> np.ndarray:
    """功能：MNIST train+test 串接，攤平成 (N, 784) float32，[0,1]（跟 repo 的 load_pool 同一種做法:ToTensor 等價）。"""
    imgs = []
    for train in (True, False):
        ds = datasets.MNIST(data_dir, train=train, download=True)
        imgs.append(np.asarray(ds.data).reshape(len(ds), -1).astype(np.float32) / 255.0)
    return np.concatenate(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="../hayes2023_reconstruction", help="h-0-0/info_reconstruction_bound 的本機路徑")
    ap.add_argument("--data-dir", default="data_cache/mnist_raw")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    split_path = os.path.join(a.repo_dir, "results", "attack", "splits", "mnist_seed42.npz")
    bound_path = os.path.join(a.repo_dir, "results", "bounds", "pixel_mnist", "entropy_bound.json")
    split = np.load(split_path)
    tr_sigma_raw = json.load(open(bound_path))["tr_Sigma_raw"]
    denom_a = math.sqrt(tr_sigma_raw) / GREY

    pool = load_mnist_pool(a.data_dir)
    shadow = pool[split["idx_shadow"]]                      # 候選池 D_shadow（58000 張）
    targets = pool[split["idx_targets"]][:100]               # 攻擊目標（repo 的 n_targets=100）

    mu = shadow.mean(0).astype(np.float64)
    shadow_u = torch.as_tensor((shadow.astype(np.float64) - mu[None]).astype(np.float32), device=a.device)
    target_u = torch.as_tensor((targets.astype(np.float64) - mu[None]).astype(np.float32), device=a.device)
    cand_c, cand_a, cand_norm2 = clip_candidates(shadow_u, C)
    target_c, _, _ = clip_candidates(target_u, C)

    print(f"{'gamma':>7} {'rho':>8} {'expected':>9} {'diff':>8}")
    for gamma, exp in zip(GAMMAS, EXPECTED_RHO):
        sigma = nm_of_gamma(gamma)
        seed = 42 * 7 + int(gamma * 1000)                   # 對應 repo 的 gen_release_single 種子公式
        G = gen_release_single(target_c, sigma * C, seed)[:, None, :]
        u_mean, _ = spm_attack(G, cand_c, cand_norm2, cand_a, sigma * C)
        est = (torch.as_tensor(mu, dtype=torch.float32)[None] + u_mean).clamp(0, 1)
        tgt = torch.as_tensor(targets, dtype=torch.float32)
        mse = ((est - tgt) ** 2).sum(1).mean().item()
        rho = math.sqrt(mse) / denom_a
        print(f"{gamma:>7.2f} {rho:>8.3f} {exp:>9.3f} {rho - exp:>+8.3f}")


if __name__ == "__main__":
    main()
