"""
檔案功能
--------
使用者 2026-11-25 的新問題:對一個已經訓練好(誠實、不惡意)的模型,如果回報的是「一個 batch(n 筆樣本)的平均
梯度」而不是單一樣本,把候選池裡所有「大小為 n 的子集合」都當成一個候選(不是把每個樣本當候選)：
    1. 這樣的候選總數是多少？C(n_aux, n)，組合數爆炸多快？
    2. 任兩個這樣的「batch 候選」的梯度,有多難區分？

關鍵理論點(跟平均的集中效應有關,不是 DP 雜訊造成的,DP 雜訊是另一層、疊加上去的):
    batch 梯度 = (1/n) Σ_{i∈batch} gᵢ(各自的乾淨梯度)
    兩個只差一個成員的 batch(交換一個人):差距 = (g_swap_in - g_swap_out) / n          → 隨 n 變小(稀釋)
    但同時,不同 batch 的梯度本身(不管差幾個成員)因為平均而往候選池的平均梯度集中(類似中央極限定理)：
    batch 梯度的變異數 ≈ Var(單一樣本梯度) × (1 - (n-1)/(n_aux-1)) / n（有限總體校正,無放回抽樣）
    所以「候選數量」跟「候選之間的可分辨距離」是同時發生的兩個效應，隨 n 同時變化：候選數爆炸式增加
    （C(n_aux,n)），但它們擠進去的空間同時在收縮——這才是 batch size 保護隱私的真正機制，不只是單樣本
    敏感度的縮放。

這裡只做幾何/統計量測(候選梯度的分布如何隨 n 變化),不做完整的 posterior-mean 攻擊(候選數量在 n 稍大
就超過可窮舉範圍,SPM 式的攻擊在這裡不現實——這正是我們想驗證的「搜尋空間爆炸」那一半)。

使用方式
--------
    python -m analysis.p12_batch_distinguishability --n-aux 60 --batch-sizes 1 2 4 8 16 32 --n-samples 500
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from scipy.special import comb

from trainers.fl_client import fedsgd_gradient
from util.registry import build

MODEL_SPECS = {
    "linear": {"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": [], "activation": "relu", "in_dim": 533, "out_dim": 2}},
    "itransformer": {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="linear", choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--n-aux", type=int, default=60)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n-samples", type=int, default=500, help="每個 batch size 隨機抽幾個 batch 來估計分布(C(n_aux,n) 通常太大無法窮舉)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="p12_batch_distinguishability")
    a = ap.parse_args()

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(ds.x))[:a.n_aux]
    x_pool = torch.as_tensor(ds.x[idx], dtype=torch.float32)
    y_pool = torch.as_tensor(ds.y[idx], dtype=torch.long)

    torch.manual_seed(42)
    model = build(MODEL_SPECS[a.model]); model.eval()
    loss_fn = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})

    print(f"[{a.model}] 算候選池裡每個人自己的乾淨梯度 ({a.n_aux} 筆)...", flush=True)
    grads = []
    for i in range(a.n_aux):
        g = fedsgd_gradient(model, loss_fn, x_pool[i:i + 1], y_pool[i:i + 1], micro_batch=1)
        grads.append(torch.cat([t.reshape(-1) for t in g]).numpy())
    G = np.stack(grads)                                               # (n_aux, d)：每個人自己的梯度
    overall_var = float(np.trace(np.cov(G.T)))                        # 單一樣本梯度的總變異數(所有維度加總)
    print(f"d={G.shape[1]}  單樣本梯度總變異數(trace Cov)={overall_var:.6g}", flush=True)

    rows = []
    for n in a.batch_sizes:
        n_cand = comb(a.n_aux, n, exact=False)
        rng_b = np.random.default_rng(a.seed + n)
        batch_grads = np.stack([G[rng_b.choice(a.n_aux, size=n, replace=False)].mean(0) for _ in range(a.n_samples)])
        spread_var = float(np.trace(np.cov(batch_grads.T)))            # 這組 batch 梯度彼此之間的變異數
        theory_var = overall_var * (1 - (n - 1) / (a.n_aux - 1)) / n   # 無放回抽樣的有限總體校正公式

        # [一個成員的差距]：隨機抽一個 batch，交換其中一個成員，量差距
        swap_dists = []
        for _ in range(min(200, a.n_samples)):
            members = rng_b.choice(a.n_aux, size=n, replace=False)
            out_pool = np.setdiff1d(np.arange(a.n_aux), members)
            if len(out_pool) == 0:
                continue
            swap_in = rng_b.choice(out_pool)
            swap_out = rng_b.choice(members)
            g_before = G[members].mean(0)
            g_after = g_before + (G[swap_in] - G[swap_out]) / n
            swap_dists.append(float(np.linalg.norm(g_after - g_before)))
        swap_dist_mean = float(np.mean(swap_dists)) if swap_dists else float("nan")

        # [隨機兩個 batch 的距離]，當作「完全不相關的候選」的對照基準
        rand_dists = np.linalg.norm(batch_grads[:, None, :] - batch_grads[None, :, :], axis=-1)
        rand_dist_mean = float(rand_dists[np.triu_indices(len(batch_grads), k=1)].mean())

        row = {"n": n, "n_candidates": n_cand, "spread_var_empirical": spread_var, "spread_var_theory": theory_var,
               "swap_one_member_dist": swap_dist_mean, "random_pair_dist": rand_dist_mean,
               "swap_vs_random_ratio": swap_dist_mean / rand_dist_mean if rand_dist_mean > 0 else float("nan")}
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        rows.append(row)

    d = os.path.join("runs", "results", a.out)
    os.makedirs(d, exist_ok=True)
    json.dump({"args": vars(a), "overall_var": overall_var, "rows": rows}, open(os.path.join(d, f"result_seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
