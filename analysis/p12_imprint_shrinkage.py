"""
檔案功能
--------
跟使用者討論「除非在 closed form 裡找到 prior 項，否則沒理由用它」之後的對照實驗：把
`attacks.imprint.invert_cumulative`（單一命中 bin 相除、其餘丟棄）換成 `attacks.imprint.shrinkage_reconstruct`
（全部 bin 一起做貝氏收縮的加權最小平方，解析解、不用訓練），在同一批測試病人、同一個雜訊種子下，跟：
    1. 原本的 closed-form（threshold + 單一 bin 除法 + fallback 成 prior）
    2. 學習型的 gradient-input ReCoNN（analysis/p12_grad_reconn.py，今天跑的 36 組結果）
三個放在一起比較。目的：確認「ReCoNN 在高雜訊下贏 closed-form」這件事，是不是單純因為 closed-form 的 prior
用得太粗糙（硬門檻 + 整個丟棄），換成一個便宜的解析式收縮就能打平，還是 ReCoNN 真的多學到東西。

使用方式
--------
    python -m analysis.p12_imprint_shrinkage --sigmas 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 --bins 16 --seed 0
"""
import argparse
import json
import os

import numpy as np
import torch

from attacks.imprint import ImprintBlock, add_gradient_noise, compute_empirical_bins, reconstruct, shrinkage_reconstruct
from trainers.fl_client import dp_clipped_gradient
from util.gdp import gdp_epsilon
from util.registry import build

ITR_SPEC = {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}}
CE_SPEC = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}
DATA_SHAPE = (13 * 41,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=200); ap.add_argument("--n-bin-aux", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=16); ap.add_argument("--linfunc", default="avg")
    ap.add_argument("--gain", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--thr-sd", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="p12_imprint_shrinkage")
    a = ap.parse_args()

    # n_aux (shadow pool for ReCoNN) 的樣本在這個 split 慣例裡位於 [n_bin_aux, n_bin_aux+2000)，這裡不訓練 ReCoNN，
    # 只是要跟 analysis/p12_grad_reconn.py 用同一組 n_bin_aux + test 切分，才能逐 sigma 直接比較同一批測試病人
    N_AUX_RESERVED = 2000
    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ds.x))
    i_bin_aux = perm[:a.n_bin_aux]
    i_test = perm[a.n_bin_aux + N_AUX_RESERVED:a.n_bin_aux + N_AUX_RESERVED + a.n_test]
    bin_aux_x = torch.as_tensor(ds.x[i_bin_aux], dtype=torch.float32)
    x_test = torch.as_tensor(ds.x[i_test], dtype=torch.float32)
    prior = bin_aux_x.mean(0)
    prior_var = float(bin_aux_x.var())

    bins_list = compute_empirical_bins(bin_aux_x, DATA_SHAPE, a.bins, linfunc=a.linfunc)
    torch.manual_seed(42)
    net = build(ITR_SPEC)
    block = ImprintBlock(DATA_SHAPE, a.bins, connection="linear", gain=a.gain, linfunc=a.linfunc, empirical_bins=bins_list)
    model = torch.nn.Sequential(block, net)
    model.eval()
    loss_fn = build(CE_SPEC)

    clean, g_norms = [], []
    for idx in i_test:
        x = torch.as_tensor(ds.x[idx:idx + 1], dtype=torch.float32)
        y = torch.zeros(1, dtype=torch.long)
        g, norm = dp_clipped_gradient(model, loss_fn, x, y, a.clip)
        clean.append(g); g_norms.append(float(norm.item()))
    frac_clipped = float(np.mean(np.array(g_norms) > a.clip))
    prior_mse = float(((prior[None] - x_test) ** 2).mean())

    rows = []
    for sigma in a.sigmas:
        cf_mse, shrink_mse = [], []
        for i, idx in enumerate(i_test):
            x = torch.as_tensor(ds.x[idx:idx + 1], dtype=torch.float32)
            g = add_gradient_noise(clean[i], sigma * a.clip, seed=1000 + i, distribution="gaussian") if sigma > 0 else clean[i]
            # [closed-form，跟 p12_imprint_b1.py 完全同一套]
            thr = a.thr_sd * 2 * sigma * a.clip if sigma > 0 else 0.0
            rec, n_hits = reconstruct(g, 1, DATA_SHAPE, bias_threshold=thr, reduction=None, structure=block.structure, clip_range=(-20.0, 20.0))
            guess_cf = rec[((rec - x) ** 2).mean(1).argmin()] if n_hits > 0 else prior
            cf_mse.append(float(((guess_cf - x[0]) ** 2).mean()))
            # [shrinkage，全部 bin 一起做貝氏收縮]
            xhat = shrinkage_reconstruct(g[0], g[1], noise_std=sigma * a.clip, prior=prior, prior_var=prior_var, structure=block.structure)
            xhat = xhat.clamp(-20.0, 20.0)
            shrink_mse.append(float(((xhat - x[0]) ** 2).mean()))
        eps = gdp_epsilon(1.0 / sigma, 1e-5) if sigma > 0 else float("inf")
        row = {"sigma": sigma, "epsilon": eps, "closed_form_mse_median": float(np.median(cf_mse)),
               "closed_form_ratio_median": float(np.median(cf_mse) / prior_mse),
               "shrinkage_mse_median": float(np.median(shrink_mse)), "shrinkage_ratio_median": float(np.median(shrink_mse) / prior_mse),
               "shrinkage_mse_mean": float(np.mean(shrink_mse)), "prior_mse": prior_mse}
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
        rows.append(row)

    d = os.path.join("runs", "results", a.out)
    os.makedirs(d, exist_ok=True)
    json.dump({"args": vars(a), "frac_clipped": frac_clipped, "rows": rows}, open(os.path.join(d, f"result_seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
