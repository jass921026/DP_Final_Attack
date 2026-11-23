"""
檔案功能
--------
P12 + iTransformer 上 imprint（Robbing the Fed）攻擊的最小可行性試驗：batch = 1。
惡意伺服器在 iTransformer 前插入 ImprintBlock（分箱邊界用輔助資料的經驗分位點，見 attacks.imprint.compute_empirical_bins，
不沿用 ImageNet 的常態/Laplace 假設），誠實客戶端對單一病人算一次梯度並回傳，伺服器用梯度差分還原輸入。
batch=1、無 DP 時理論上應該幾乎完美還原（只要該病人落在某個有效 bin）。

DP：採用與其他攻擊（imprint_dp_clip1*、balle_nb_dp_*）相同的慣例，梯度先逐筆裁剪到範數 <= clip，
再加 N(0, (sigma*clip)^2)（batch=1 時 n=1，雜訊不會被平均稀釋）。

重要：gain 只是 imprint 區塊的內部縮放（connection="linear" 會用 1/gain 還原，下游網路行為不受影響），
但它會讓「伺服器讀到的原始梯度」正比於 1/gain；若不做裁剪，惡意伺服器可以把 gain 調到任意小，讀到任意大的
原始訊號，躺贏任何固定尺度的雜訊（exploration 記錄裡先發現了這個假象）。只要客戶端照標準 DP-SGD 把梯度
裁剪到固定範數 clip，這個招數就會被完全抵消：裁剪後的訊號大小與 gain 幾乎無關（本檔預設 clip=1.0，
與專案裡其他 DP 設定一致），所以下面的掃描一律先裁剪。
使用方式
--------
    python -m analysis.p12_imprint_b1 [--n 200] [--bins 16] [--gain 1e-4] [--clip 1.0]
        [--sigmas 0 1e-4 3e-4 1e-3 3e-3 1e-2] [--out p12_imprint_b1]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--n-aux", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=16); ap.add_argument("--linfunc", default="avg")
    ap.add_argument("--gain", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--thr-sd", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ds.x))
    i_aux, i_test = perm[:a.n_aux], perm[a.n_aux:a.n_aux + a.n]
    aux_x = torch.as_tensor(ds.x[i_aux], dtype=torch.float32)
    prior = aux_x.mean(0)

    bins = compute_empirical_bins(aux_x, DATA_SHAPE, a.bins, linfunc=a.linfunc)
    torch.manual_seed(42)
    net = build(ITR_SPEC)
    block = ImprintBlock(DATA_SHAPE, a.bins, connection="linear", gain=a.gain, linfunc=a.linfunc, empirical_bins=bins)
    model = torch.nn.Sequential(block, net)
    model.eval()
    loss_fn = build(CE_SPEC)

    # [裁剪] 每個病人各自裁剪到範數 <= clip（n=1，dp_clipped_gradient 的平均步驟是除以 1，等於不變）
    clean, g_norms = [], []
    for idx in i_test:
        x = torch.as_tensor(ds.x[idx:idx + 1], dtype=torch.float32)
        y = torch.as_tensor(ds.y[idx:idx + 1], dtype=torch.long)
        g, norm = dp_clipped_gradient(model, loss_fn, x, y, a.clip)
        clean.append(g); g_norms.append(float(norm.item()))
    frac_clipped = float(np.mean(np.array(g_norms) > a.clip))

    rows = []
    for sigma in a.sigmas:
        mse_hit, hits, mse_all = [], [], []
        for i, idx in enumerate(i_test):
            x = torch.as_tensor(ds.x[idx:idx + 1], dtype=torch.float32)
            g = add_gradient_noise(clean[i], sigma * a.clip, seed=a.seed, distribution="gaussian") if sigma > 0 else clean[i]
            thr = a.thr_sd * 2 * sigma * a.clip if sigma > 0 else 0.0
            rec, n_hits = reconstruct(g, 1, DATA_SHAPE, bias_threshold=thr, reduction=None, structure=block.structure, clip_range=(-20.0, 20.0))
            hits.append(n_hits)
            guess = rec[((rec - x) ** 2).mean(1).argmin()] if n_hits > 0 else prior
            if n_hits > 0:
                mse_hit.append(float(((guess - x[0]) ** 2).mean()))
            mse_all.append(float(((guess - x[0]) ** 2).mean()))
        prior_mse = float(((prior[None] - torch.as_tensor(ds.x[i_test], dtype=torch.float32)) ** 2).mean())
        eps = gdp_epsilon(1.0 / sigma, 1e-5) if sigma > 0 else float("inf")
        row = {"sigma": sigma, "epsilon": eps, "clip": a.clip, "frac_clipped": frac_clipped,
               "hit_rate": float(np.mean([h > 0 for h in hits])), "mean_n_hits": float(np.mean(hits)),
               "mse_given_hit": float(np.mean(mse_hit)) if mse_hit else None, "mse_overall": float(np.mean(mse_all)),
               "prior_mse": prior_mse, "mse_ratio_overall": float(np.mean(mse_all) / prior_mse)}
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
        rows.append(row)

    if a.out:
        d = os.path.join("runs", "results", a.out)
        os.makedirs(d, exist_ok=True)
        json.dump({"args": vars(a), "rows": rows}, open(os.path.join(d, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
