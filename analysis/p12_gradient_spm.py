"""
檔案功能
--------
使用者 2026-11-25 的設計(跟 Imprint / Balle weight-ReCoNN 並排驗證):用誠實架構(不插入 ImprintBlock)的
單樣本、單步梯度,套用 `attacks/prior_aware_spm.py` 的 posterior-mean 機制重建資料——介於 Imprint(需要惡意
架構,但便宜、精確)跟 Balle weight-ReCoNN(誠實架構,但要訓練上千個 shadow 模型)中間的第三條路。

數學結構(對應討論裡的 1-4 步,D⁻=∅、batch=1、先無 DP 驗證)：
    1. 攻擊者已知 θ(固定隨機初始化,seed=42,跟其他 P12 腳本一致)。
    2. 對候選池裡每個候選 zⱼ 跟測試目標 z*,各自算「單樣本、單步」的梯度 gⱼ = ∇loss(θ, zⱼ)
       （`trainers.fl_client.dp_clipped_gradient`,n=1,D⁻=∅：這一步本來就等於直接觀測梯度，不用減掉固定集
       貢獻；clip 到範數 ≤ C）。
    3. 用 `attacks.prior_aware_spm.spm_attack` 對這些梯度做似然比 posterior mean，得到：
           ĝ = Σ wⱼ gⱼ                    （梯度空間的 posterior mean，跟 z* 的真實梯度比較用的診斷量）
           x̂ = Σ wⱼ zⱼ                    （同一組 wⱼ，套用在候選的原始資料上 —— 貝氏推論下，只要 wⱼ 算對，
                                             這對任何 f(z) 都是 E[f(z*)|觀測] 的估計，不需要額外轉換）
    4. x̂ 就是這次攻擊的重建結果，用跟 Imprint/Balle/SPM-on-data 同一套 `rank_metrics` 評分，才能直接比較。

--model linear：拿掉所有非線性(單一個 nn.Linear(D,2)，無隱藏層、無啟動函數)，驗證數學結構本身有沒有刻對
    （GLM 下 Σwⱼzⱼ 應該精確等於資料空間的貝氏最優估計，數字應該接近直接對資料做 SPM 的結果，不是巧合）。
--model {itransformer,mlp,gru}：誠實的真實架構(不插入 ImprintBlock)，驗證「梯度→圖片」這個轉換在非線性下
    還剩多少效果——使用者的預期：non-DP 下介於 Imprint 跟 Balle 之間，如果轉換很不平凡則可能比 Balle 還差。

使用方式
--------
    python -m analysis.p12_gradient_spm --model linear
    python -m analysis.p12_gradient_spm --model itransformer
"""
import argparse
import json
import os

import numpy as np
import torch

from attacks.prior_aware_spm import clip_candidates, spm_attack
from trainers.fl_client import dp_clipped_gradient
from util.registry import build

DATA_SHAPE = (13 * 41,)
CE_SPEC = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}
MODEL_SPECS = {
    "linear": {"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": [], "activation": "relu", "in_dim": 533, "out_dim": 2}},
    "itransformer": {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}},
    "mlp": {"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": [256, 128], "activation": "relu", "in_dim": 533, "out_dim": 2}},
    "gru": {"module": "models.gru_cls", "type": "GRUClassifier", "kwargs": {}},
}


def rank_metrics(xhat: np.ndarray, x_true: np.ndarray, pool: np.ndarray):
    """功能：同 analysis/p12_wg_reconn.py 的 rank_metrics（NN-oracle 池子 + 百分位排名）。"""
    d_pool = (xhat ** 2).sum(1, keepdims=True) + (pool ** 2).sum(1)[None, :] - 2 * xhat @ pool.T
    d_true = ((xhat - x_true) ** 2).sum(1)
    pct = ((d_pool < d_true[:, None]).sum(1) + 0.5 * (d_pool == d_true[:, None]).sum(1)) / len(pool)
    roll = np.roll(np.arange(len(x_true)), 1)
    d_null = ((xhat[roll] - x_true) ** 2).sum(1)
    pct_null = (d_pool[roll] < d_null[:, None]).sum(1) / len(pool)
    return float(1 - pct.mean()), float(1 - pct_null.mean())


def flat_grads(model, loss_fn, x_all, clip):
    """功能：對每一筆樣本各自算一次(單樣本、D⁻=空)、裁剪過的全參數梯度，攤平成一個向量。"""
    out = []
    for i in range(len(x_all)):
        g, _ = dp_clipped_gradient(model, loss_fn, x_all[i:i + 1], torch.zeros(1, dtype=torch.long), clip)
        out.append(torch.cat([t.reshape(-1) for t in g]))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--n-aux", type=int, default=1500); ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_tag = a.out or f"p12_gradient_spm_{a.model}"

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ds.x))
    i_aux = perm[:a.n_aux]                                       # 候選池
    i_test = perm[a.n_aux:a.n_aux + a.n_test]                    # 攻擊目標
    x_aux = torch.as_tensor(ds.x[i_aux], dtype=torch.float32)
    x_test = torch.as_tensor(ds.x[i_test], dtype=torch.float32)

    torch.manual_seed(42)                                        # theta：固定隨機初始化，跟其他 P12 腳本一致
    model = build(MODEL_SPECS[a.model]); model.eval()
    loss_fn = build(CE_SPEC)

    print(f"[{a.model}] 算候選池梯度 ({len(x_aux)} 筆)...", flush=True)
    g_aux = flat_grads(model, loss_fn, x_aux, a.clip)
    print(f"[{a.model}] 算目標梯度 ({len(x_test)} 筆)...", flush=True)
    g_test = flat_grads(model, loss_fn, x_test, a.clip)

    cand_c, cand_a, cand_norm2 = clip_candidates(g_aux, a.clip)   # 已經裁剪過一次了（dp_clipped_gradient），這裡 clip_candidates 等於是 identity（範數已 <= clip）
    noise_std = max(a.sigma * a.clip, 1e-8)
    g_target_noisy = g_test + torch.randn_like(g_test) * noise_std if a.sigma > 0 else g_test
    G = g_target_noisy[:, None, :]

    u_mean, u_map = spm_attack(G, cand_c, cand_norm2, cand_a, noise_std)   # 梯度空間的 posterior mean（第 3 步）
    # [對應第 4 步] 同一組 posterior 權重 wⱼ，套用在候選的原始資料 x_aux 上（不是梯度）
    scores = _scores(G, cand_c, cand_norm2, noise_std)
    w = torch.softmax(scores, 1)
    xhat = (w.float() @ x_aux).numpy()

    g_err = float(((u_mean - g_test) ** 2).mean())                # 診斷：梯度空間的估計 vs 真實梯度
    rank_score, rank_null = rank_metrics(xhat, x_test.numpy(), x_aux.numpy())
    row = {"model": a.model, "sigma": a.sigma, "grad_mse": g_err, "rank_score": rank_score, "rank_null": rank_null}
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))

    d = os.path.join("runs", "results", out_tag)
    os.makedirs(d, exist_ok=True)
    json.dump({"args": vars(a), "row": row}, open(os.path.join(d, f"result_seed{a.seed}.json"), "w"), indent=1)


def _scores(G, cand_c, cand_norm2, noise_std):
    dots = torch.einsum("ntd,cd->ntc", G, cand_c).double()
    llr = (2.0 * dots - cand_norm2[None, None, :].double()) / (2.0 * noise_std ** 2)
    return llr.sum(1)


if __name__ == "__main__":
    main()
