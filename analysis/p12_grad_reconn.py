"""
檔案功能
--------
把 Balle 2022 的 ReCoNN 想法搬到「單筆梯度」層級：不是用 (最終模型參數 -> 目標資料) 訓練重建網路，
而是用 (單一病人、單次 DP-clipped(+noised) 梯度 -> 該病人的原始資料) 訓練重建網路 phi，取代
p12_imprint_b1.py 用的 closed-form 梯度差分除法 (invert_cumulative / reconstruct)。

動機（跟使用者口頭討論的對齊）：
    closed-form 的 dW_i/db_i 是「單一 bin 的單一比值」，雜訊 sigma 一加大就直接炸開（db_i 很小時尤其明顯，
    reconstruct() 要靠 bias_threshold 篩掉這些 bin）。但是整個 dW（長度 D，D=533，遠大於 1）都共享同一個
    db_i（rank-1 外積結構：dW_i = x_hat * db_i + noise_i，其中 noise 是 D 維 iid Gaussian、"signal" 卻是
    1 維方向上的纯量縮放）——ReCoNN 直接吃全部 (dW, db)（num_bins 個 bin 一起，而不是逐 bin 處理），讓神經網路自己
    學怎麼把 num_bins*D 維的 bin 梯度 + num_bins 維的 bias 梯度，聯合去噪、聯合估計出 D 維的 x，等於用「多個 bin
    對同一個輸入的不同線性視角」做聯合估計，而不是只信任「命中」的那個 bin 的單一除法。另外 phi 可以學到資料的
    先驗分布（P12 有 ~66% 補值 0），在雜訊蓋過訊號時自動往 prior 收縮，這是 closed-form 除法做不到的
    （對應跟使用者討論的「post-processing 不能無中生有，但可以用先驗/冗餘換 MSE」）。

每個 sigma 各自訓練一個 ReCoNN（攻擊者知道 DP 機制與 sigma，是標準假設——跟 Balle shadow 訓練用同一個演算法 A
一致）。訓練時每個 epoch 對同一批 aux 梯度重新抽一次高斯雜訊（等於 denoising 訓練，讓 phi 學到「雜訊分佈」而不是
背熟某一次雜訊抽樣），留出集（固定種子）在每個 epoch 加同分佈但固定一次的雜訊來做模型選擇，測試集同理固定一次。

使用方式
--------
    python -m analysis.p12_grad_reconn --sigma 0 --bins 16 --n-aux 800 --n-test 150 --epochs 150
    python -m analysis.p12_grad_reconn --sigma 1e-3 --bins 16 ...
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from attacks.imprint import ImprintBlock, add_gradient_noise, compute_empirical_bins, reconstruct
from trainers.fl_client import dp_clipped_gradient
from util.gdp import gdp_epsilon
from util.registry import build

ITR_SPEC = {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}}
CE_SPEC = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}
DATA_SHAPE = (13 * 41,)
D = DATA_SHAPE[0]


def build_victim(bins, gain, linfunc, empirical_bins):
    """功能：建立惡意伺服器插入 ImprintBlock 後的 victim 模型（跟 p12_imprint_b1 同一套固定隨機初始化，seed=42）。"""
    torch.manual_seed(42)
    net = build(ITR_SPEC)
    block = ImprintBlock(DATA_SHAPE, bins, connection="linear", gain=gain, linfunc=linfunc, empirical_bins=empirical_bins)
    model = torch.nn.Sequential(block, net)
    model.eval()
    return model, block


def clean_grad_features(model, loss_fn, clip, x_all):
    """
    功能
        對每個病人各自算一次 DP-clipped 梯度（n=1，裁剪慣例與 p12_imprint_b1 相同），只取出 ImprintBlock.linear0
        的 (dW, db)（model.parameters() 的前兩個張量：見 attacks/imprint.py ImprintBlock.__init__ 的參數建立順序），
        拉平成一個特徵向量。回傳 (n, bins*D+bins) 的「乾淨」（sigma=0）裁剪後梯度，之後各 sigma 各自加噪（見 add_noise）。
    """
    feats, norms = [], []
    for i in range(len(x_all)):
        xi = x_all[i:i + 1]
        yi = torch.zeros(1, dtype=torch.long)       # imprint 攻擊不看標籤正確性，loss 只是為了產生梯度，標籤值任意（固定 0）
        g, norm = dp_clipped_gradient(model, loss_fn, xi, yi, clip)
        feats.append(torch.cat([g[0].reshape(-1), g[1].reshape(-1)]).numpy())
        norms.append(float(norm.item()))
    return np.stack(feats).astype(np.float32), np.array(norms)


def add_noise_np(feats: np.ndarray, sigma: float, clip: float, seed: int) -> np.ndarray:
    """功能：對 (n, bins*D+bins) 的梯度特徵逐筆加 N(0, (sigma*clip)^2)（跟 add_gradient_noise 同分佈，直接在 numpy 上做比較快）。"""
    if sigma <= 0:
        return feats
    rng = np.random.default_rng(seed)
    return (feats + rng.normal(0.0, sigma * clip, feats.shape)).astype(np.float32)


def closed_form_baseline(model, block, loss_fn, clip, sigma, x_all, prior, thr_sd=3.0):
    """功能：同一批測試病人上跑 p12_imprint_b1 的 closed-form reconstruct()，當作對照組（複用同一套固定雜訊種子慣例）。"""
    mse_hit, hits, mse_all = [], [], []
    for i in range(len(x_all)):
        xi = x_all[i:i + 1]
        yi = torch.zeros(1, dtype=torch.long)
        g, _ = dp_clipped_gradient(model, loss_fn, xi, yi, clip)
        g = add_gradient_noise(g, sigma * clip, seed=1000 + i, distribution="gaussian") if sigma > 0 else g
        thr = thr_sd * 2 * sigma * clip if sigma > 0 else 0.0
        rec, n_hits = reconstruct(g, 1, DATA_SHAPE, bias_threshold=thr, reduction=None, structure=block.structure, clip_range=(-20.0, 20.0))
        hits.append(n_hits)
        guess = rec[((rec - xi) ** 2).mean(1).argmin()] if n_hits > 0 else prior
        if n_hits > 0:
            mse_hit.append(float(((guess - xi[0]) ** 2).mean()))
        mse_all.append(float(((guess - xi[0]) ** 2).mean()))
    return {"hit_rate": float(np.mean([h > 0 for h in hits])), "mean_n_hits": float(np.mean(hits)),
            "mse_given_hit": float(np.mean(mse_hit)) if mse_hit else None, "mse_overall": float(np.mean(mse_all))}


def train_grad_reconn(feat_aux, x_aux, feat_test, x_test, sigma, clip, hidden, epochs, lr, bs, seed, device, val_frac=0.2):
    """
    功能
        訓練 ReCoNN phi：(裁剪後梯度特徵, 隨機 DP 雜訊) -> 標準化後的 x。每個 epoch 對整批 aux 特徵重新抽一次雜訊
        （denoising 訓練），留出集與測試集各用一次固定種子的雜訊評估（可重複比較）。
    Pseudo code
        mu_feat, sd_feat <- aux 特徵逐座標標準化統計量（算在 sigma=0 的乾淨特徵上，雜訊是零均值不改變統計量）
        zmu <- aux 資料的均值（prior，ReCoNN 輸出殘差）
        for epoch in 1..epochs:
            feat_noisy <- feat_aux + N(0,(sigma*clip)^2)           # 每個 epoch 重新抽
            phi <- 用 (scale(feat_noisy), x_aux - zmu) 的 MAE+MSE 訓練一個 epoch
            算留出集（固定雜訊）MSE，留最好的 epoch
    """
    torch.manual_seed(seed)
    n = len(feat_aux)
    nval = int(n * val_frac)
    idx = np.random.default_rng(seed).permutation(n)
    i_tr, i_val = idx[nval:], idx[:nval]
    mu_f, sd_f = feat_aux.mean(0), feat_aux.std(0) + 1e-8
    zmu = x_aux[i_tr].mean(0)

    phi = build({"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": hidden, "activation": "relu"}},
                in_dim=feat_aux.shape[1], out_dim=D).to(device)
    opt = torch.optim.AdamW(phi.parameters(), lr=lr, weight_decay=1e-4)

    feat_val_noisy = add_noise_np(feat_aux[i_val], sigma, clip, seed=999)
    Xval = torch.as_tensor((feat_val_noisy - mu_f) / sd_f, device=device)
    Yval = torch.as_tensor(x_aux[i_val] - zmu, dtype=torch.float32, device=device)

    best, best_state, hist = (1e9, 0), None, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        feat_tr_noisy = add_noise_np(feat_aux[i_tr], sigma, clip, seed=10_000 + ep)
        Xtr = torch.as_tensor((feat_tr_noisy - mu_f) / sd_f, device=device)
        Ytr = torch.as_tensor(x_aux[i_tr] - zmu, dtype=torch.float32, device=device)
        phi.train()
        perm = torch.randperm(len(i_tr), generator=g)
        for s in range(0, len(i_tr), bs):
            b = perm[s:s + bs]
            e = phi(Xtr[b]) - Ytr[b]
            loss = e.abs().mean() + (e ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(phi.parameters(), 5.0)   # 高雜訊時單筆 outlier 容易讓 MSE 項梯度暴衝，裁剪穩定訓練
            opt.step()
        phi.eval()
        with torch.no_grad():
            va = float(((phi(Xval) - Yval) ** 2).mean())
        hist.append(va)
        if va < best[0]:
            best, best_state = (va, ep), {k: v.clone() for k, v in phi.state_dict().items()}
    phi.load_state_dict(best_state); phi.eval()

    feat_test_noisy = add_noise_np(feat_test, sigma, clip, seed=1000)
    Xte = torch.as_tensor((feat_test_noisy - mu_f) / sd_f, device=device)
    with torch.no_grad():
        xhat = phi(Xte).cpu().numpy() + zmu
    return xhat, zmu, {"best_epoch": best[1], "holdout_mse_best": best[0], "holdout_prior_mse": float((Ytr.cpu().numpy() ** 2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aux", type=int, default=800); ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--n-bin-aux", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=16); ap.add_argument("--linfunc", default="avg")
    ap.add_argument("--gain", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1000, 1000]); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="p12_grad_reconn")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t0 = time.time()

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ds.x))
    i_bin_aux = perm[:a.n_bin_aux]
    i_aux = perm[a.n_bin_aux:a.n_bin_aux + a.n_aux]
    i_test = perm[a.n_bin_aux + a.n_aux:a.n_bin_aux + a.n_aux + a.n_test]
    bin_aux_x = torch.as_tensor(ds.x[i_bin_aux], dtype=torch.float32)
    x_aux = torch.as_tensor(ds.x[i_aux], dtype=torch.float32)
    x_test = torch.as_tensor(ds.x[i_test], dtype=torch.float32)
    prior = bin_aux_x.mean(0)

    bins_list = compute_empirical_bins(bin_aux_x, DATA_SHAPE, a.bins, linfunc=a.linfunc)
    model, block = build_victim(a.bins, a.gain, a.linfunc, bins_list)
    loss_fn = build(CE_SPEC)

    # 乾淨（sigma=0）裁剪後梯度跟 sigma 無關（雜訊是之後才加的），同一組 (bins, n_aux, n_test, seed, ...) 下
    # 掃多個 sigma 時只算一次、存成快取重複使用（省掉逐病人單獨跑一次 autograd 的時間，這部分跟 sigma 無關）
    cache_dir = os.path.join("analysis_out", "p12_grad_reconn_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"bins{a.bins}_{a.linfunc}_gain{a.gain}_clip{a.clip}_naux{a.n_aux}_ntest{a.n_test}_nbinaux{a.n_bin_aux}_seed{a.seed}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        feat_aux, feat_test, norms_aux, norms_test = cached["feat_aux"], cached["feat_test"], cached["norms_aux"], cached["norms_test"]
    else:
        feat_aux, norms_aux = clean_grad_features(model, loss_fn, a.clip, x_aux)
        feat_test, norms_test = clean_grad_features(model, loss_fn, a.clip, x_test)
        np.savez(cache_path, feat_aux=feat_aux, feat_test=feat_test, norms_aux=norms_aux, norms_test=norms_test)
    frac_clipped = float(np.mean(np.concatenate([norms_aux, norms_test]) > a.clip))

    xhat, zmu, train_info = train_grad_reconn(feat_aux, x_aux.numpy(), feat_test, None, a.sigma, a.clip,
                                               a.hidden, a.epochs, a.lr, a.bs, a.seed, a.device)
    # 輸出截到資料的合理範圍（跟 closed-form reconstruct() 的 clip_range 一致），只是避免個別病人梯度在高雜訊
    # 下把 MLP 的輸出推到不合理的極端值，不是訓練的一部分（post-hoc 後處理，不影響梯度/訓練過程）
    xhat = np.clip(xhat, -20.0, 20.0)
    mse_reconn = ((xhat - x_test.numpy()) ** 2).mean(1)
    prior_mse = float(((prior.numpy()[None] - x_test.numpy()) ** 2).mean())

    baseline = closed_form_baseline(model, block, loss_fn, a.clip, a.sigma, x_test, prior)
    eps = gdp_epsilon(1.0 / a.sigma, 1e-5) if a.sigma > 0 else float("inf")

    result = {
        "args": vars(a), "frac_clipped": frac_clipped, "epsilon": eps, "train_info": train_info,
        "reconn": {"mse_mean": float(mse_reconn.mean()), "mse_median": float(np.median(mse_reconn)),
                   "frac_perfect": float((mse_reconn < 0.05).mean()), "frac_better_than_prior": float((mse_reconn < prior_mse).mean()),
                   "prior_mse": prior_mse, "mse_ratio_median": float(np.median(mse_reconn) / prior_mse)},
        "closed_form_baseline": baseline, "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=1))
    d = os.path.join("runs", "results", a.out)
    os.makedirs(d, exist_ok=True)
    tag = f"sigma{a.sigma}_bins{a.bins}_naux{a.n_aux}_seed{a.seed}"
    json.dump(result, open(os.path.join(d, f"result_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
