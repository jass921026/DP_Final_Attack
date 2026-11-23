"""
檔案功能
--------
Balle 重建攻擊的「排名分位」分析（補充 NN oracle，只看第一名的不足）：
    對每張重建圖 z_hat_i，算它與整個候選池（固定集 ∪ 影子目標）每一張圖的分數（MSE / PSNR / SSIM），
    得到一個分布；再看真正的目標 z_i 的分數落在這個分布的哪個位置。
        分位 p_i = 池中比目標「更接近 z_hat_i」的比例（平手算一半）。p 越小攻擊越成功；沒有資訊時 p 約為均勻分布。
    對照組（shuffle null）：改用別人的重建圖 z_hat_j 來看目標 z_i 的分位，估計「沒有關於 z_i 的資訊」時的分布。
    PSNR 與 MSE 單張排名相同（單調變換），因此排名只對 MSE 與 SSIM 各算一次；PSNR 僅用於報告數值。
使用方式
--------
    python -m analysis.balle_rank balle_nb_dp_clip_only balle_nb_dp_eps10000 ... [--out analysis_out/balle_rank]
    輸入：runs/results/<name>_*/reconstructions.npz 與 result.json（取 shadow_record 以讀池中的影子目標）
    輸出：<out>/summary.json、<out>/ranks_<name>.npz、<out>/rank_hist.png
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from util.registry import build
from util.splits import make_disjoint_splits

WIN, K1, K2, L = 7, 0.01, 0.03, 1.0


def _fixed_x(cfg):
    ds = build(cfg["data"])
    sp = make_disjoint_splits(len(ds), cfg["splits"]["sizes"], cfg["splits"]["seed"])
    return ds.get(sp["fixed"])[0]


def _stats(x):                                     # x: (n, 784) -> 窗內平均 mu、E[x^2]
    im = x.reshape(-1, 1, 28, 28)
    return F.avg_pool2d(im, WIN, 1), F.avg_pool2d(im * im, WIN, 1)


def ssim_matrix(rec, pool, device, chunk=8):
    """(n, m) 的 SSIM；窗口 7x7 均勻窗、無 padding；分塊以控制記憶體。"""
    c1, c2 = (K1 * L) ** 2, (K2 * L) ** 2
    P = torch.as_tensor(pool, dtype=torch.float32, device=device).reshape(-1, 1, 28, 28)
    mu_p, ex2_p = _stats(P.reshape(len(P), -1))
    out = np.empty((len(rec), len(pool)), dtype=np.float32)
    for s in range(0, len(rec), chunk):
        R = torch.as_tensor(rec[s:s + chunk], dtype=torch.float32, device=device).reshape(-1, 1, 28, 28)
        mu_r, ex2_r = _stats(R.reshape(len(R), -1))
        exy = F.avg_pool2d((R[:, None] * P[None]).reshape(-1, 1, 28, 28), WIN, 1).reshape(len(R), len(P), -1)
        mu_r_, mu_p_ = mu_r.reshape(len(R), 1, -1), mu_p.reshape(1, len(P), -1)
        var_r = (ex2_r.reshape(len(R), 1, -1) - mu_r_ ** 2)
        var_p = (ex2_p.reshape(1, len(P), -1) - mu_p_ ** 2)
        cov = exy - mu_r_ * mu_p_
        ss = ((2 * mu_r_ * mu_p_ + c1) * (2 * cov + c2)) / ((mu_r_ ** 2 + mu_p_ ** 2 + c1) * (var_r + var_p + c2))
        out[s:s + chunk] = ss.mean(-1).cpu().numpy()
    return out


def mse_matrix(rec, pool):
    r, p = rec.astype(np.float64), pool.astype(np.float64)
    return ((r ** 2).sum(1)[:, None] + (p ** 2).sum(1)[None] - 2 * r @ p.T).clip(min=0) / r.shape[1]


def percentile(pool_scores, target_scores, lower_is_better):
    """池中比目標更好的比例（平手算一半）。"""
    t = target_scores[:, None]
    better = (pool_scores < t) if lower_is_better else (pool_scores > t)
    tie = pool_scores == t
    return (better.sum(1) + 0.5 * tie.sum(1)) / pool_scores.shape[1]


def analyse(name, device, rng):
    d = os.path.dirname(sorted(glob.glob(f"runs/results/{name}_*/result.json"))[0])
    res, z = json.load(open(f"{d}/result.json")), np.load(f"{d}/reconstructions.npz")
    cfg = json.load(open(f"configs/{name}.json"))
    shadow = np.load(f"runs/balle_shadow_{res['shadow_record']}/arrays.npz")["targets"]
    pool = np.concatenate([_fixed_x(cfg), shadow]).astype(np.float32)
    rec, tgt = z["z_hat"].astype(np.float32), z["targets"].astype(np.float32)
    n = len(rec)
    perm = (np.arange(n) + rng.integers(1, n)) % n                       # 別人的重建圖（循環位移，必不等於自己）
    out = {"name": name, "eps": res.get("dp_epsilon_delta1e-5"), "n_pool": len(pool), "ranks": {}}
    mse_pool = mse_matrix(rec, pool)
    mse_t = ((rec - tgt) ** 2).mean(1)
    mse_t_null = ((rec[perm] - tgt) ** 2).mean(1)
    ssim_pool = ssim_matrix(rec, pool, device)
    ssim_t = _pairs(rec, tgt, device)
    ssim_t_null = _pairs(rec[perm], tgt, device)
    for key, pm, t, tn, lower in [("mse", mse_pool, mse_t, mse_t_null, True), ("ssim", ssim_pool, ssim_t, ssim_t_null, False)]:
        p = percentile(pm, t, lower)
        pn = percentile(pm[perm], tn, lower)                              # 別人的重建圖那一列的池分布
        out["ranks"][key] = p
        out[key] = {
            "mean_score": float(t.mean()), "auc": float(1 - p.mean()), "auc_null": float(1 - pn.mean()),
            "top1pct": float((p <= 0.01).mean()), "top5pct": float((p <= 0.05).mean()),
            "top1pct_null": float((pn <= 0.01).mean()), "top5pct_null": float((pn <= 0.05).mean()),
            "rank1": float((p <= 0.5 / len(pool)).mean()),
        }
        out["ranks"][key + "_null"] = pn
    out["psnr_mean"] = float((10 * np.log10(1.0 / np.maximum(mse_t, 1e-12))).mean())
    return out


def _pairs(a, b, device, chunk=1):
    """逐對 SSIM(a_i, b_i)：用 ssim_matrix 對每張只算 1 x 1。"""
    return np.array([ssim_matrix(a[i:i + 1], b[i:i + 1], device)[0, 0] for i in range(len(a))], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default="analysis_out/balle_rank")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng, summ, allr = np.random.default_rng(0), [], {}
    for nm in a.names:
        r = analyse(nm, a.device, rng)
        np.savez(f"{a.out}/ranks_{nm}.npz", **r["ranks"])
        allr[nm] = r["ranks"]
        r.pop("ranks")
        summ.append(r)
        print(f"{nm:28s} eps={r['eps']!s:10.10s} PSNR {r['psnr_mean']:.2f} | MSE auc {r['mse']['auc']:.3f} (null {r['mse']['auc_null']:.3f}) top1% {r['mse']['top1pct']:.3f} "
              f"| SSIM auc {r['ssim']['auc']:.3f} (null {r['ssim']['auc_null']:.3f}) top1% {r['ssim']['top1pct']:.3f}", flush=True)
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(a.names), figsize=(3 * len(a.names), 5), sharey=True, squeeze=False)
    for j, nm in enumerate(a.names):
        for i, key in enumerate(["mse", "ssim"]):
            ax[i, j].hist(allr[nm][key], bins=40, range=(0, 1), alpha=.7, label="attack", density=True)
            ax[i, j].hist(allr[nm][key + "_null"], bins=40, range=(0, 1), alpha=.4, label="shuffled", density=True)
            ax[i, j].set_title(f"{nm.replace('balle_nb_dp_', '')} · {key}", fontsize=8)
    ax[0, 0].legend(fontsize=7); fig.supxlabel("rank percentile of the target in the pool (0 = best)")
    fig.tight_layout(); fig.savefig(f"{a.out}/rank_hist.png", dpi=130)


if __name__ == "__main__":
    main()
