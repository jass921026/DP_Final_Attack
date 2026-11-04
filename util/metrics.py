"""
檔案功能
--------
重建攻擊的評估指標（Balle et al. 2022, Sec. V-B）：
    - 逐樣本 MSE（目標 z 與重建 z_hat 的像素均方誤差）
    - Nearest-Neighbor (NN) oracle：對每個目標，取對手可用資料池中與它 MSE 最小的那一點，其 MSE 作為「成功重建」的保守門檻
    - 「勝過 NN」的比例

使用方式
--------
    from util.metrics import mse_per_sample, nn_oracle_mse, frac_better_than_nn
    m = mse_per_sample(recon, targets)                 # (n,)
    nn = nn_oracle_mse(targets, pool)                  # (n,)
    frac_better_than_nn(m, nn)                         # 標量
注意：成對距離用矩陣乘法算，不建立 n x m x D 的中間張量（先前實驗曾因此耗盡記憶體）。
"""
import numpy as np


def mse_per_sample(recon: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    功能
        每個樣本的 MSE。論文 Sec. V-B(a)：reports the MSE between a target and its reconstruction。
    Pseudo code
        for i in 1..n:  mse_i <- mean_d (recon[i,d] - target[i,d])^2
    用法
        mse_per_sample(recon, target)   # 形狀 (n, D) 與 (n, D)
    """
    # [對應 Pseudo code] 向量化的逐樣本平均平方誤差
    return ((recon - target) ** 2).mean(axis=1)


def nn_oracle_mse(targets: np.ndarray, pool: np.ndarray, chunk: int = 256) -> np.ndarray:
    """
    功能
        NN oracle（Balle 2022 Sec. V-B(d)）：對每個目標 z，猜測對手可用資料池 D- ∪ D-bar 中
        MSE 距離最小的點 z_hat；回傳該最小 MSE。若重建的 MSE 小於它，代表重建比池中任何點都更接近目標，
        因此攻擊必定從模型中取得了目標的獨有資訊。
    Pseudo code (論文文字描述)
        for each target z_i:
            z_hat_i <- argmin_{p in pool} MSE(z_i, p)
            nn_i    <- MSE(z_i, z_hat_i)
    用法
        nn_oracle_mse(test_targets, np.concatenate([fixed_x, shadow_x]))
    """
    # [對應 Pseudo code] 以 ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab 計算成對 MSE，分塊以控制記憶體
    t = targets.astype(np.float64)
    p = pool.astype(np.float64)
    d = t.shape[1]
    p_sq = (p ** 2).sum(1)
    out = np.empty(len(t))
    for s in range(0, len(t), chunk):
        tb = t[s:s + chunk]
        dist = ((tb ** 2).sum(1)[:, None] + p_sq[None, :] - 2 * tb @ p.T).clip(min=0) / d
        out[s:s + chunk] = dist.min(axis=1)   # argmin 的距離
    return out


def frac_better_than_nn(mse: np.ndarray, nn_mse: np.ndarray) -> float:
    """
    功能
        重建 MSE 小於該目標 NN-oracle MSE 的目標比例（本專案沿用的報告方式）。
    Pseudo code
        return mean_i [ mse_i < nn_i ]
    用法
        frac_better_than_nn(mse, nn)
    """
    # [對應 Pseudo code]
    return float((mse < nn_mse).mean())
