"""
檔案功能
--------
RMIA（Zarifzadeh, Liu, Shokri; ICML 2024, "Low-Cost High-Power Membership Inference Attacks"）的離線版本，
只依賴 softmax 訊號與成員矩陣，與模型、資料集、訓練方式無關（訓練由 models/、loss/、trainers/ 負責，訊號存成 RunRecord）。
包含：
    softmax_true_class      訊號：真實類別的 softmax 機率（官方 get_signals.get_softmax）
    select_out_signals      從參考模型取出每個樣本的 OUT 訊號（selection="master" 複製官方 master；"paper" 為論文版）
    rmia_scores             核心：成對似然比檢定的 MIA 分數（論文 Eq. 與 Appendix Algorithm 1，離線）
    rmia_attack             整合：選參考模型 -> 分數
    tune_offline_a          以配對模型調 offline_a（官方作法）
    roc_summary             評估：AUC 與指定 FPR 下的 TPR
tests/test_rmia.py 以官方 ml_privacy_meter 的 run_rmia 與官方存下的分數驗證一致性。

使用方式
--------
    from attacks.rmia import rmia_attack, roc_summary
    scores = rmia_attack(all_signals, pop_signals, memberships, target_idx=0, num_ref=4, a=0.3, gamma=2.0)
    stats = roc_summary(scores, memberships[:, 0])
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from attacks.base import BaseAttack
from util.data import make_loader


def softmax_true_class(logits: torch.Tensor, labels: torch.Tensor, temp: float = 1.0) -> np.ndarray:
    """
    功能
        MIA 訊號 p(x|theta)：模型在真實類別上的 softmax 機率（可帶溫度）。
    Pseudo code (官方 get_signals.get_softmax)
        p <- softmax(logits / temp);  signal <- p[true_label]
    用法
        sig = softmax_true_class(model(x), y)     # (n,1) numpy
    """
    p = torch.softmax(logits / temp, dim=1)
    return p.gather(1, labels.reshape(-1, 1)).cpu().numpy()


@torch.no_grad()
def model_signals(model: torch.nn.Module, dataset, indices=None, device: str = "cpu", batch_size: int = 1000, temp: float = 1.0) -> np.ndarray:
    """
    功能
        模型在資料集（torch Dataset，可指定索引子集）上的 MIA 訊號：以 DataLoader 逐批取資料，算真實類別 softmax。
    Pseudo code
        for (x_b, y_b) in DataLoader(dataset[indices]):  signal_b <- softmax(model(x_b)/temp)[y_b]
        return concat(signal_b)
    用法
        sig = model_signals(model, train_ds, audit_idx, "mps")     # (n,1)
    """
    model = model.to(device).eval()
    out = [softmax_true_class(model(xb.to(device)), yb.to(device), temp) for xb, yb in make_loader(dataset, indices, batch_size)]
    model.cpu()
    return np.concatenate(out)


def select_out_signals(all_signals: np.ndarray, memberships: np.ndarray, target_idx: int, num_ref: int,
                       selection: str = "master") -> np.ndarray:
    """
    功能
        每個樣本取 num_ref 個「OUT」參考訊號（該樣本沒在訓練集中的參考模型上的 softmax）。
    Pseudo code
        selection="master"（官方 master attacks.get_rmia_out_signals）:
            columns <- 非目標、非配對（成員互補）模型的前 2*num_ref 個
            out <- signals[:, columns] * (非成員)            # 成員處補 0
            每列由大到小排序，取前 num_ref 個                 # OUT 不足 num_ref 個時以 0 補入平均
        selection="paper"（論文離線設定：參考模型對該樣本皆為 OUT）:
            每個樣本只用「OUT 的參考模型」的訊號，取前 num_ref 個（不足時以 NaN 忽略，後續用 nanmean）
    用法
        out = select_out_signals(all_signals, memberships, 0, 4)      # (n, num_ref)
    """
    paired = target_idx + 1 if target_idx % 2 == 0 else target_idx - 1
    cols = [i for i in range(all_signals.shape[1]) if i != target_idx and i != paired][: 2 * num_ref]
    sig, non_member = all_signals[:, cols], ~memberships[:, cols]
    if selection == "master":
        out = sig * non_member
        return -np.sort(-out, axis=1)[:, :num_ref]
    out = np.where(non_member, sig, np.nan)
    order = np.argsort(np.isnan(out), axis=1, kind="stable")            # NaN 排到最後
    return np.take_along_axis(out, order, axis=1)[:, :num_ref]


def rmia_scores(target_x: np.ndarray, target_z: np.ndarray, out_x: np.ndarray, out_z: np.ndarray,
                a: float, gamma: float = 1.0, chunk: int = 2000) -> np.ndarray:
    """
    功能
        RMIA 分數：樣本 x 在多少比例的母體樣本 z 上「γ-支配」z。論文 Eq.(2)-(3) 與 Appendix Algorithm 1（離線）。
    Pseudo code (論文)
        p_OUT(x) <- mean over OUT 參考模型 of Pr(x | theta')
        Pr(x)    <- ((1 + a)/2) * p_OUT(x) + (1 - a)/2                 # 離線近似
        Ratio_x  <- Pr(x | theta) / Pr(x);   Ratio_z <- Pr(z | theta) / Pr(z)    (z 同理，皆為 OUT)
        LR(x, z) <- Ratio_x / Ratio_z
        Score(x) <- (1/|Z|) * sum_{z in Z} 1[ LR(x, z) > gamma ]
    用法
        s = rmia_scores(sig_x, sig_z, out_x, out_z, a=0.3, gamma=2.0)    # (n,)
        target_x: (n,) 或 (n,1)；target_z: (m,)；out_x: (n,R)；out_z: (m,R)
    """
    # [對應 Pseudo code: p_OUT, Pr(x)]
    mean_x = (1 + a) / 2 * np.nanmean(out_x, axis=1) + (1 - a) / 2
    mean_z = (1 + a) / 2 * np.nanmean(out_z, axis=1) + (1 - a) / 2
    # [對應 Pseudo code: Ratio_x, Ratio_z]
    ratio_x = target_x.ravel() / mean_x
    ratio_z = target_z.ravel() / mean_z
    # [對應 Pseudo code: LR 與 Score] 對 x 分塊計算（n x m 的比較矩陣不一次放進記憶體；結果與不分塊相同）
    return np.concatenate([np.average(ratio_x[s:s + chunk, None] / ratio_z > gamma, axis=1) for s in range(0, len(ratio_x), chunk)])


def rmia_attack(all_signals: np.ndarray, pop_signals: np.ndarray, memberships: np.ndarray, target_idx: int,
                num_ref: int, a: float, gamma: float = 1.0, selection: str = "master") -> np.ndarray:
    """
    功能
        整合流程：為目標模型選參考模型的 OUT 訊號（審計樣本與母體樣本各一份），再算 RMIA 分數。
        母體樣本對所有模型皆為 OUT（官方：population_memberships 全為 False）。
    Pseudo code
        out_x <- select_out_signals(all_signals, memberships, target, R)
        out_z <- select_out_signals(pop_signals, 全 False 的成員矩陣, target, R)
        return rmia_scores(all_signals[:, target], pop_signals[:, target], out_x, out_z, a, gamma)
    用法
        scores = rmia_attack(sig, pop, mem, 0, 4, a=0.3, gamma=2.0)
        all_signals: (n_samples, n_models)；pop_signals: (n_pop, n_models)；memberships: (n_samples, n_models) bool
    """
    out_x = select_out_signals(all_signals, memberships, target_idx, num_ref, selection)
    out_z = select_out_signals(pop_signals, np.zeros(pop_signals.shape, dtype=bool), target_idx, num_ref, selection)
    return rmia_scores(all_signals[:, target_idx], pop_signals[:, target_idx], out_x, out_z, a, gamma)


def roc_summary(scores: np.ndarray, members: np.ndarray, fprs: Sequence[float] = (0.0, 0.001, 0.01, 0.1)) -> Dict[str, float]:
    """
    功能
        AUC 與在指定 FPR 下的 TPR（取 FPR <= 指定值 的最大 TPR；FPR=0 即不誤判任何非成員時的 TPR）。
    Pseudo code
        fpr, tpr <- ROC(scores, members);  auc <- 面積
        for f in fprs:  tpr_at_f <- max{ tpr_i : fpr_i <= f }
    用法
        roc_summary(scores, members)  ->  {"auc":..., "tpr@0":..., "tpr@0.001":..., ...}
    """
    from sklearn.metrics import auc as _auc, roc_curve
    fpr, tpr, _ = roc_curve(members.ravel(), scores.ravel())
    out = {"auc": float(_auc(fpr, tpr))}
    for f in fprs:
        out[f"tpr@{f:g}"] = float(tpr[fpr <= f].max()) if (fpr <= f).any() else 0.0
    return out


def tune_offline_a(target_idx: int, all_signals: np.ndarray, pop_signals: np.ndarray, memberships: np.ndarray,
                   gamma: float = 1.0, selection: str = "master", grid: Sequence[float] = tuple(np.arange(0, 1.1, 0.1))) -> Tuple[float, float]:
    """
    功能
        調 offline_a：用「配對模型」（與目標成員互補，攻擊者可取得）當攻擊對象，用 1 個參考模型對，選 AUC 最大的 a
        （官方 attacks.tune_offline_a；論文 Appendix B.2.2）。回傳 (a, 該 a 的 AUC)。
    Pseudo code
        paired <- 配對模型
        for a in grid:  AUC_a <- AUC( RMIA(target=paired, R=1, a), memberships[:, paired] )
        return argmax_a AUC_a
    用法
        a, auc = tune_offline_a(0, sig, pop, mem)
    """
    from sklearn.metrics import auc as _auc, roc_curve
    paired = target_idx + 1 if target_idx % 2 == 0 else target_idx - 1
    best_a, best_auc = 0.0, 0.0
    for test_a in grid:
        s = rmia_attack(all_signals, pop_signals, memberships, paired, 1, test_a, gamma, selection)
        fpr, tpr, _ = roc_curve(memberships[:, paired].ravel(), s.ravel())
        cur = _auc(fpr, tpr)
        if cur > best_auc:
            best_a, best_auc = float(test_a), float(cur)
    return best_a, best_auc


class RMIAAttack(BaseAttack):
    """
    功能
        RMIA 的 BaseAttack 實作：run() = attacks.rmia_experiment.run_rmia_experiment(cfg, root)
        （訓練成對模型 -> 訊號 -> 攻擊 -> 彙整）。
    用法
        RMIAAttack(cfg, root="runs").run()
    """

    def run(self):
        """功能：執行完整 RMIA 實驗並回傳彙整結果。"""
        from attacks.rmia_experiment import run_rmia_experiment
        return run_rmia_experiment(self.cfg, self.root)
