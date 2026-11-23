"""
檔案功能
--------
聯邦學習中「誠實客戶端」的本地計算：用本地資料算出要回傳給伺服器的梯度（FedSGD，一次全批梯度）。
這是 Robbing the Fed 的威脅模型裡，被攻擊者看到的東西：伺服器下發（被竄改的）模型，客戶端照常計算梯度並回傳。
FedAvg：本地多步 SGD，回傳參數差 (p_local - p_server)（官方 breaching UserMultiStep 的作法）。

使用方式
--------
    from trainers.fl_client import fedsgd_gradient
    grads = fedsgd_gradient(model, loss_fn, x_norm, y, micro_batch=8)   # list，順序同 model.parameters()；loss_fn 來自 loss/
    # 或以類別：FedSGDClient(model, loss_fn, x, y).fit() / FedAvgClient(model, loss_fn, x, y, steps, per_step, lr).fit()
"""
from typing import List

import torch

from trainers.base import BaseTrainer


def fedsgd_gradient(model: torch.nn.Module, loss_fn: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, micro_batch: int = 8) -> List[torch.Tensor]:
    """
    功能
        客戶端在其本地批次上的平均交叉熵梯度。為了省記憶體，分成微批次累加；模型在 eval 模式（BN 固定）
        時，微批次加權累加 = 整批平均梯度。
    Pseudo code (FedSGD，論文 Sec. 3 威脅模型)
        g <- grad_theta  1/n * sum_i l(f_theta(x_i), y_i)        # 一次全批梯度
        回傳 g
        （實作：g <- sum_{chunks c} (|c|/n) * grad_theta mean_{i in c} l，數學上相同）
    用法
        grads = fedsgd_gradient(model, x_norm, y, micro_batch=8)
    """
    n = len(x)
    grads = None
    # [對應 Pseudo code: 實作那一行] 逐微批次累加加權梯度
    for i in range(0, n, micro_batch):
        xb, yb = x[i:i + micro_batch], y[i:i + micro_batch]
        loss = loss_fn(model(xb), yb) * (len(xb) / n)
        g = torch.autograd.grad(loss, list(model.parameters()))
        grads = list(g) if grads is None else [a + b for a, b in zip(grads, g)]
    return grads


def dp_clipped_gradient(model: torch.nn.Module, loss_fn: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, clip: float):
    """
    功能
        DP-FedSGD 客戶端的「裁剪」部分：逐筆梯度裁剪到範數 <= clip，再取平均（雜訊由呼叫端加）。
        回傳 (平均的裁剪梯度 list, 逐筆原始梯度範數 (n,))。add/remove 鄰接下，總和的敏感度 = clip。
    Pseudo code
        for i in 1..n:
            g_i <- grad l(f(x_i), y_i)
            c_i <- min(1, clip / ||g_i||)
        回傳 (1/n) * sum_i c_i * g_i
    用法
        g, norms = dp_clipped_gradient(model, loss_fn, x_norm, y, clip=1.0)    # 模型須在 eval 模式（BN 固定，樣本互相獨立）
    """
    n, params = len(x), list(model.parameters())
    acc = [torch.zeros_like(p) for p in params]
    norms = torch.zeros(n)
    for i in range(n):                                     # [對應 Pseudo code: for i] 每筆單獨反向，累加裁剪後的梯度
        g = torch.autograd.grad(loss_fn(model(x[i:i + 1]), y[i:i + 1]), params)
        norm = torch.sqrt(sum((gi.double() ** 2).sum() for gi in g)).item()
        norms[i] = norm
        c = min(1.0, clip / (norm + 1e-12))
        for a, gi in zip(acc, g):
            a.add_(gi, alpha=c / n)
    return acc, norms


def fedavg_update(model: torch.nn.Module, loss_fn: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, per_step: int, lr: float) -> List[torch.Tensor]:
    """
    功能
        FedAvg 客戶端：在本地資料上做 `steps` 步 SGD（每步用不同的 per_step 筆資料），回傳與伺服器模型的參數差。
        對應官方 breaching UserMultiStep.compute_local_updates 與論文 Sec. 4「Multiple local updates」。
    Pseudo code
        theta_0 <- 伺服器下發的參數
        for t in 0..steps-1:
            batch_t <- 資料第 [t*per_step, (t+1)*per_step) 筆（超過資料量時取模）
            theta <- theta - lr * grad CE(batch_t)                     # 原生 SGD，無動量、無重量衰減
        回傳 theta - theta_0                                            # = -lr * sum_t grad_t，攻擊只看到這個
    用法
        update = fedavg_update(model, loss_fn, x_norm, y, steps=8, per_step=8, lr=1e-4)    # 呼叫後 model 參數會被還原
    """
    # [對應 Pseudo code: theta_0] 備份，結束後還原，讓同一個模型可重複使用
    theta0 = [p.detach().clone() for p in model.parameters()]
    n, seen = len(x), 0
    # [對應 Pseudo code: for t ...] 本地 SGD
    for _ in range(steps):
        idx = torch.arange(seen, seen + per_step) % n
        seen = (seen + per_step) % n
        loss = loss_fn(model(x[idx]), y[idx])
        g = torch.autograd.grad(loss, list(model.parameters()))
        with torch.no_grad():
            for p, gi in zip(model.parameters(), g):
                p -= lr * gi
    # [對應 Pseudo code: 回傳 theta - theta_0] 並還原模型
    update = [(p.detach() - p0).clone() for p, p0 in zip(model.parameters(), theta0)]
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), theta0):
            p.copy_(p0)
    return update


class FedSGDClient(BaseTrainer):
    """
    功能
        FedSGD 客戶端的 BaseTrainer 包裝：fit() 回傳本地批次的平均損失梯度（list，順序同 model.parameters()）。
    用法
        grads = FedSGDClient(model, loss_fn, x_norm, y, micro_batch=8).fit()
    """

    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, micro_batch: int = 8):
        self.model, self.loss_fn, self.x, self.y, self.micro_batch = model, loss_fn, x, y, micro_batch

    def fit(self) -> List[torch.Tensor]:
        """功能：呼叫 fedsgd_gradient。"""
        return fedsgd_gradient(self.model, self.loss_fn, self.x, self.y, self.micro_batch)


class FedAvgClient(BaseTrainer):
    """
    功能
        FedAvg 客戶端的 BaseTrainer 包裝：fit() 回傳多步本地 SGD 後與伺服器模型的參數差。
    用法
        update = FedAvgClient(model, loss_fn, x_norm, y, steps=8, per_step=8, lr=1e-4).fit()
    """

    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, per_step: int, lr: float):
        self.model, self.loss_fn, self.x, self.y, self.steps, self.per_step, self.lr = model, loss_fn, x, y, steps, per_step, lr

    def fit(self) -> List[torch.Tensor]:
        """功能：呼叫 fedavg_update。"""
        return fedavg_update(self.model, self.loss_fn, self.x, self.y, self.steps, self.per_step, self.lr)
