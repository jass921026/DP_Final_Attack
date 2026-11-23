"""
檔案功能
--------
DP-SGD 訓練（Poisson 抽樣 + 逐樣本裁剪 + 高斯雜訊，Opacus 的 GradSampleModule / DPOptimizer 做逐樣本梯度與裁剪），
並用 RDP accountant 由目標 (epsilon, delta) 反推噪音倍率、回報實際達到的 epsilon。
Opacus 要求模型不含 BatchNorm（用 GroupNorm）、ReLU 不可 in-place。

使用方式
--------
    from trainers.dp_sgd import train_dp_sgd
    info = train_dp_sgd(model, x, y, epochs=20, lr=0.05, momentum=0.9, batch_size=256, clip=1.0,
                        epsilon=100, delta=1e-5, device="mps", seed=0, augment=random_hflip)
    # model 被就地訓練；info = {"sigma", "epsilon", "steps", "sample_rate"}
"""
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer


def train_dp_sgd(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, momentum: float, batch_size: int,
                 clip: float, epsilon: float, delta: float, device: str, seed: int, augment: Optional[Callable] = None) -> Dict[str, Any]:
    """
    功能
        在 (x, y) 上做 DP-SGD。每步以機率 q = batch_size / N 獨立抽每筆樣本（Poisson 抽樣），裁剪每筆梯度到 clip，
        加 N(0, (sigma*clip)^2)，除以期望批量。總步數 = int(epochs / q)。
    Pseudo code
        q <- B / N;  T <- int(epochs / q);  sigma <- RDP 反推使 (epsilon, delta) 成立
        for t in 1..T:
            S <- {i : u_i < q}（獨立 Bernoulli）
            g <- (sum_{i in S} clip_C(grad l_i) + N(0, sigma^2 C^2 I)) / B
            theta <- SGD(momentum)(g)
    用法
        見檔頭。epsilon=None 或 inf 時 sigma=0（只裁剪、仍用 Poisson 抽樣）。
    """
    N = len(x)
    q = batch_size / N
    steps = int(epochs / q)
    sigma = 0.0 if (epsilon is None or epsilon == float("inf")) else get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta, sample_rate=q, steps=steps, accountant="rdp", epsilon_tolerance=0.01)
    model = model.to(device).train()
    gm = GradSampleModule(model)
    opt = DPOptimizer(torch.optim.SGD(gm.parameters(), lr=lr, momentum=momentum), noise_multiplier=sigma, max_grad_norm=clip,
                      expected_batch_size=batch_size)
    gen = torch.Generator().manual_seed(seed)
    x, y = x.to(device), y.to(device)
    for _ in range(steps):
        mask = torch.rand(N, generator=gen) < q
        if not mask.any():
            continue
        mask_dev = mask.to(device)                  # 同一個 mask 只搬一次（原來分別對 x、y 各搬一次，數值不變，省一次多餘的搬移）
        xb, yb = x[mask_dev], y[mask_dev]
        if augment is not None:
            xb = augment(xb)
        opt.zero_grad()
        F.cross_entropy(gm(xb), yb).backward()
        opt.step()
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)] if sigma > 0 else []
    eps = acc.get_epsilon(delta) if sigma > 0 else float("inf")
    gm._module.eval()
    return {"sigma": float(sigma), "epsilon": float(eps), "steps": steps, "sample_rate": q}
