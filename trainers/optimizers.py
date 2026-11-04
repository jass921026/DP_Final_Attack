"""
檔案功能
--------
優化器的建構（由設定檔的名稱與超參數）。除了 torch 內建的 SGD / Adam / RMSprop，另含 optax 風格的 RMSProp
（官方 Balle notebook 使用 optax）。

使用方式
--------
    from trainers.optimizers import build_optimizer
    opt = build_optimizer(model.parameters(), {"name": "sgd", "lr": 0.05, "momentum": 0.9})
    opt = build_optimizer(model.parameters(), {"name": "optax_rmsprop", "lr": 1e-3})
"""
from typing import Any, Dict

import torch


class OptaxRMSprop(torch.optim.Optimizer):
    """
    功能
        複製 optax.rmsprop（官方 notebook 使用）的更新規則。與 torch.optim.RMSprop 的差別：eps 放在平方根「內」。
    Pseudo code (optax.scale_by_rms，decay=0.9, eps=1e-8, 無動量)
        nu <- decay * nu + (1 - decay) * g^2
        theta <- theta - lr * g / sqrt(nu + eps)
    用法
        opt = OptaxRMSprop(model.parameters(), lr=1e-3); opt.zero_grad(); loss.backward(); opt.step()
    """

    def __init__(self, params, lr: float, decay: float = 0.9, eps: float = 1e-8):
        super().__init__(params, dict(lr=lr, decay=decay, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "nu" not in st:
                    st["nu"] = torch.zeros_like(p)          # optax: nu 初始為 0
                nu = st["nu"]
                nu.mul_(g["decay"]).addcmul_(p.grad, p.grad, value=1 - g["decay"])   # [對應 Pseudo code 第一行]
                p.add_(p.grad * torch.rsqrt(nu + g["eps"]), alpha=-g["lr"])          # [對應 Pseudo code 第二行]


def build_optimizer(params, spec: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    功能
        依 {"name": ..., "lr": ..., ...} 建立優化器。name：sgd | adam | torch_rmsprop | optax_rmsprop。
    Pseudo code
        name -> 對應的類別；其餘欄位當作超參數傳入
    用法
        build_optimizer(model.parameters(), {"name": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0})
    """
    kw = {k: v for k, v in spec.items() if k != "name"}
    name = spec["name"]
    if name == "sgd":
        return torch.optim.SGD(params, **kw)
    if name == "adam":
        return torch.optim.Adam(params, **kw)
    if name == "torch_rmsprop":
        return torch.optim.RMSprop(params, **{"alpha": 0.9, "eps": 1e-8, **kw})      # alpha 0.9 = optax decay
    if name == "optax_rmsprop":
        return OptaxRMSprop(params, **kw)
    raise ValueError(f"unknown optimizer {name!r}")
