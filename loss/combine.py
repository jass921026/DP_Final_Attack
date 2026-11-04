"""
檔案功能
--------
把多個損失加權相加（對應 SUMMIT 的 loss/combine_loss.py）。每一項是一個 {"module","type","kwargs"} 設定，
由 util.registry.build 建構，因此任何 loss/ 下的損失都能組合。Balle MNIST 的 MAE + MSE 就是兩項、權重 1。

使用方式
--------
    {"module": "loss.combine", "type": "WeightedSum", "kwargs": {"terms": [
        {"weight": 1.0, "loss": {"module": "loss.regression", "type": "MAE", "kwargs": {"activation": "sigmoid"}}},
        {"weight": 1.0, "loss": {"module": "loss.regression", "type": "MSE", "kwargs": {"activation": "sigmoid"}}}]}}
"""
from typing import Any, Dict, List

import torch
from torch import nn

from util.registry import build
from loss.base import BaseLoss


class WeightedSum(BaseLoss):
    """
    功能
        L = sum_k w_k * L_k。各項自己的 reduction 決定縮減方式（本類別的 reduction 不再額外作用）。
    Pseudo code
        forward    <- sum_k w_k * L_k(output, target)
        per_sample <- sum_k w_k * L_k.per_sample(output, target)
    用法
        WeightedSum(terms=[{"weight": 1, "loss": spec_mae}, {"weight": 1, "loss": spec_mse}])
    """

    def __init__(self, terms: List[Dict[str, Any]]):
        super().__init__("mean")
        # [對應 Pseudo code: L_k 與 w_k]
        self.weights = [float(t.get("weight", 1.0)) for t in terms]
        self.terms = nn.ModuleList([build(t["loss"]) for t in terms])

    def per_sample(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """功能：加權的逐樣本損失 (n,)。"""
        return sum(w * t.per_sample(output, target) for w, t in zip(self.weights, self.terms))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """功能：加權總和（先算第一項再依序相加，與舊程式 mae + mse 的運算順序相同）。"""
        out = None
        for w, t in zip(self.weights, self.terms):
            v = t(output, target) * w if w != 1.0 else t(output, target)
            out = v if out is None else out + v
        return out
