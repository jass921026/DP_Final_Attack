"""
檔案功能
--------
影像重建用的迴歸損失：MAE（L1）與 MSE（L2），可先對輸出套 sigmoid（Balle 的重建網路輸出 logits，
影像值在 [0,1]）。Balle 論文 Sec. V-A(c)：MNIST 的訓練目標為 MAE + MSE（以 loss.combine.WeightedSum 組合）。
逐元素誤差在所有元素上取平均（mean）或加總（sum）。

使用方式
--------
    {"module": "loss.regression", "type": "MAE", "kwargs": {"activation": "sigmoid"}}
    {"module": "loss.regression", "type": "MSE", "kwargs": {"activation": "sigmoid"}}
"""
import torch

from loss.base import BaseLoss


class _ElementwiseRegression(BaseLoss):
    """
    功能
        逐元素誤差的共同實作；子類別只給 _error(pred, target)。
    Pseudo code
        pred <- sigmoid(output) 若 activation == "sigmoid"，否則 output
        per_sample_i <- mean_d _error(pred_i, target_i)
        forward      <- mean（全部元素）或 sum
    用法
        見 MAE / MSE
    """

    def __init__(self, reduction: str = "mean", activation: str = "none"):
        super().__init__(reduction)
        assert activation in ("none", "sigmoid")
        self.activation = activation

    def _error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _pred(self, output: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(output) if self.activation == "sigmoid" else output

    def per_sample(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """功能：每個樣本的（維度平均）誤差 (n,)。"""
        return self._error(self._pred(output), target).flatten(1).mean(dim=1)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """功能：縮減後的純量（mean = 所有元素的平均，與舊程式 mean(|.|)、mean((.)^2) 相同）。"""
        e = self._error(self._pred(output), target)
        return e.mean() if self.reduction == "mean" else e.sum()


class MAE(_ElementwiseRegression):
    """功能：平均絕對誤差 |pred - target|。用法：MAE(activation="sigmoid")。"""

    def _error(self, pred, target):
        return (pred - target).abs()


class MSE(_ElementwiseRegression):
    """功能：均方誤差 (pred - target)^2。用法：MSE(activation="sigmoid")。"""

    def _error(self, pred, target):
        return (pred - target) ** 2
