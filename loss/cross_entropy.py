"""
檔案功能
--------
交叉熵損失（分類）。Balle 的演算法 A（影子／釋出模型）、一般監督式訓練（RMIA 參考模型）、FL 客戶端都用它。
輸入 logits (n, C)、整數標籤 (n,)。

使用方式
--------
    {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {"reduction": "mean"}}
"""
import torch
import torch.nn.functional as F

from loss.base import BaseLoss


class CrossEntropy(BaseLoss):
    """
    功能
        CE(logits, y) = -log softmax(logits)[y]
    Pseudo code
        per_sample_i <- -log softmax(logits_i)[y_i]
        forward      <- mean_i per_sample_i  或  sum_i per_sample_i
    用法
        CrossEntropy("mean")(logits, y)
    """

    def per_sample(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """功能：逐樣本交叉熵 (n,)。"""
        return F.cross_entropy(output, target, reduction="none")

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """功能：縮減後的交叉熵（直接呼叫 F.cross_entropy，與舊程式的數值運算相同）。"""
        return F.cross_entropy(output, target, reduction=self.reduction)
