"""
檔案功能
--------
損失函數的母類 BaseLoss：所有 loss/ 下的損失都繼承它（torch.nn.Module）。與 SUMMIT (code_submit_v2/loss/) 一樣，
trainer 只做 `loss = loss_fn(output, target)`，損失的定義全部在這個資料夾。
介面：
    forward(output, target)     依 self.reduction 回傳純量（"mean" 或 "sum"）
    per_sample(output, target)  回傳形狀 (n,) 的逐樣本損失（trainer 需要自行組合時使用，例如 Balle 的
                                「D- 的總和 + 一筆 z」再除以 n+1，或逐樣本梯度裁剪）

使用方式（新增損失）
--------------------
    class MyLoss(BaseLoss):
        def per_sample(self, output, target): ...      # (n,)
        def forward(self, output, target): ...         # 純量（需與 per_sample 一致）
    # 設定檔：{"module": "loss.my_loss", "type": "MyLoss", "kwargs": {"reduction": "mean"}}
"""
from torch import nn


class BaseLoss(nn.Module):
    """
    功能
        損失的共同母類；持有 reduction 並檢查合法值。
    Pseudo code
        forward     = reduce(per_sample)
        per_sample  = 逐樣本損失 (n,)
    用法
        loss_fn = SomeLoss(reduction="mean");  loss_fn(logits, y);  loss_fn.per_sample(logits, y)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum"), f"unsupported reduction {reduction!r}"
        self.reduction = reduction

    def per_sample(self, output, target):
        """功能：逐樣本損失 (n,)。子類別必須實作。"""
        raise NotImplementedError

    def forward(self, output, target):
        """功能：依 reduction 縮減後的純量損失。子類別必須實作。"""
        raise NotImplementedError
