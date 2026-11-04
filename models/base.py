"""
檔案功能
--------
模型的母類 BaseModel：所有 models/ 下的模型都繼承它（它是 torch.nn.Module）。
提供各攻擊與訓練器共用的小工具：num_params()、flat_params()（依 parameters() 順序串接成一維向量；
Balle 攻擊把各層參數展平餵給重建網路）。子類別只需要實作 __init__ 與 forward。

使用方式（新增模型）
--------------------
    class MyNet(BaseModel):
        def __init__(self, ...): super().__init__(); ...
        def forward(self, x): ...
    # 設定檔：{"module": "models.my_net", "type": "MyNet", "kwargs": {...}}
"""
import torch
from torch import nn


class BaseModel(nn.Module):
    """
    功能
        nn.Module 加上參數統計與展平工具。
    Pseudo code
        num_params  <- sum_p |p|
        flat_params <- concat_{p in parameters()} vec(p)
    用法
        model.num_params();  model.flat_params()
    """

    def num_params(self) -> int:
        """功能：參數總數。"""
        return sum(p.numel() for p in self.parameters())

    def flat_params(self) -> torch.Tensor:
        """功能：依 parameters() 順序串接成一維向量（不帶梯度）。"""
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])
