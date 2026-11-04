"""
檔案功能
--------
可設定的多層感知機（MLP）。同一個類別同時用於
    (1) Balle 2022 的 released / shadow 模型：MNIST，1 個隱藏層、寬度 10、ELU，約 8K 參數（Table VI 的 θ, θ̄）
    (2) Balle 2022 的 reconstructor φ：2 個隱藏層、寬度 1000、ReLU，輸出影像（Table VI 的 φ）
架構以設定檔指定，不寫死在程式裡。

使用方式
--------
    from util.registry import build
    released = build({"module": "models.mlp", "type": "MLP",
                      "kwargs": {"in_dim": 784, "hidden": [10], "out_dim": 10, "activation": "elu"}})
    # 參數展平：models.mlp.flatten_params(model) -> 1D tensor（Balle 攻擊把各層參數串接後餵給 φ）
"""
from typing import List

import torch
from torch import nn

from models.base import BaseModel

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}


class MLP(BaseModel):
    """
    功能
        in_dim -> hidden[0] -> ... -> hidden[-1] -> out_dim 的全連接網路，隱藏層之間使用指定的啟動函數，
        輸出層為線性（輸出 logits；若要影像請在外部接 sigmoid）。
    Pseudo code (標準 MLP，非論文專有)
        h_0 <- x
        for l in 1..L:  h_l <- act(W_l h_{l-1} + b_l)
        return W_{L+1} h_L + b_{L+1}
    用法
        MLP(in_dim=784, hidden=[10], out_dim=10, activation="elu")
        init="haiku"：沿用官方 notebook 的 Haiku 預設初始化（論文未規定初始化方式，故以官方實作為準）：
            權重 ~ 截斷常態(±2 標準差) 並縮放使標準差 = 1/sqrt(fan_in)，偏置 = 0。init="torch" 為 PyTorch 預設。
    """

    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, activation: str = "elu", init: str = "torch"):
        super().__init__()
        # [對應 Pseudo code] 依 hidden 逐層堆疊 Linear + 啟動函數
        act = _ACTIVATIONS[activation]
        dims = [in_dim] + list(hidden)
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act()]
        # [對應 Pseudo code 最後一行] 線性輸出層
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)
        # [對應 官方 Haiku 預設初始化] hk.Linear: w_init = TruncatedNormal(stddev=1/sqrt(fan_in)), b_init = zeros
        if init == "haiku":
            for m in self.net:
                if isinstance(m, nn.Linear):
                    fan_in = m.weight.shape[1]
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
                    # Haiku 以 0.87962566 (= 截斷於 ±2 的標準常態之標準差) 修正，使最終標準差恰為 1/sqrt(fan_in)
                    m.weight.data.mul_(1.0 / (fan_in ** 0.5) / 0.87962566103423978)
                    nn.init.zeros_(m.bias)
        elif init != "torch":
            raise ValueError(f"unknown init {init!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能：前向傳播。用法：model(x)，x 形狀 (batch, in_dim)。"""
        return self.net(x)


def flatten_params(model: nn.Module) -> torch.Tensor:
    """
    功能
        把模型所有參數依 named_parameters 順序串成一維向量（Balle 2022 Sec. V-A(c)：
        shadow model parameters across layers are flattened and concatenated together）。
    Pseudo code
        return concat_{p in parameters(model)} vec(p)
    用法
        theta = flatten_params(model)
    """
    # [對應 Pseudo code]
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
