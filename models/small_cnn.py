"""
檔案功能
--------
小型 CNN（本機 MPS 快速驗證用，不是論文的 WRN-28-2）。三個卷積區塊（Conv-BN-ReLU，後兩個含 2x2 最大池化）
加全域平均池化與線性分類層，約 0.3M 參數，CIFAR-10 上訓練 ~15 epochs 即可明顯記憶訓練集（成員推論有訊號）。
論文設定（WRN-28-2, 100 epochs）需另外的模型檔與 GPU。

使用方式
--------
    from util.registry import build
    m = build({"module": "models.small_cnn", "type": "SmallCNN", "kwargs": {"width": 64, "num_classes": 10}})
    m(torch.zeros(2, 3, 32, 32))    # (2, 10)
"""
import torch
from torch import nn

from models.base import BaseModel


def _block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    """功能：Conv3x3 - BN - ReLU（可選 2x2 最大池化）。"""
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallCNN(BaseModel):
    """
    功能
        32x32 影像分類。
    Pseudo code
        h <- block(3, w) ; block(w, 2w, pool) ; block(2w, 4w, pool) ; block(4w, 4w, pool)
        logits <- Linear(4w, classes)(GlobalAvgPool(h))
    用法
        SmallCNN(width=64, num_classes=10)
    """

    def __init__(self, width: int = 64, num_classes: int = 10):
        super().__init__()
        w = width
        # [對應 Pseudo code: h]
        self.features = nn.Sequential(_block(3, w, False), _block(w, 2 * w, True), _block(2 * w, 4 * w, True), _block(4 * w, 4 * w, True))
        # [對應 Pseudo code: logits]
        self.head = nn.Linear(4 * w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能：前向傳播。用法：model(x)，x 為 (B,3,32,32)。"""
        return self.head(self.features(x).mean(dim=(2, 3)))
