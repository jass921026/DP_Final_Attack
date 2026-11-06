"""
檔案功能
--------
WRN-28-2（Zagoruyko & Komodakis）：RMIA 論文在 CIFAR-10 使用的模型（約 1.5M 參數）。
pre-activation 殘差區塊，3 個階段（16*k, 32*k, 64*k 通道），每階段 (depth-4)/6 個區塊。

使用方式
--------
    from util.registry import build
    m = build({"module": "models.wrn", "type": "WideResNet", "kwargs": {"depth": 28, "widen": 2, "num_classes": 10}})
    m(torch.zeros(2, 3, 32, 32))    # (2, 10)
"""
import torch
from torch import nn

from models.base import BaseModel


class _Block(nn.Module):
    """功能：BN-ReLU-Conv3x3-BN-ReLU-Conv3x3 加捷徑（通道或步幅改變時用 1x1 卷積）。"""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm2d(cin), nn.BatchNorm2d(cout)
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.short = None if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = torch.relu(self.bn1(x))
        y = self.c1(o)
        y = self.c2(torch.relu(self.bn2(y)))
        return y + (x if self.short is None else self.short(o))


class WideResNet(BaseModel):
    """
    功能
        32x32 影像分類。
    Pseudo code
        h <- Conv3x3(3, 16) ; 三個階段各 n=(depth-4)/6 個 Block（步幅 1, 2, 2）
        logits <- Linear(64k, classes)(GlobalAvgPool(ReLU(BN(h))))
    用法
        WideResNet(depth=28, widen=2, num_classes=10)
    """

    def __init__(self, depth: int = 28, widen: int = 2, num_classes: int = 10):
        super().__init__()
        n, k = (depth - 4) // 6, widen
        widths = [16, 16 * k, 32 * k, 64 * k]
        self.conv = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        layers = []
        for i, stride in enumerate((1, 2, 2)):
            for j in range(n):
                layers.append(_Block(widths[i] if j == 0 else widths[i + 1], widths[i + 1], stride if j == 0 else 1))
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(widths[3])
        self.head = nn.Linear(widths[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能：前向傳播。用法：model(x)，x 為 (B,3,32,32)。"""
        h = torch.relu(self.bn(self.blocks(self.conv(x))))
        return self.head(h.mean(dim=(2, 3)))
