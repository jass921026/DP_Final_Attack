"""
檔案功能
--------
torchvision ResNet-18 的轉接器（Robbing the Fed 的目標模型，ImageNet 1000 類）。
固定種子建立；預設隨機初始化（pretrained=False）。官方 breaching 設定預設使用預訓練權重（pretrained: True），
需要連網下載；因此預訓練是明確的選項，不是預設。模型固定在 eval 模式（BatchNorm 用已有的統計量，
與官方 notebook 的 provide_public_buffers 一致，所以逐微批次累加的梯度 = 整批平均梯度）。

使用方式
--------
    from util.registry import build
    net = build({"module": "models.resnet", "type": "ResNet18", "kwargs": {"seed": 0, "pretrained": False}})
    net(torch.zeros(2, 3, 224, 224))   # (2, 1000) logits
"""
import torch
import torchvision

from models.base import BaseModel


class ResNet18(BaseModel):
    """
    功能
        包裝 torchvision.models.resnet18，固定種子、固定 eval 模式。
    Pseudo code
        torch.manual_seed(seed); net <- resnet18(weights = IMAGENET1K_V1 if pretrained else None); net.eval()
    用法
        ResNet18(num_classes=1000, seed=0, pretrained=False)
    """

    def __init__(self, num_classes: int = 1000, seed: int = 0, pretrained: bool = False):
        super().__init__()
        # [對應 Pseudo code] 固定種子建立；預訓練時載入 torchvision 權重
        torch.manual_seed(seed)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.resnet18(weights=weights, num_classes=num_classes if not pretrained else 1000)
        self.net.eval()

    def train(self, mode: bool = True):
        """功能：永遠保持 eval（BN 用固定統計量）。用法：由 nn.Module 自動呼叫，無需手動使用。"""
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能：前向傳播。用法：model(x)，x 為正規化後的 (B,3,224,224)。"""
        return self.net(x)
