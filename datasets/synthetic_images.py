"""
檔案功能
--------
小型合成「影像」資料集，只用於單元測試與流程煙霧測試（不需要網路、幾毫秒產生）。
每個類別有一個固定的原型向量，樣本 = 原型 + 高斯雜訊，截斷到 [0,1]。介面與 datasets.mnist.MNIST 相同。

使用方式
--------
    ds = build({"module": "datasets.synthetic_images", "type": "SyntheticImages",
                "kwargs": {"n": 600, "dim": 16, "n_classes": 4, "seed": 0}})
"""
import numpy as np

from datasets.base import BaseDataset


class SyntheticImages(BaseDataset):
    """
    功能
        提供 x (n, dim) float32 in [0,1]、y (n,) int64、get(idx)、__len__。
    Pseudo code (自訂)
        proto_c <- Uniform(0,1)^dim  for each class c
        y_i <- i mod n_classes;  x_i <- clip(proto_{y_i} + N(0, 0.1^2), 0, 1)
    用法
        SyntheticImages(n=600, dim=16, n_classes=4, seed=0)
    """

    def __init__(self, n: int = 600, dim: int = 16, n_classes: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        # [對應 Pseudo code: proto_c]
        proto = rng.uniform(0, 1, size=(n_classes, dim))
        # [對應 Pseudo code: y_i 與 x_i]
        y = (np.arange(n) % n_classes).astype(np.int64)
        x = np.clip(proto[y] + rng.normal(0, 0.1, size=(n, dim)), 0, 1).astype(np.float32)
        super().__init__(x, y)
