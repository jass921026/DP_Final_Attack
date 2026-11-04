"""
檔案功能
--------
資料集的母類 BaseDataset：所有 datasets/ 下的資料集都繼承它。它是 torch.utils.data.Dataset，
內部以 numpy 陣列 x、y 保存整份資料，提供
    - 標準的 Dataset 介面：__len__、__getitem__（單筆）、__getitems__（整批，DataLoader 的快速路徑）
    - arrays(idx)：取得 numpy 陣列（攻擊端如 NN oracle、imprint 需要原始陣列）
    - get(idx)：arrays 的別名（舊介面）
來源資料（Hugging Face parquet 等）由子類別在 _load 或 __init__ 中讀取、轉成 (x, y) 後呼叫 super().__init__(x, y)；
Hugging Face 只是「來源與快取」，對外介面一律是 torch Dataset。

使用方式（新增資料集）
----------------------
    class MyData(BaseDataset):
        def __init__(self, cache_dir="data_cache", split="train"):
            x, y = ...                       # 載入並轉成 numpy：x float32 (n, ...)，y int64 (n,)
            super().__init__(x, y)
    # 設定檔：{"module": "datasets.my_data", "type": "MyData", "kwargs": {...}}
"""
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    功能
        整份資料在記憶體中的 torch Dataset。
    Pseudo code
        存 x (n, ...) 與 y (n,)
        __getitem__(i)       -> (tensor(x[i]), tensor(y[i]))
        __getitems__(list)   -> (tensor(x[list]), tensor(y[list]))      # 整批，一次索引
        arrays(idx)          -> (x[idx], y[idx]) 或全部
    用法
        ds = BaseDataset(x, y);  x_b, y_b = ds.arrays(np.arange(8))
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y), "x and y must have the same length"
        self.x, self.y = x, y

    def __len__(self) -> int:
        """功能：樣本總數。"""
        return len(self.y)

    def __getitem__(self, i: int):
        """功能：單筆 (x, y) 張量。"""
        return torch.as_tensor(self.x[i]), torch.as_tensor(self.y[i])

    def __getitems__(self, idx):
        """功能：整批 (x, y) 張量（DataLoader 會優先呼叫；配合 util.data 的恆等 collate）。"""
        idx = np.asarray(idx)
        return torch.as_tensor(self.x[idx]), torch.as_tensor(self.y[idx])

    def arrays(self, idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """功能：取得 numpy 陣列。用法：x, y = ds.arrays(splits['fixed'])；不給 idx 則回傳全部。"""
        return (self.x, self.y) if idx is None else (self.x[idx], self.y[idx])

    def get(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """功能：arrays(idx) 的別名（舊介面）。"""
        return self.arrays(idx)
