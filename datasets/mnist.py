"""
檔案功能
--------
載入 MNIST（train + test 共 70,000 張），轉成 float32、範圍 [0,1]、展平成 784 維。
資料來源是 Hugging Face 的 parquet（ylecun/mnist），第一次下載後存成本地 npz 快取；之後不再連網。
順序固定為「train 在前、test 在後」，切分由 util.splits 以索引決定，所以本檔不做任何隨機切分。
對應 Balle 2022 Table VI：MNIST，28x28 灰階，共 70K。

使用方式
--------
    from util.registry import build
    ds = build({"module": "datasets.mnist", "type": "MNIST", "kwargs": {"cache_dir": "data_cache"}})
    ds.x      # (70000, 784) float32
    ds.y      # (70000,) int64
    len(ds)   # 70000
    ds.get(idx)  # (x[idx], y[idx])；ds 是 torch Dataset，可用 util.data.make_loader 建 DataLoader
"""
import io
import os
from typing import Tuple

import numpy as np

from datasets.base import BaseDataset


class MNIST(BaseDataset):
    """
    功能
        提供 x / y 陣列與依索引取樣的 get()。這是資料集轉接器的最小介面（__len__, x, y, get）。
    Pseudo code
        1. 若快取 npz 存在 -> 讀取並回傳
        2. 否則從 Hugging Face 下載 train / test parquet
        3. 每張 PNG 解碼為 28x28 uint8，除以 255 -> float32，展平 784
        4. train 與 test 串接（train 在前），存成快取
    用法
        MNIST(cache_dir="data_cache")
    """

    HF_REPO = "ylecun/mnist"

    def __init__(self, cache_dir: str = "data_cache"):
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "mnist_70k.npz")
        # [對應 步驟 1] 優先讀快取
        if os.path.isfile(path):
            z = np.load(path)
            super().__init__(z["x"], z["y"])
            return
        # [對應 步驟 2-4] 下載、解碼、串接、存快取
        xs, ys = [], []
        for split in ("train", "test"):
            x, y = self._load_split(split)
            xs.append(x)
            ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
        np.savez(path, x=x, y=y)
        super().__init__(x, y)

    @classmethod
    def _load_split(cls, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """功能：下載並解碼單一 split（步驟 2-3）。"""
        import pandas as pd
        from huggingface_hub import hf_hub_download
        from PIL import Image
        # [對應 步驟 2] 下載 parquet
        p = hf_hub_download(cls.HF_REPO, f"mnist/{split}-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(p)
        # [對應 步驟 3] PNG -> 28x28 uint8 -> float32 [0,1] -> 784
        imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("L")) for d in df["image"]])
        x = (imgs.astype(np.float32) / 255.0).reshape(len(imgs), -1)
        return x, df["label"].to_numpy().astype(np.int64)
