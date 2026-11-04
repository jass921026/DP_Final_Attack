"""
檔案功能
--------
載入 CIFAR-10（Hugging Face uoft-cs/cifar10 parquet，公開資料集），存成本地 npz 快取。
split="train"（5 萬）或 "test"（1 萬）；x 為 (n,3,32,32) float32，已正規化為 (x-0.5)/0.5（與官方 ml_privacy_meter
的 CIFAR-10 轉換相同：ToTensor + Normalize(0.5, 0.5)），y 為 int64。
RMIA 對應：官方以 train（5 萬）為「資料集」（成員／非成員的來源），test（1 萬）為「母體」（population，提供 z）。

使用方式
--------
    from util.registry import build
    train = build({"module": "datasets.cifar10", "type": "CIFAR10", "kwargs": {"cache_dir": "data_cache"}})
    test = build(spec, split="test")            # 以 overrides 取 test split
"""
import io
import os

import numpy as np

from datasets.base import BaseDataset


class CIFAR10(BaseDataset):
    """
    功能
        提供 train / test 陣列。
    Pseudo code
        1. 快取 npz 存在 -> 讀取
        2. 否則下載 train / test parquet，逐張解碼 PNG -> uint8 (32,32,3) -> (3,32,32)
        3. float32 / 255，再 (x-0.5)/0.5；存快取
    用法
        CIFAR10(cache_dir="data_cache", split="train")
    """
    REPO = "uoft-cs/cifar10"

    def __init__(self, cache_dir: str = "data_cache", split: str = "train"):
        assert split in ("train", "test")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "cifar10.npz")
        # [對應 步驟 2-3] 快取不存在時下載並存檔
        if not os.path.isfile(path):
            out = {}
            for sp in ("train", "test"):
                x, y = self._load_split(sp)
                out[f"x_{sp}"], out[f"y_{sp}"] = x, y
            np.savez(path, **out)
        # [對應 步驟 1] 讀取指定 split
        z = np.load(path)
        super().__init__(z[f"x_{split}"], z[f"y_{split}"])

    @classmethod
    def _load_split(cls, split: str):
        """功能：下載並解碼單一 split（步驟 2-3）。"""
        import pandas as pd
        from huggingface_hub import hf_hub_download
        from PIL import Image
        df = pd.read_parquet(hf_hub_download(cls.REPO, f"plain_text/{split}-00000-of-00001.parquet", repo_type="dataset"))
        imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df["img"]])
        x = (imgs.astype(np.float32) / 255.0 - 0.5) / 0.5
        return x.transpose(0, 3, 1, 2).copy(), df["label"].to_numpy().astype(np.int64)
