"""
檔案功能
--------
DataLoader 相關輔助：以索引子集建立 torch DataLoader，並保證「每個 epoch 的打亂順序」可由種子重現。
資料集（datasets，皆為 torch Dataset）資料量小、整份在記憶體中，所以 DataLoader 走「整批索引」的快速路徑：
一次 __getitems__ 取出一個 batch（不逐筆組裝），collate 為恆等函式。

使用方式
--------
    from util.data import make_loader
    loader = make_loader(dataset, indices=keep_idx, batch_size=256, shuffle=True, seed=0)
    for xb, yb in loader: ...          # 張量，形狀 (<=256, ...)
"""
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, Subset


class EpochPermutation(Sampler):
    """
    功能
        每個 epoch 回傳一次新的隨機排列，排列由 np.random.default_rng(seed) 依序產生（第 e 個 epoch 用第 e 次呼叫），
        因此與「手動 rng.permutation(n)」的訓練迴圈得到完全相同的順序。
    Pseudo code
        rng <- default_rng(seed)
        for each epoch:  yield rng.permutation(n)
    用法
        DataLoader(ds, batch_size=32, sampler=EpochPermutation(len(ds), seed=0))
    """

    def __init__(self, n: int, seed: int = 0):
        self.n, self.rng = n, np.random.default_rng(seed)

    def __iter__(self):
        # [對應 Pseudo code: yield rng.permutation(n)]
        return iter(self.rng.permutation(self.n).tolist())

    def __len__(self) -> int:
        return self.n


def _identity(batch):
    """功能：collate 恆等（dataset 的 __getitems__ 已回傳整批張量）。"""
    return batch


def make_loader(dataset: Dataset, indices: Optional[Sequence[int]] = None, batch_size: int = 256,
                shuffle: bool = False, seed: int = 0, drop_last: bool = False) -> DataLoader:
    """
    功能
        建立資料子集的 DataLoader。shuffle=True 時用 EpochPermutation(seed)；否則依序。
    Pseudo code
        ds <- Subset(dataset, indices) 若給了 indices
        sampler <- EpochPermutation(len(ds), seed) 若 shuffle，否則依序
        return DataLoader(ds, batch_size, sampler, collate=恆等)
    用法
        loader = make_loader(train_ds, indices, 256, shuffle=True, seed=3)
    """
    # [對應 Pseudo code: ds, sampler]
    ds = Subset(dataset, list(indices)) if indices is not None else dataset
    sampler = EpochPermutation(len(ds), seed) if shuffle else SequentialSampler(ds)
    # [對應 Pseudo code: return DataLoader]
    # DataLoader 每次開始迭代都會從「全域」torch RNG 抽一個 base seed；給它私有的 generator，避免偷偷改變訓練中其他隨機操作
    # （如資料增強、dropout）的亂數序列，讓結果只由 torch.manual_seed 與 shuffle seed 決定
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=_identity, drop_last=drop_last, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
