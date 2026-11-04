"""
檔案功能
--------
把資料集的索引確定性地切成互不相交的幾份（例如 Balle 2022 的 固定集 D-、影子集 D-bar、測試目標、模型效能評估集）。
確定性 = 只由 (n_total, sizes, seed) 決定，所以切分可以只存下這三個值就重建，也能寫進 RunRecord 供日後復用。
這也是避免 RMIA 實驗中「審計樣本子集與快取訊號錯位」問題的基礎：子集由明確的索引定義，而不是隱含的亂數狀態。

使用方式
--------
    from util.splits import make_disjoint_splits, index_hash
    s = make_disjoint_splits(70000, {"eval": 1000, "test_targets": 1000, "fixed": 1000, "shadow": 10000}, seed=42)
    s["fixed"]            # np.ndarray of indices
    index_hash(s["fixed"])  # 用於快取鍵
"""
import hashlib
from typing import Dict

import numpy as np


def make_disjoint_splits(n_total: int, sizes: Dict[str, int], seed: int) -> Dict[str, np.ndarray]:
    """
    功能
        依序從一個隨機排列中切出互不相交的索引區段。
    Pseudo code (本框架自訂；對應 Balle 2022 Sec. V-A(a)「將資料集切成三個互不相交的部分」
                 與官方 notebook load_data() 的切法)
        1. perm <- RNG(seed).permutation(n_total)
        2. 依 sizes 的給定順序，從 perm 的前端依次取走 sizes[name] 個索引
        3. 若總和超過 n_total 則報錯
    用法
        splits = make_disjoint_splits(70000, {"fixed": 1000, "shadow": 10000}, seed=0)
    """
    # [對應 步驟 3] 檢查大小總和
    total = sum(sizes.values())
    if total > n_total:
        raise ValueError(f"requested {total} indices but only {n_total} available")
    # [對應 步驟 1] 確定性排列
    perm = np.random.default_rng(seed).permutation(n_total)
    # [對應 步驟 2] 依序切段
    out, start = {}, 0
    for name, size in sizes.items():
        out[name] = perm[start:start + size].copy()
        start += size
    return out


def index_hash(idx: np.ndarray) -> str:
    """
    功能
        回傳索引陣列的內容雜湊（前 16 碼），用作快取鍵的一部分。
    Pseudo code
        1. 轉成 int64 連續陣列
        2. sha256(bytes) 取前 16 個十六進位字元
    用法
        key = index_hash(splits["fixed"])
    """
    # [對應 步驟 1]
    a = np.ascontiguousarray(idx, dtype=np.int64)
    # [對應 步驟 2]
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]
