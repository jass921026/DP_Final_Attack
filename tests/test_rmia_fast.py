"""
檔案功能
--------
RMIA 加速路徑（attacks/rmia.model_signals 的 move 參數、attacks/rmia_experiment.build_signals 的 cfg.train.fast）測試：
    1. move=False（假設模型已在 device、呼叫端自己管理搬移）與預設 move=True 的訊號逐位元相同（batch size 相同時，
       少了搬移這件事本身不影響任何數值）
    2. build_signals 在 fast=True / False 下算出數值上一致的 signals / pop_signals（RunRecord 的 spec 不含 fast，
       兩次跑分別用不同的 root 避免互相讀到快取；fast 預設用更大的推論 batch size，矩陣乘法加總順序不同會有
       float32 量級的捨入差異，不是逐位元相同）
不依賴 CIFAR10 下載：用合成的小型資料集與 MLP。
使用方式：pytest tests/test_rmia_fast.py -q
"""
import shutil
import tempfile

import numpy as np
import torch

import datasets.base as dbase
from attacks.rmia import model_signals
from attacks.rmia_experiment import build_signals
from util.registry import build

MODEL_SPEC = {"module": "models.mlp", "type": "MLP", "kwargs": {"in_dim": 8, "hidden": [16], "out_dim": 3, "activation": "elu", "init": "haiku"}}


class _ToyDS(dbase.BaseDataset):
    """供測試用的最小資料集（build_signals 需要 build(cfg['data']) 與 build(cfg['data'], split='test') 都能用）。"""

    def __init__(self, split: str = "train"):
        rng = np.random.default_rng(0 if split == "train" else 1)
        n = 300 if split == "train" else 60
        x = rng.random((n, 8)).astype(np.float32)
        y = rng.integers(0, 3, n)
        super().__init__(x, y)


dbase._ToyDS = _ToyDS       # build() 用 importlib + getattr，掛在 datasets.base 上即可被設定檔的 {"module": "datasets.base", "type": "_ToyDS"} 找到


def test_move_false_matches_move_true():
    torch.manual_seed(0)
    model = build(MODEL_SPEC).eval()
    ds = _ToyDS()
    a = model_signals(model, ds, np.arange(50), device="cpu", batch_size=16, move=True)
    b = model_signals(model, ds, np.arange(50), device="cpu", batch_size=16, move=False)   # 假設已在 device（這裡 device=cpu）
    assert np.array_equal(a, b)


def test_fast_build_signals_matches_default():
    cfg_base = {"name": "t", "data": {"module": "datasets.base", "type": "_ToyDS", "kwargs": {}}, "model": MODEL_SPEC,
                "train": {"epochs": 1, "lr": 0.1, "batch_size": 32, "flip": False},
                "models": {"num_pairs": 1, "seed": 0}, "audit": {"size": 40, "seed": 1}, "population": {"size": 20, "seed": 2}}
    d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        cfg_slow = dict(cfg_base, train=dict(cfg_base["train"], fast=False))
        cfg_fast = dict(cfg_base, train=dict(cfg_base["train"], fast=True, signal_batch_size=7))
        rec_slow = build_signals(cfg_slow, d1)
        rec_fast = build_signals(cfg_fast, d2)
        # 不同推論 batch size 會改變矩陣乘法內部加總的順序，float32 下有 ~1e-7 量級的捨入差異（不是逐位元相同，
        # 但遠小於任何會影響分類門檻或 RMIA 分數排序的量級）
        assert np.allclose(rec_slow.arrays["signals"], rec_fast.arrays["signals"], atol=1e-5)
        assert np.allclose(rec_slow.arrays["pop_signals"], rec_fast.arrays["pop_signals"], atol=1e-5)
        assert np.array_equal(rec_slow.arrays["train_acc"], rec_fast.arrays["train_acc"])
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)
