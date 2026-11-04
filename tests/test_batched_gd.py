"""
檔案功能
--------
一致性測試：trainers/batched_gd.py 的「vmap 平行訓練」必須與標準的 torch.optim.SGD(momentum) 逐模型訓練得到相同參數。
這是本框架對「我們自己實作的訓練演算法 A」的基準檢查（目前沒有官方 PyTorch 版可對照，故以 PyTorch 標準優化器為基準）。
使用方式：pytest tests/test_batched_gd.py -q
"""
import numpy as np
import torch
import torch.nn.functional as F

from util.registry import build

CE = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})
from models.mlp import flatten_params
from trainers.batched_gd import accuracy_of_flat_params, train_models_batched

SPEC = {"module": "models.mlp", "type": "MLP", "kwargs": {"in_dim": 16, "hidden": [6], "out_dim": 4, "activation": "elu"}}


def _reference_one(fx, fy, zx, zy, seed, lr, mom, epochs):
    """基準：用標準 torch.optim.SGD 訓練單一模型（損失 = D- ∪ {z} 的平均交叉熵）。"""
    torch.manual_seed(seed)
    m = build(SPEC).double()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=mom)
    X, Y = torch.cat([fx, zx[None]]).double(), torch.cat([fy, zy[None]])
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(m(X), Y).backward()
        opt.step()
    return flatten_params(m).numpy()


def test_batched_equals_reference_sgd():
    """[驗證 train_models_batched] 3 個模型（不同額外資料）的結果與逐一 torch.optim.SGD 訓練相同（float32 誤差內）。"""
    rng = np.random.default_rng(0)
    fx = rng.random((40, 16)).astype(np.float32); fy = rng.integers(0, 4, 40)
    zx = rng.random((3, 16)).astype(np.float32); zy = rng.integers(0, 4, 3)
    got = train_models_batched(SPEC, 7, fx, fy, zx, zy, lr=0.1, momentum=0.9, epochs=30, loss_fn=CE, chunk_size=2)   # chunk 不整除，順便測邊界
    for i in range(3):
        ref = _reference_one(torch.as_tensor(fx), torch.as_tensor(fy), torch.as_tensor(zx[i]), torch.as_tensor(zy[i]), 7, 0.1, 0.9, 30)
        assert np.allclose(got[i], ref, atol=1e-4), np.abs(got[i] - ref).max()


def test_same_init_and_different_targets_differ():
    """[驗證 theta_0 共用] epochs=0 時所有模型等於同一個初始參數；訓練後不同目標得到不同參數。"""
    rng = np.random.default_rng(1)
    fx = rng.random((20, 16)).astype(np.float32); fy = rng.integers(0, 4, 20)
    zx = rng.random((2, 16)).astype(np.float32); zy = np.array([0, 1])
    p0 = train_models_batched(SPEC, 3, fx, fy, zx, zy, 0.1, 0.9, 0, CE)
    assert np.allclose(p0[0], p0[1])
    p1 = train_models_batched(SPEC, 3, fx, fy, zx, zy, 0.1, 0.9, 20, CE)
    assert not np.allclose(p1[0], p1[1])


def test_accuracy_helper():
    """[驗證 accuracy_of_flat_params] 與直接用模型算出的準確率一致。"""
    rng = np.random.default_rng(2)
    fx = rng.random((60, 16)).astype(np.float32); fy = rng.integers(0, 4, 60)
    params = train_models_batched(SPEC, 5, fx, fy, fx[:2], fy[:2], 0.1, 0.9, 10, CE)
    acc = accuracy_of_flat_params(SPEC, params, fx, fy)
    torch.manual_seed(5); m = build(SPEC)
    off, sizes = 0, [p.numel() for p in m.parameters()]
    with torch.no_grad():
        for p, n in zip(m.parameters(), sizes):
            p.copy_(torch.as_tensor(params[0][off:off + n]).reshape(p.shape)); off += n
        direct = (m(torch.as_tensor(fx)).argmax(-1).numpy() == fy).mean()
    assert abs(acc[0] - direct) < 1e-6


def test_fast_path_equals_vmap_path():
    """[驗證 MLP 快速路徑] fast=True（批次 GEMM）與 fast=False（通用 vmap）得到相同參數，含兩層隱藏層與 chunk 不整除。"""
    rng = np.random.default_rng(3)
    for hidden, act in [([6], "elu"), ([5, 4], "relu")]:
        spec = {"module": "models.mlp", "type": "MLP", "kwargs": {"in_dim": 16, "hidden": hidden, "out_dim": 4, "activation": act}}
        fx = rng.random((40, 16)).astype(np.float32); fy = rng.integers(0, 4, 40)
        zx = rng.random((5, 16)).astype(np.float32); zy = rng.integers(0, 4, 5)
        a = train_models_batched(spec, 7, fx, fy, zx, zy, 0.1, 0.9, 30, CE, chunk_size=2, fast=True)
        b = train_models_batched(spec, 7, fx, fy, zx, zy, 0.1, 0.9, 30, CE, chunk_size=2, fast=False)
        assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()


def test_per_model_init_seeds():
    """[驗證 init_seeds] epochs=0 時第 i 個模型 = 以 init_seeds[i] 單獨建立的模型；兩條路徑（fast / vmap）一致；訓練後仍可進行。"""
    rng = np.random.default_rng(4)
    fx = rng.random((20, 16)).astype(np.float32); fy = rng.integers(0, 4, 20)
    zx = rng.random((3, 16)).astype(np.float32); zy = np.array([0, 1, 2])
    seeds = np.array([11, 12, 13])
    p0 = train_models_batched(SPEC, 7, fx, fy, zx, zy, 0.1, 0.9, 0, CE, chunk_size=2, init_seeds=seeds)
    for i, sd in enumerate(seeds):
        torch.manual_seed(int(sd))
        assert np.allclose(p0[i], flatten_params(build(SPEC)).numpy())
    a = train_models_batched(SPEC, 7, fx, fy, zx, zy, 0.1, 0.9, 20, CE, chunk_size=2, init_seeds=seeds, fast=True)
    b = train_models_batched(SPEC, 7, fx, fy, zx, zy, 0.1, 0.9, 20, CE, chunk_size=2, init_seeds=seeds, fast=False)
    assert np.allclose(a, b, atol=1e-5)
