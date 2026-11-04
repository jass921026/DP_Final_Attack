"""
檔案功能
--------
框架結構的測試：
    1. 每個資料夾的母類（BaseDataset / BaseModel / BaseLoss / BaseTrainer / BaseAttack）與其子類別的繼承關係
    2. 資料集（torch Dataset）與 DataLoader：整批取樣 = 手動切片；每個 epoch 的打亂順序可由種子重現
    3. loss/ 下的損失與直接以 torch.nn.functional 計算的結果一致；WeightedSum = MAE + MSE
    4. Trainer 與「重構前的手寫訓練迴圈」得到逐元素相同的權重（含資料翻轉增強與 SGD 動量）
使用方式：pytest tests/test_framework_structure.py -q
"""
import importlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from util.data import EpochPermutation, make_loader
from util.registry import build
from datasets.base import BaseDataset
from trainers.optimizers import OptaxRMSprop
from trainers.trainer import Trainer, random_hflip


@pytest.mark.parametrize("module,cls,base_module,base", [
    ("datasets.mnist", "MNIST", "datasets.base", "BaseDataset"),
    ("datasets.cifar10", "CIFAR10", "datasets.base", "BaseDataset"),
    ("datasets.imagenet_val", "ImageNetVal", "datasets.base", "BaseDataset"),
    ("datasets.synthetic_images", "SyntheticImages", "datasets.base", "BaseDataset"),
    ("models.mlp", "MLP", "models.base", "BaseModel"),
    ("models.small_cnn", "SmallCNN", "models.base", "BaseModel"),
    ("models.resnet", "ResNet18", "models.base", "BaseModel"),
    ("attacks.imprint", "ImprintBlock", "models.base", "BaseModel"),
    ("loss.cross_entropy", "CrossEntropy", "loss.base", "BaseLoss"),
    ("loss.regression", "MAE", "loss.base", "BaseLoss"),
    ("loss.regression", "MSE", "loss.base", "BaseLoss"),
    ("loss.combine", "WeightedSum", "loss.base", "BaseLoss"),
    ("trainers.trainer", "Trainer", "trainers.base", "BaseTrainer"),
    ("trainers.batched_gd", "BatchedGDTrainer", "trainers.base", "BaseTrainer"),
    ("trainers.fl_client", "FedSGDClient", "trainers.base", "BaseTrainer"),
    ("trainers.fl_client", "FedAvgClient", "trainers.base", "BaseTrainer"),
    ("attacks.balle_recon", "BalleReconstructionAttack", "attacks.base", "BaseAttack"),
    ("attacks.imprint", "ImprintAttack", "attacks.base", "BaseAttack"),
    ("attacks.rmia", "RMIAAttack", "attacks.base", "BaseAttack"),
])
def test_every_class_inherits_its_folder_base(module, cls, base_module, base):
    """[驗證母類] 每個資料夾的每個類別都繼承該資料夾的母類。"""
    assert issubclass(getattr(importlib.import_module(module), cls), getattr(importlib.import_module(base_module), base))


def test_dataset_and_loader():
    """[驗證 BaseDataset / make_loader] 單筆、整批、arrays 一致；不打亂時 batch = 手動切片；邊界 batch 較小。"""
    x = np.random.default_rng(0).random((25, 3)).astype(np.float32); y = np.arange(25) % 4
    ds = BaseDataset(x, y)
    assert len(ds) == 25 and torch.equal(ds[3][0], torch.as_tensor(x[3]))
    xb, yb = ds.__getitems__([4, 1, 9])
    assert np.array_equal(xb.numpy(), x[[4, 1, 9]]) and np.array_equal(ds.arrays([4, 1, 9])[1], y[[4, 1, 9]])
    idx = np.arange(5, 22)
    batches = list(make_loader(ds, idx, 8, shuffle=False))
    assert [len(b[0]) for b in batches] == [8, 8, 1]
    assert np.array_equal(torch.cat([b[1] for b in batches]).numpy(), y[idx])


def test_epoch_permutation_reproduces_numpy_rng():
    """[驗證 EpochPermutation] 第 e 個 epoch 的順序 = default_rng(seed) 的第 e 次 permutation（與舊的手動迴圈相同）。"""
    ds = BaseDataset(np.zeros((12, 2), np.float32), np.arange(12))
    loader = make_loader(ds, None, 5, shuffle=True, seed=7)
    rng = np.random.default_rng(7)
    for _ in range(3):
        got = torch.cat([b[1] for b in loader]).numpy()
        assert np.array_equal(got, rng.permutation(12))


def test_losses_match_functional():
    """[驗證 loss/] CE 與 F.cross_entropy 相同；MAE/MSE（含 sigmoid）與手算相同；WeightedSum = MAE + MSE；per_sample 的平均 = mean。"""
    g = torch.Generator().manual_seed(0)
    logits, y = torch.randn(7, 5, generator=g), torch.randint(0, 5, (7,), generator=g)
    ce = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})
    assert torch.equal(ce(logits, y), F.cross_entropy(logits, y))
    assert torch.allclose(ce.per_sample(logits, y).mean(), ce(logits, y))
    ce_sum = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {"reduction": "sum"}})
    assert torch.allclose(ce_sum(logits, y), F.cross_entropy(logits, y, reduction="sum"))
    out, tgt = torch.randn(6, 9, generator=g), torch.rand(6, 9, generator=g)
    mae = build({"module": "loss.regression", "type": "MAE", "kwargs": {"activation": "sigmoid"}})
    mse = build({"module": "loss.regression", "type": "MSE", "kwargs": {"activation": "sigmoid"}})
    p = torch.sigmoid(out)
    assert torch.equal(mae(out, tgt), (p - tgt).abs().mean()) and torch.equal(mse(out, tgt), ((p - tgt) ** 2).mean())
    comb = build({"module": "loss.combine", "type": "WeightedSum", "kwargs": {"terms": [
        {"weight": 1.0, "loss": {"module": "loss.regression", "type": "MAE", "kwargs": {"activation": "sigmoid"}}},
        {"weight": 1.0, "loss": {"module": "loss.regression", "type": "MSE", "kwargs": {"activation": "sigmoid"}}}]}})
    assert torch.equal(comb(out, tgt), mae(out, tgt) + mse(out, tgt))
    assert torch.allclose(comb.per_sample(out, tgt).mean(), comb(out, tgt))


def _old_train_one(spec, x, y, epochs, lr, batch_size, seed, momentum=0.9, wd=0.0, flip=True):
    """重構前（已移除的）standard_sgd.train_one 的手寫迴圈（作為基準；僅 CPU）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(spec)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    model.train()
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(len(xt)))
        for s in range(0, len(xt), batch_size):
            idx = perm[s:s + batch_size]
            xb = xt[idx]
            if flip:
                m = torch.rand(len(idx)) < 0.5
                xb = torch.where(m[:, None, None, None], xb.flip(3), xb)
            loss = F.cross_entropy(model(xb), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def test_trainer_equals_old_loop():
    """[驗證 Trainer] 與重構前的手寫訓練迴圈得到相同權重（SmallCNN、翻轉增強、SGD 動量、batch 不整除）。"""
    spec = {"module": "models.small_cnn", "type": "SmallCNN", "kwargs": {"width": 4, "num_classes": 3}}
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3, 8, 8)).astype(np.float32); y = rng.integers(0, 3, 50)
    ref = _old_train_one(spec, x, y, epochs=3, lr=0.05, batch_size=16, seed=11)
    torch.manual_seed(11)
    model = build(spec)
    loader = make_loader(BaseDataset(x, y), np.arange(50), 16, shuffle=True, seed=11)
    Trainer(model, build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}), loader,
            {"name": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 0.0}, 3, "cpu", augment=random_hflip).fit()
    for a, b in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_optax_rmsprop_step():
    """[驗證 OptaxRMSprop] 一步更新 = theta - lr * g / sqrt((1-decay) g^2 + eps)（nu 初始為 0，eps 在根號內）。"""
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = OptaxRMSprop([p], lr=0.1)
    p.grad = torch.tensor([0.5, -0.25])
    expect = p.detach() - 0.1 * p.grad / torch.sqrt(0.1 * p.grad ** 2 + 1e-8)
    opt.step()
    assert torch.allclose(p.detach(), expect)
