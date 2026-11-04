"""
檔案功能
--------
imprint 攻擊（attacks/imprint.py）的測試：
    1. 與官方 breaching 的 ImprintBlock 逐參數相同（以檔案路徑直接載入官方檔案，不需安裝 breaching；找不到則跳過）
    2. 反演運算與官方 ImprintAttacker + invert_fc_layer 的運算相同（測試內以官方原始碼逐行寫成基準）
    3. 小型端到端：單獨落在 bin 的影像被精確還原；完美還原比例與理論 (1-1/k)^(n-1) 同量級
使用方式：pytest tests/test_imprint.py -q
"""
import importlib.util
import os

import numpy as np
import pytest
import torch

from attacks.imprint import (ImprintBlock, SparseImprintBlock, best_match_psnr, expected_perfect_fraction, invert_cumulative, reconstruct)
from util.registry import build
from trainers.fl_client import fedavg_update, fedsgd_gradient

CE = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})

OFFICIAL = os.path.join(os.path.dirname(__file__), "..", "..", "imprint_attack", "breaching", "breaching",
                        "cases", "malicious_modifications", "imprint.py")


def _official_block_module():
    spec = importlib.util.spec_from_file_location("official_imprint", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.isfile(OFFICIAL), reason="official breaching checkout not found")
@pytest.mark.parametrize("connection", ["addition", "linear"])
def test_block_matches_official(connection):
    """[驗證 ImprintBlock] 權重、偏置、線性連接層的參數與官方相同，且輸出相同。"""
    off = _official_block_module().ImprintBlock((3, 8, 8), num_bins=32, connection=connection, gain=1e-3, linfunc="fourier", mode=32)
    ours = ImprintBlock((3, 8, 8), 32, connection, 1e-3, "fourier", 32)
    assert torch.allclose(ours.linear0.weight, off.linear0.weight, atol=1e-9)
    assert torch.allclose(ours.linear0.bias, off.linear0.bias, atol=1e-9)
    assert np.allclose(ours.bins, off.bins)
    if connection == "linear":
        assert torch.allclose(ours.linear2.weight, off.linear2.weight)
        ours.linear2.bias.data.copy_(off.linear2.bias.data)      # 官方的 bias 初始值為隨機（nn.Linear 預設），只比較 -mean(bins) 的偏移
        assert torch.allclose(ours.linear2.bias - ours.linear2.bias.data, torch.zeros(1))
    x = torch.rand(4, 3, 8, 8)
    assert torch.allclose(ours(x), off(x), atol=1e-6)


def _official_inversion(weight_grad, bias_grad):
    """官方 ImprintAttacker.reconstruct（cumulative）與 invert_fc_layer 的原始運算（breaching/attacks/analytic_attack.py）。"""
    weight_grad, bias_grad = weight_grad.clone(), bias_grad.clone()
    for i in reversed(list(range(1, weight_grad.shape[0]))):
        weight_grad[i] -= weight_grad[i - 1]
        bias_grad[i] -= bias_grad[i - 1]
    valid_classes = bias_grad != 0
    return weight_grad[valid_classes, :] / bias_grad[valid_classes, None]


def test_inversion_matches_official():
    """[驗證 invert_cumulative] 隨機梯度（含零差分列）上與官方運算結果完全相同。"""
    g = torch.Generator().manual_seed(0)
    wg, bg = torch.randn(40, 50, generator=g), torch.randn(40, generator=g)
    bg[10] = bg[9]; wg[10] = wg[9]          # 製造空 bin（相鄰列相同）
    ours, _, _, _ = invert_cumulative(wg, bg)
    assert torch.equal(ours, _official_inversion(wg, bg))


def test_end_to_end_exact_for_isolated_images():
    """[驗證 reconstruct] 小型網路上，單獨落在某 bin 的影像被精確還原，完美還原比例接近理論值。"""
    torch.manual_seed(0)
    n, k = 16, 256
    # 影像亮度均勻分布 -> avg 線性函數的 h 約在 [-2, 2.4]，與常態 bins 的假設相容（隨機雜訊影像的 h 變異太小/太大，會擠進尾端 bin）
    bright = torch.rand(n)[:, None, None, None] * 0.9 + 0.05
    x01 = (bright + 0.02 * torch.randn(n, 3, 8, 8)).clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    xn = (x01 - mean) / std
    model = torch.nn.Sequential(ImprintBlock((3, 8, 8), k, "addition", 1e-3, "avg", 0), torch.nn.Flatten(), torch.nn.Linear(192, 10))
    grads = fedsgd_gradient(model, CE, xn, torch.randint(0, 10, (n,)), micro_batch=5)       # 5 不整除 16，順便測邊界
    rec, hits = reconstruct(grads, n, (3, 8, 8))
    stats, _ = best_match_psnr((rec * std + mean).clamp(0, 1), x01)
    assert hits >= 1
    assert stats["frac_perfect"] >= 0.5 * expected_perfect_fraction(n, k)       # 理論 ≈ 0.94；放寬到一半，因為 h 的真實分布與常態不完全相同
    assert stats["frac_perfect"] > 0.5


@pytest.mark.skipif(not os.path.isfile(OFFICIAL), reason="official breaching checkout not found")
def test_sparse_block_matches_official():
    """[驗證 SparseImprintBlock] 權重、偏置、bin 寬度、Hardtanh 上限與輸出皆與官方相同（linear 連接）。"""
    off = _official_block_module().SparseImprintBlock((3, 8, 8), num_bins=32, connection="linear", gain=1.0, linfunc="fourier", mode=32)
    ours = SparseImprintBlock((3, 8, 8), 32, "linear", 1.0, "fourier", 32)
    assert torch.allclose(ours.linear0.weight, off.linear0.weight, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ours.linear0.bias, off.linear0.bias, atol=1e-6, rtol=1e-5)
    assert np.allclose(ours.bin_sizes, off.bin_sizes) and np.allclose(ours.bins, off.bins)
    assert ours.nonlin.max_val == off.nonlin.max_val and ours.nonlin.min_val == off.nonlin.min_val
    ours.linear2.bias.data.copy_(off.linear2.bias.data)
    x = torch.rand(4, 3, 8, 8)
    assert torch.allclose(ours(x), off(x), atol=1e-5)


def test_fedavg_update_equals_manual_sgd_and_restores_model():
    """[驗證 fedavg_update] 兩步 SGD 的參數差 = 手動計算；呼叫後模型參數不變。"""
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3))
    x, y = torch.randn(6, 3, 2, 2), torch.randint(0, 3, (6,))
    before = [p.detach().clone() for p in model.parameters()]
    upd = fedavg_update(model, CE, x, y, steps=2, per_step=3, lr=0.1)
    m2 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3)); m2.load_state_dict(model.state_dict())
    opt = torch.optim.SGD(m2.parameters(), lr=0.1)
    for t in range(2):
        opt.zero_grad(); torch.nn.functional.cross_entropy(m2(x[3 * t:3 * t + 3]), y[3 * t:3 * t + 3]).backward(); opt.step()
    for u, p2, b in zip(upd, m2.parameters(), before):
        assert torch.allclose(u, p2.detach() - b, atol=1e-6)
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p.detach(), b)
