"""
檔案功能
--------
compute_empirical_bins（attacks/imprint.py）與 batch=1 imprint 還原的測試（不依賴 P12 資料集，用合成資料）：
    1. 經驗分位點大致等於 h(aux_x) 的真實分位數，且遞增
    2. 用這些分位點建的 ImprintBlock，在 batch=1、無雜訊時精確還原輸入
    3. 經過 dp_clipped_gradient 裁剪後，batch=1 的還原比例（dW/db）不變（裁剪只是整個梯度乘上同一個標量）
使用方式：pytest tests/test_imprint_empirical_bins.py -q
"""
import torch

from attacks.imprint import ImprintBlock, compute_empirical_bins, reconstruct
from trainers.fl_client import dp_clipped_gradient, fedsgd_gradient
from util.registry import build

CE = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})
SHAPE = (20,)


def test_empirical_bins_match_true_quantiles():
    torch.manual_seed(0)
    aux = torch.rand(2000, *SHAPE) - 0.5                      # 非常態、非 Laplace 的分布（均勻）也要能估對
    bins = compute_empirical_bins(aux, SHAPE, num_bins=8, linfunc="avg")
    tmp = ImprintBlock(SHAPE, 8, linfunc="avg", gain=1.0)
    h = aux @ tmp.linear0.weight.data[0]
    true_q = [float(torch.quantile(h, i / 8)) for i in range(1, 8)]
    assert bins[0] < min(true_q) - 1.0                        # c_0 近似 -inf
    assert all(a < b for a, b in zip(bins[1:], bins[2:]))     # 遞增
    assert max(abs(a - b) for a, b in zip(bins[1:], true_q)) < 1e-4


def test_batch1_exact_reconstruction_with_empirical_bins():
    torch.manual_seed(1)
    aux = torch.randn(500, *SHAPE)
    bins = compute_empirical_bins(aux, SHAPE, num_bins=12, linfunc="avg")
    model = torch.nn.Sequential(ImprintBlock(SHAPE, 12, "linear", 1e-4, "avg", empirical_bins=bins), torch.nn.Linear(20, 3))
    model.eval()
    x = torch.randn(1, *SHAPE)
    y = torch.randint(0, 3, (1,))
    grads = fedsgd_gradient(model, CE, x, y, micro_batch=1)
    rec, n_hits = reconstruct(grads, 1, SHAPE, reduction=None, structure="cumulative", clip_range=(-50.0, 50.0))
    assert n_hits >= 1
    best = rec[((rec - x) ** 2).mean(1).argmin()]
    assert torch.allclose(best, x[0], atol=1e-4)


def test_clipping_preserves_reconstruction_ratio_for_batch1():
    """裁剪（單一標量縮放整個梯度）不改變 dW/db，所以還原結果在 clip 前後相同（只要沒被裁到 0）。"""
    torch.manual_seed(2)
    aux = torch.randn(500, *SHAPE)
    bins = compute_empirical_bins(aux, SHAPE, num_bins=12, linfunc="avg")
    model = torch.nn.Sequential(ImprintBlock(SHAPE, 12, "linear", 1e-4, "avg", empirical_bins=bins), torch.nn.Linear(20, 3))
    model.eval()
    x = torch.randn(1, *SHAPE)
    y = torch.randint(0, 3, (1,))
    clean = fedsgd_gradient(model, CE, x, y, micro_batch=1)
    clipped, norm = dp_clipped_gradient(model, CE, x, y, clip=float(norm_val := torch.sqrt(sum((g.double() ** 2).sum() for g in clean)).item()) / 2)
    rec_a, hits_a = reconstruct(clean, 1, SHAPE, reduction=None, structure="cumulative", clip_range=(-50.0, 50.0))
    rec_b, hits_b = reconstruct(clipped, 1, SHAPE, reduction=None, structure="cumulative", clip_range=(-50.0, 50.0))
    assert hits_a == hits_b
    assert torch.allclose(rec_a, rec_b, atol=1e-4)
