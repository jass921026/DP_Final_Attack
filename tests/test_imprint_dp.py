"""
檔案功能
--------
DP-FedSGD 客戶端（trainers/fl_client.py 的 dp_clipped_gradient）與 GDP 換算（util/gdp.py）的測試：
  1. 與 torch.func 逐樣本梯度 + 手動裁剪的獨立參考相同；
  2. clip 極大時等於一般 FedSGD 梯度；
  3. 回傳的逐筆範數與參考一致；
  4. GDP：sigma_for_epsilon 與 gdp_epsilon 互逆，且 mu 越大 eps 越大。
使用方式：pytest tests/test_imprint_dp.py -q
"""
import torch
from torch.func import functional_call, grad, vmap

from trainers.fl_client import dp_clipped_gradient, fedsgd_gradient
from util.gdp import gdp_epsilon, sigma_for_epsilon
from util.registry import build

CE = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 12), torch.nn.ReLU(), torch.nn.Linear(12, 5)).eval()


def _data(n=10):
    g = torch.Generator().manual_seed(1)
    return torch.randn(n, 3, 4, 4, generator=g) * 2, torch.randint(0, 5, (n,), generator=g)


def _reference(net, x, y, clip):
    p = {k: v.detach() for k, v in net.named_parameters()}
    one = lambda pp, xi, yi: torch.nn.functional.cross_entropy(functional_call(net, pp, (xi[None],)), yi[None])
    g = vmap(grad(one), in_dims=(None, 0, 0))(p, x, y)
    norms = torch.sqrt(sum((v.reshape(len(x), -1) ** 2).sum(1) for v in g.values()))
    c = (clip / (norms + 1e-12)).clamp(max=1.0)
    return [(v * c.reshape(-1, *([1] * (v.dim() - 1)))).mean(0) for v in g.values()], norms


def test_matches_vmap_reference():
    net, (x, y) = _net(), _data()
    for clip in (0.05, 1.0, 100.0):
        got, norms = dp_clipped_gradient(net, CE, x, y, clip)
        ref, ref_norms = _reference(net, x, y, clip)
        assert torch.allclose(norms, ref_norms, atol=1e-5)
        for a, b in zip(got, ref):
            assert torch.allclose(a, b, atol=1e-6), (clip, (a - b).abs().max())
    assert ref_norms.max() > 0.05                         # 小 clip 真的有裁剪到


def test_huge_clip_equals_fedsgd():
    net, (x, y) = _net(), _data()
    got, _ = dp_clipped_gradient(net, CE, x, y, 1e9)
    for a, b in zip(got, fedsgd_gradient(net, CE, x, y, 4)):
        assert torch.allclose(a, b, atol=1e-6)


def test_gdp_roundtrip_and_monotone():
    for eps in (0.5, 1.0, 10.0, 1000.0):
        s = sigma_for_epsilon(eps, 1e-5, steps=1)
        assert abs(gdp_epsilon(1.0 / s, 1e-5) - eps) / eps < 1e-4
        s100 = sigma_for_epsilon(eps, 1e-5, steps=100)          # 100 步需要 10 倍的 sigma
        assert abs(s100 / s - 10) < 1e-6
    assert gdp_epsilon(2.0) > gdp_epsilon(1.0) > gdp_epsilon(0.5)
