"""
檔案功能
--------
DP-GD（trainers/batched_gd.py 的 _mlp_dp_grads）的一致性測試：
  1. sigma=0（只裁剪）：與 torch.func 逐樣本梯度 + 手動裁剪的獨立參考相同；
  2. 與 Opacus 的 DPOptimizer（noise_multiplier=0）相同；
  3. 雜訊：差值的標準差 = sigma * clip / (n+1)；
  4. 整個訓練迴圈：dp=None 與 clip 極大且 sigma=0 時結果相同。
使用方式：pytest tests/test_dp_gd.py -q
"""
import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from util.registry import build
from trainers.batched_gd import _mlp_dp_grads, train_models_batched
from models.mlp import _ACTIVATIONS

CE = build({"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})
SPEC = {"module": "models.mlp", "type": "MLP", "kwargs": {"in_dim": 16, "hidden": [6, 5], "out_dim": 4, "activation": "elu", "init": "haiku"}}


def _setup(B=3, n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    fx, fy = torch.rand(n, 16, generator=g), torch.randint(0, 4, (n,), generator=g)
    zx, zy = torch.rand(B, 16, generator=g) * 3, torch.randint(0, 4, (B,), generator=g)
    torch.manual_seed(seed)
    base = build(SPEC)
    per = []
    for b in range(B):                                   # 每個模型不同的參數
        torch.manual_seed(100 + b)
        per.append({k: v.detach() for k, v in build(SPEC).named_parameters()})
    params = {k: torch.stack([d[k] for d in per]) for k in per[0]}
    return base, params, per, fx, fy, zx, zy


def _reference(base, p, fx, fy, zx, zy, clip):
    """獨立參考：torch.func 逐樣本梯度，裁剪後加總，除以 n+1。"""
    X, Y = torch.cat([fx, zx[None]]), torch.cat([fy, zy[None]])

    def one(pp, x, y):
        return torch.nn.functional.cross_entropy(functional_call(base, pp, (x[None],)), y[None])

    g = vmap(grad(one), in_dims=(None, 0, 0))(p, X, Y)
    norms = torch.sqrt(sum((v.reshape(len(X), -1) ** 2).sum(1) for v in g.values()))
    c = (clip / (norms + 1e-12)).clamp(max=1.0)
    return {k: (v * c.reshape(-1, *([1] * (v.dim() - 1)))).sum(0) / len(X) for k, v in g.items()}, norms


def test_clip_only_matches_vmap_reference():
    base, params, per, fx, fy, zx, zy = _setup()
    act = _ACTIVATIONS["elu"]()
    for clip in (0.05, 0.5, 100.0):                       # 小：全部被裁剪；大：幾乎不裁剪
        got = _mlp_dp_grads(params, act, CE, fx, fy, zx, zy, clip, 0.0, None)
        for b in range(len(zx)):
            ref, norms = _reference(base, per[b], fx, fy, zx[b], zy[b], clip)
            for k in ref:
                assert torch.allclose(got[k][b], ref[k], atol=1e-6), (clip, b, k, (got[k][b] - ref[k]).abs().max())
    assert norms.max() > 0.5                              # 確認小 clip 真的有裁剪到


def test_matches_opacus_clipping():
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer
    base, params, per, fx, fy, zx, zy = _setup(B=2)
    act = _ACTIVATIONS["elu"]()
    clip = 0.3
    got = _mlp_dp_grads(params, act, CE, fx, fy, zx, zy, clip, 0.0, None)
    for b in range(2):
        m = build(SPEC)
        m.load_state_dict({k: per[b][k] for k in per[b]})
        gm = GradSampleModule(m)
        opt = DPOptimizer(torch.optim.SGD(gm.parameters(), lr=1.0), noise_multiplier=0.0, max_grad_norm=clip, expected_batch_size=len(fx) + 1)
        X, Y = torch.cat([fx, zx[b][None]]), torch.cat([fy, zy[b][None]])
        loss = torch.nn.functional.cross_entropy(gm(X), Y, reduction="sum")     # 逐樣本梯度 = 各樣本自己的損失梯度
        loss.backward()
        opt.pre_step()                                     # 裁剪 + 加總 + 除以 expected_batch_size，結果在 p.grad
        for k, p in m.named_parameters():
            assert torch.allclose(got[k][b], p.grad, atol=1e-6), (b, k, (got[k][b] - p.grad).abs().max())


def test_noise_scale():
    _, params, _, fx, fy, zx, zy = _setup(B=16)
    act = _ACTIVATIONS["elu"]()
    clip, sigma, n = 0.5, 2.0, len(fx)
    clean = _mlp_dp_grads(params, act, CE, fx, fy, zx, zy, clip, 0.0, None)
    noisy = _mlp_dp_grads(params, act, CE, fx, fy, zx, zy, clip, sigma, torch.Generator().manual_seed(0))
    d = torch.cat([(noisy[k] - clean[k]).reshape(-1) for k in clean])
    assert abs(d.std().item() / (sigma * clip / (n + 1)) - 1) < 0.1, d.std().item()


def test_training_loop_clip_huge_equals_plain():
    rng = np.random.default_rng(0)
    fx = rng.random((40, 16)).astype(np.float32); fy = rng.integers(0, 4, 40)
    zx = rng.random((3, 16)).astype(np.float32); zy = rng.integers(0, 4, 3)
    a = train_models_batched(SPEC, 7, fx, fy, zx, zy, 0.1, 0.9, 20, CE, chunk_size=2)
    b = train_models_batched(SPEC, 7, fx, fy, zx, zy, 0.1, 0.9, 20, CE, chunk_size=2, dp={"clip": 1e9, "sigma": 0.0, "seed": 0})
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()
