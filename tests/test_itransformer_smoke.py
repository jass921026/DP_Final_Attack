"""
檔案功能
--------
PR #1 (iTransformer copy) 的 smoke test：
  1. 匯入與前向：分類輸出形狀 (B, num_class)、數值有限；預測任務輸出 (B, pred_len, N)；
  2. 反向：所有參數都拿到有限梯度（沒有斷掉的分支）；
  3. 可學習：固定小批次上訓練 150 步，損失明顯下降；
  4. 評估模式下確定性，且樣本彼此獨立（打亂批次順序，輸出跟著打亂）；
  5. 與 DP 相容：eval 模式下 torch.func 的逐樣本梯度（vmap）可算，且等於逐筆單獨反向的結果；
  6. 注意力是沿「變數」軸：token 數 = 變數數（enc_in），不是時間步數。
使用方式：pytest tests/test_itransformer_smoke.py -q
"""
import pytest
import torch
from torch.func import functional_call, grad, vmap

from models.iTransformer import Model


class Cfg:
    def __init__(self, **kw):
        self.task_name = "classification"
        self.seq_len, self.pred_len, self.enc_in, self.num_class = 13, 0, 7, 3
        self.d_model, self.d_ff, self.n_heads, self.dropout = 16, 32, 2, 0.1
        self.activation, self.e_layers = "gelu", 2
        self.__dict__.update(kw)


def _run(m, x):
    return m(x, None, None, None)


def test_classification_forward_shape_finite():
    torch.manual_seed(0)
    m = Model(Cfg()).eval()
    y = _run(m, torch.randn(5, 13, 7))
    assert y.shape == (5, 3) and torch.isfinite(y).all()


@pytest.mark.parametrize("task", ["long_term_forecast", "short_term_forecast"])
def test_forecast_forward_shape(task):
    m = Model(Cfg(task_name=task, pred_len=4)).eval()
    x = torch.randn(2, 13, 7)
    y = m(x, torch.randn(2, 13, 4), None, torch.randn(2, 4, 4))   # x_mark 會被當成額外變數 token（官方行為）
    assert y.shape == (2, 4, 7) and torch.isfinite(y).all()


def test_all_parameters_receive_gradient():
    torch.manual_seed(0)
    m = Model(Cfg(dropout=0.0)).train()
    loss = torch.nn.functional.cross_entropy(_run(m, torch.randn(8, 13, 7)), torch.randint(0, 3, (8,)))
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, bad
    assert all(p.grad.abs().sum() > 0 for p in m.parameters() if p.dim() > 1)


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    m = Model(Cfg(dropout=0.0)).train()
    x, y = torch.randn(16, 13, 7), torch.randint(0, 3, (16,))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(_run(m, x), y)
        loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first, (first, loss.item())


def test_eval_deterministic_and_batch_independent():
    torch.manual_seed(0)
    m = Model(Cfg(dropout=0.3)).eval()
    x = torch.randn(6, 13, 7)
    a = _run(m, x)
    assert torch.equal(a, _run(m, x))                       # dropout 在 eval 關閉
    perm = torch.randperm(6)
    assert torch.allclose(_run(m, x[perm]), a[perm], atol=1e-5)


def test_per_sample_grads_vmap_match_loop():
    torch.manual_seed(0)
    m = Model(Cfg(dropout=0.0)).eval()
    x, y = torch.randn(4, 13, 7), torch.randint(0, 3, (4,))
    params = {k: v.detach() for k, v in m.named_parameters()}

    def loss(p, xi, yi):
        out = functional_call(m, p, (xi[None], None, None, None))
        return torch.nn.functional.cross_entropy(out, yi[None])

    g = vmap(grad(loss), in_dims=(None, 0, 0))(params, x, y)
    for i in range(4):
        m.zero_grad()
        torch.nn.functional.cross_entropy(_run(m, x[i:i + 1]), y[i:i + 1]).backward()
        for k, p in m.named_parameters():
            assert torch.allclose(g[k][i], p.grad, atol=1e-5), (i, k)


def test_tokens_are_variates_not_time_steps():
    m = Model(Cfg(seq_len=13, enc_in=7)).eval()
    seen = {}
    m.encoder.register_forward_hook(lambda mod, inp, out: seen.update(shape=tuple(inp[0].shape)))
    _run(m, torch.randn(2, 13, 7))
    assert seen["shape"] == (2, 7, 16)                      # (B, 變數數, d_model)
