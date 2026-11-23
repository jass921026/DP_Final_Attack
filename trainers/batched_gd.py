"""
檔案功能
--------
同時訓練「很多個」小模型，每個模型的訓練集 = 共同的固定集 D- 加上各自不同的一筆額外資料（目標）。
這正是 Balle et al. 2022 的訓練演算法 A_{D-}(z)：
    - 對手用它訓練影子模型  theta_bar_i = A_{D-}(z_bar_i)          （Sec. IV-B 步驟 1）
    - 模型開發者用它訓練被攻擊的釋出模型  theta = A_{D-}(z)          （Algorithm 1 的 theta <- A(D- ∪ {z})）
演算法 A = 全批次（不抽 mini-batch）的梯度下降加動量，交叉熵損失，所有模型從同一組初始參數出發；損失由外部的 loss_fn（loss/ 下的 BaseLoss，使用其 per_sample）給定
（論文預設：對手知道初始化，Sec. V-A(b)；Table VI：lr 0.2、momentum 0.9、100 epochs）。
用 torch.func.vmap 一次平行訓練一批模型；與「逐個用 torch.optim.SGD 訓練」的結果相同（tests/test_batched_gd.py 驗證）。

使用方式
--------
    from trainers.batched_gd import train_models_batched          # 或 BatchedGDTrainer(...).fit()
    params = train_models_batched(
        model_spec={"module": "models.mlp", "type": "MLP",
                    "kwargs": {"in_dim": 784, "hidden": [10], "out_dim": 10, "activation": "elu"}},
        init_seed=42, fixed_x=Dx, fixed_y=Dy, extra_x=Zx, extra_y=Zy,
        lr=0.1, momentum=0.9, epochs=100, loss_fn=build({"module": "loss.cross_entropy", "type": "CrossEntropy"}), chunk_size=50)
    params.shape   # (len(Zx), n_params)  每列 = 一個模型展平後的參數（順序同 model.parameters()）
快速路徑：模型為 models.mlp.MLP 時，改用手寫的批次矩陣乘法（固定集只做一次 GEMM，不為每個模型複製），
記憶體與時間都遠低於 vmap 版本；數學上與 vmap 版本相同（tests/test_batched_gd.py 比對兩者）。vmap 版本是通用的參考實作。
DP-GD 版本（dp={"clip": C, "sigma": 噪音倍數, "seed": 整數}）：每步 g <- (sum_i clip_C(g_i) + N(0, (sigma C)^2 I)) / (n+1)（全批次、無子抽樣、add/remove 鄰接），
MLP 快速路徑（逐樣本梯度範數用「幽靈範數」：||dW_i||^2 = ||delta_i||^2 ||a_i||^2，不實體化逐樣本梯度；tests/test_dp_gd.py 以 vmap 逐樣本梯度與 Opacus 驗證）；其他模型（如 iTransformer）走通用 vmap 路徑，實體化逐樣本梯度（fast=False 的 MLP 與快速路徑一致，同測試檔驗證）。
"""
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from util.registry import build
from trainers.base import BaseTrainer


def _make_loss(base: torch.nn.Module, loss_fn: torch.nn.Module, fixed_x: torch.Tensor, fixed_y: torch.Tensor):
    """
    功能
        回傳單一模型的損失函數 L(params, z_x, z_y)：
        在 D- ∪ {z} 上的平均損失 = ( sum_{(x,y) in D-} l + l(z) ) / (n_fixed + 1)，l 由 loss_fn.per_sample 給定（Balle 為交叉熵）。
    Pseudo code (論文 Algorithm 1 的 theta <- A(D- ∪ {z}) 中，A 所最小化的目標)
        L(theta; D- ∪ {z}) = 1/(n+1) * ( sum_{j=1..n} CE(f_theta(x_j), y_j) + CE(f_theta(z_x), z_y) )
    用法
        loss = _make_loss(base, loss_fn, fx, fy); g = grad(loss)(params, zx, zy)
    """
    n = fixed_x.shape[0]

    def loss(params: Dict[str, torch.Tensor], z_x: torch.Tensor, z_y: torch.Tensor) -> torch.Tensor:
        # [對應 Pseudo code 第一項] 固定集 D- 上的交叉熵總和
        logits_fixed = functional_call(base, params, (fixed_x,))
        ce_fixed = loss_fn.per_sample(logits_fixed, fixed_y).sum()
        # [對應 Pseudo code 第二項] 額外一筆資料 z 的交叉熵
        logits_z = functional_call(base, params, (z_x[None],))
        ce_z = loss_fn.per_sample(logits_z, z_y[None]).sum()
        # [對應 Pseudo code 的 1/(n+1)] 平均
        return (ce_fixed + ce_z) / (n + 1)

    return loss


def _mlp_fast_loss(params: Dict[str, torch.Tensor], act: torch.nn.Module, loss_fn: torch.nn.Module, fixed_x: torch.Tensor,
                   fixed_y: torch.Tensor, z_x: torch.Tensor, z_y: torch.Tensor) -> torch.Tensor:
    """
    功能
        MLP 專用的批次損失：B 個模型各自在 D- ∪ {z_b} 上的平均損失（loss_fn.per_sample）之「總和」。因模型互相獨立，
        對總和求導得到的第 b 個切片 = 第 b 個模型自己的梯度（與 _make_loss 的 vmap(grad) 相同）。
    Pseudo code (與 _make_loss 相同的目標，只是把 B 個模型一起算)
        for b in 1..B:  L_b = 1/(n+1) * ( sum_j CE(f_{theta_b}(x_j), y_j) + CE(f_{theta_b}(z_b_x), z_b_y) )
        return sum_b L_b
    用法
        由 train_models_batched 在快速路徑內呼叫；params["net.{i}.weight"] 形狀 (B, out, in)，bias 形狀 (B, out)。
    """
    n = fixed_x.shape[0]
    idx = sorted({int(k.split(".")[1]) for k in params})
    B = z_x.shape[0]

    def forward(shared: bool) -> torch.Tensor:
        # [對應 Pseudo code: f_{theta_b}] 逐層：第一層對固定集用單一 GEMM（不複製 x），之後用 bmm
        h = None
        for pos, i in enumerate(idx):
            W, b = params[f"net.{i}.weight"], params[f"net.{i}.bias"]
            if pos == 0 and shared:
                h = (fixed_x @ W.reshape(-1, W.shape[-1]).T).reshape(n, B, -1).permute(1, 0, 2) + b[:, None, :]
            elif pos == 0:
                h = torch.einsum("bi,boi->bo", z_x, W)[:, None, :] + b[:, None, :]
            else:
                h = torch.baddbmm(b[:, None, :], h, W.transpose(1, 2))
            if pos < len(idx) - 1:
                h = act(h)
        return h

    logits_fixed = forward(True)                                   # (B, n, C)
    logits_z = forward(False)                                      # (B, 1, C)
    # [對應 Pseudo code: sum_j CE(...) 與 CE(z)] 兩項交叉熵（總和），再除以 (n+1)
    ce_fixed = loss_fn.per_sample(logits_fixed.reshape(B * n, -1), fixed_y.repeat(B)).sum()
    ce_z = loss_fn.per_sample(logits_z.reshape(B, -1), z_y).sum()
    return (ce_fixed + ce_z) / (n + 1)


def _mlp_dp_grads(params: Dict[str, torch.Tensor], act: torch.nn.Module, loss_fn: torch.nn.Module, fixed_x: torch.Tensor,
                  fixed_y: torch.Tensor, z_x: torch.Tensor, z_y: torch.Tensor, clip: float, sigma: float,
                  gen: Optional[torch.Generator]) -> Dict[str, torch.Tensor]:
    """
    功能
        MLP 專用的 DP-GD 梯度：B 個模型各自在 D- ∪ {z_b} 上的「逐樣本裁剪後加總，加高斯雜訊，除以 (n+1)」。
    Pseudo code (DP-GD，全批次；Abadi 2016 去掉子抽樣；與 Balle 2022 Sec. VI-D 的 DP-GD 相同)
        for each model b, each sample i in D- ∪ {z_b}:   g_i <- grad_theta CE(f_theta(x_i), y_i)
            c_i <- min(1, clip / ||g_i||_2)                 # ||g_i||^2 = sum_l ||delta_{l,i}||^2 (1 + ||a_{l-1,i}||^2)
        G_b <- ( sum_i c_i g_i + N(0, (sigma * clip)^2 I) ) / (n + 1)
    用法
        由 train_models_batched 在 dp 不為 None 時呼叫；回傳 dict，形狀同 params。sigma=0 即只裁剪。
    """
    n, B = fixed_x.shape[0], z_x.shape[0]
    idx = sorted({int(k.split(".")[1]) for k in params})
    L = len(idx)
    W = [params[f"net.{i}.weight"].detach() for i in idx]
    bs = [params[f"net.{i}.bias"].detach() for i in idx]

    def forward(shared: bool):
        hs, acts, a = [], [], None
        for pos in range(L):
            if pos == 0 and shared:
                h = (fixed_x @ W[0].reshape(-1, W[0].shape[-1]).T).reshape(n, B, -1).permute(1, 0, 2) + bs[0][:, None, :]
                acts.append(fixed_x)
            elif pos == 0:
                h = torch.einsum("bi,boi->bo", z_x, W[0])[:, None, :] + bs[0][:, None, :]
                acts.append(z_x[:, None, :])
            else:
                h = torch.baddbmm(bs[pos][:, None, :], a, W[pos].transpose(1, 2))
                acts.append(a)
            if pos == 0:
                h = h.detach().requires_grad_(True)          # 葉節點：之後對它求 delta
            hs.append(h)
            a = act(h) if pos < L - 1 else None
        return hs, acts

    with torch.enable_grad():
        hs_f, acts_f = forward(True)
        hs_z, acts_z = forward(False)
        total = loss_fn.per_sample(hs_f[-1].reshape(B * n, -1), fixed_y.repeat(B)).sum() + loss_fn.per_sample(hs_z[-1].reshape(B, -1), z_y).sum()
        deltas = torch.autograd.grad(total, hs_f + hs_z)
    d_f, d_z = deltas[:L], deltas[L:]
    # [對應 Pseudo code: ||g_i||^2] 逐樣本梯度範數（幽靈範數）
    a_f = [acts_f[0]] + [t.detach() for t in acts_f[1:]]
    a_z = [acts_z[0]] + [t.detach() for t in acts_z[1:]]
    sq_f = torch.zeros(B, n, device=fixed_x.device); sq_z = torch.zeros(B, 1, device=fixed_x.device)
    for l in range(L):
        an_f = (a_f[l] ** 2).sum(-1) if l > 0 else (a_f[0] ** 2).sum(-1)[None, :]
        sq_f = sq_f + (d_f[l] ** 2).sum(-1) * (1 + an_f)
        sq_z = sq_z + (d_z[l] ** 2).sum(-1) * (1 + (a_z[l] ** 2).sum(-1))
    c_f = (clip / (sq_f.sqrt() + 1e-12)).clamp(max=1.0)
    c_z = (clip / (sq_z.sqrt() + 1e-12)).clamp(max=1.0)
    # [對應 Pseudo code: sum_i c_i g_i] 以加權的 delta 做一次 GEMM，不實體化逐樣本梯度
    out = {}
    for l, i in enumerate(idx):
        cd_f, cd_z = c_f[..., None] * d_f[l], c_z[..., None] * d_z[l]
        if l == 0:
            gW = torch.einsum("bno,ni->boi", cd_f, a_f[0]) + torch.einsum("bno,bni->boi", cd_z, a_z[0])
        else:
            gW = torch.einsum("bno,bni->boi", cd_f, a_f[l]) + torch.einsum("bno,bni->boi", cd_z, a_z[l])
        gb = cd_f.sum(1) + cd_z.sum(1)
        out[f"net.{i}.weight"], out[f"net.{i}.bias"] = gW, gb
    # [對應 Pseudo code: N(0, (sigma clip)^2) 與除以 (n+1)]
    for k in out:
        if sigma > 0:
            noise = torch.randn(out[k].shape, generator=gen, dtype=out[k].dtype).to(out[k].device)
            out[k] = out[k] + sigma * clip * noise
        out[k] = out[k] / (n + 1)
    return out


def train_models_batched(
    model_spec: Dict[str, Any],
    init_seed: int,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
    extra_x: np.ndarray,
    extra_y: np.ndarray,
    lr: float,
    momentum: float,
    epochs: int,
    loss_fn: torch.nn.Module,
    chunk_size: int = 50,
    device: str = "cpu",
    dp: Optional[Dict[str, Any]] = None,
    fast: Optional[bool] = None,
    init_seeds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    功能
        對 extra_x 的每一筆 z_i，訓練一個模型 theta_i = A_{D-}(z_i)，回傳所有模型展平後的最終參數。
    Pseudo code (Balle 2022 Sec. V-A(b)：gradient descent with momentum, full batches, 同一初始化)
        theta_0 <- init(init_seed)                       # 所有模型共用（對手知道初始化）
        for each target z_i (平行處理):
            theta <- theta_0;  v <- 0
            repeat `epochs` times:
                g <- grad_theta L(theta; D- ∪ {z_i})
                v <- g + momentum * v                    # 動量（與 optax.sgd / torch.optim.SGD 相同）
                theta <- theta - lr * v
        return flatten(theta) for all i
    用法
        見檔頭。chunk_size 控制一次平行訓練幾個模型（記憶體 vs 速度）。fast=None 時自動：MLP 走快速路徑，其他走 vmap。
        init_seeds=None：所有模型共用 init_seed 的初始參數（對手知道初始化，論文預設）；
        給定長度 = 模型數的整數陣列：第 i 個模型用 init_seeds[i] 各自初始化（論文 Table II「Adversary does not know initial parameters」）。
    """
    if dp is not None and not (fast if fast is not None else (model_spec.get("module") == "models.mlp" and model_spec.get("type") == "MLP")):
        pass                                                   # 其他模型：通用 vmap 路徑（逐樣本梯度實體化，較慢）

    # [對應 Pseudo code: theta_0 <- init(init_seed)] 以固定種子建立基底模型，所有模型共用其初始參數
    torch.manual_seed(init_seed)
    base = build(model_spec).to(device)
    theta0 = {k: v.detach().clone() for k, v in base.named_parameters()}

    fx = torch.as_tensor(fixed_x, dtype=torch.float32, device=device)
    fy = torch.as_tensor(fixed_y, dtype=torch.long, device=device)
    zx_all = torch.as_tensor(extra_x, dtype=torch.float32, device=device)
    zy_all = torch.as_tensor(extra_y, dtype=torch.long, device=device)

    # [對應 Pseudo code: grad_theta L] 單一模型的梯度函數，對「模型維度」做 vmap 以平行處理 chunk 內所有模型
    loss_fn = loss_fn.to(device)
    batched_grad = vmap(grad(_make_loss(base, loss_fn, fx, fy)), in_dims=(0, 0, 0))

    if dp is not None:
        def _one(pp, x, y):
            return loss_fn.per_sample(functional_call(base, pp, (x[None],)), y[None]).sum()
        clip_c = float(dp["clip"])

        def dp_one(pp, z_x, z_y):
            X, Y = torch.cat([fx, z_x[None]]), torch.cat([fy, z_y[None]])
            g = vmap(grad(_one), in_dims=(None, 0, 0))(pp, X, Y)             # 逐樣本梯度 (n+1, ...)
            norms = torch.sqrt(sum((v.reshape(len(X), -1) ** 2).sum(1) for v in g.values()))
            c = (clip_c / (norms + 1e-12)).clamp(max=1.0)
            return {k: (v * c.reshape(-1, *([1] * (v.dim() - 1)))).sum(0) / len(X) for k, v in g.items()}
        batched_dp_grad = vmap(dp_one, in_dims=(0, 0, 0))

    use_fast = fast if fast is not None else (model_spec.get("module") == "models.mlp" and model_spec.get("type") == "MLP")
    if use_fast:
        from models.mlp import _ACTIVATIONS
        act = _ACTIVATIONS[model_spec["kwargs"].get("activation", "elu")]()
    out = []
    for s in range(0, len(zx_all), chunk_size):
        zx, zy = zx_all[s:s + chunk_size], zy_all[s:s + chunk_size]
        b = zx.shape[0]
        # [對應 Pseudo code: theta <- theta_0; v <- 0] 為 chunk 內每個模型複製初始參數與動量緩衝
        params = {k: v[None].repeat(b, *([1] * v.dim())) for k, v in theta0.items()}
        if init_seeds is not None:
            # [對應 Table II 的未知初始化] 每個模型各自用自己的種子建立初始參數，覆蓋共用的 theta_0
            per = []
            for sd in init_seeds[s:s + b]:
                torch.manual_seed(int(sd))
                per.append({k: v.detach() for k, v in build(model_spec).named_parameters()})
            params = {k: torch.stack([d[k] for d in per]).to(device) for k in theta0}
        vel = {k: torch.zeros_like(v) for k, v in params.items()}
        if use_fast:
            # [對應 Pseudo code: repeat epochs 次] 全批次更新（快速路徑：autograd 對「總和損失」求導）
            gen = torch.Generator().manual_seed(int(dp.get("seed", 0)) * 1_000_003 + s) if dp is not None else None
            for _ in range(epochs):
                if dp is not None:
                    gd = _mlp_dp_grads(params, act, loss_fn, fx, fy, zx, zy, float(dp["clip"]), float(dp.get("sigma", 0.0)), gen)
                    gs, keys = [gd[k] for k in params], list(params)
                else:
                    leaves = {k: v.detach().requires_grad_(True) for k, v in params.items()}
                    with torch.enable_grad():
                        total = _mlp_fast_loss(leaves, act, loss_fn, fx, fy, zx, zy)
                        gs = torch.autograd.grad(total, list(leaves.values()))
                    keys = list(leaves)
                with torch.no_grad():
                    for k, g_k in zip(keys, gs):
                        vel[k] = g_k + momentum * vel[k]            # v <- g + momentum * v
                        params[k] = params[k] - lr * vel[k]         # theta <- theta - lr * v
            out.append(torch.cat([params[k].reshape(b, -1) for k in theta0], dim=1).cpu())
            continue
        # [對應 Pseudo code: repeat epochs 次] 全批次更新（通用 vmap 路徑）
        gen = torch.Generator().manual_seed(int(dp.get("seed", 0)) * 1_000_003 + s) if dp is not None else None
        for _ in range(epochs):
            g = batched_dp_grad(params, zx, zy) if dp is not None else batched_grad(params, zx, zy)
            if dp is not None and float(dp.get("sigma", 0.0)) > 0:       # 加高斯雜訊 N(0, (sigma C)^2) / (n+1)
                nz = float(dp["sigma"]) * float(dp["clip"]) / (len(fx) + 1)
                g = {k: v + nz * torch.randn(v.shape, generator=gen).to(v.device) for k, v in g.items()}
            for k in params:
                vel[k] = g[k] + momentum * vel[k]           # v <- g + momentum * v
                params[k] = params[k] - lr * vel[k]         # theta <- theta - lr * v
        # [對應 Pseudo code: flatten(theta)] 依 named_parameters 順序串接
        out.append(torch.cat([params[k].reshape(b, -1) for k in theta0], dim=1).cpu())
    return torch.cat(out).numpy()


def accuracy_of_flat_params(model_spec: Dict[str, Any], flat_params: np.ndarray, x: np.ndarray, y: np.ndarray,
                            device: str = "cpu", chunk_size: int = 100) -> np.ndarray:
    """
    功能
        計算每個（展平參數表示的）模型在資料 (x, y) 上的準確率；用來報告釋出模型的效能
        （Balle 2022 Sec. V-A(b)：released models achieve over 94% accuracy on MNIST）。
    Pseudo code
        for each model i:  acc_i <- mean[ argmax f_{theta_i}(x) == y ]
    用法
        acc = accuracy_of_flat_params(spec, released_params, eval_x, eval_y)   # (n_models,)
    """
    base = build(model_spec).to(device)
    names_shapes = [(k, v.shape) for k, v in base.named_parameters()]
    sizes = [int(np.prod(s)) for _, s in names_shapes]
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)

    def acc_one(flat: torch.Tensor) -> torch.Tensor:
        # [對應 Pseudo code: f_{theta_i}] 把展平向量切回各層參數
        parts = torch.split(flat, sizes)
        params = {k: p.reshape(s) for (k, s), p in zip(names_shapes, parts)}
        pred = functional_call(base, params, (xt,)).argmax(-1)
        return (pred == yt).float().mean()

    out = []
    flat_all = torch.as_tensor(flat_params, dtype=torch.float32, device=device)
    for s in range(0, len(flat_all), chunk_size):
        out.append(vmap(acc_one)(flat_all[s:s + chunk_size]).cpu())
    return torch.cat(out).numpy()


class BatchedGDTrainer(BaseTrainer):
    """
    功能
        train_models_batched 的 BaseTrainer 包裝：組合「模型設定、損失、資料、超參數」，fit() 回傳所有模型的展平參數。
    Pseudo code
        fit(): return train_models_batched(model_spec, init_seed, fixed, extra, lr, momentum, epochs, loss_fn, ...)
    用法
        params = BatchedGDTrainer(model_spec, loss_fn, fx, fy, zx, zy, lr=0.1, momentum=0.9, epochs=100, init_seed=42).fit()
    """

    def __init__(self, model_spec: Dict[str, Any], loss_fn: torch.nn.Module, fixed_x: np.ndarray, fixed_y: np.ndarray,
                 extra_x: np.ndarray, extra_y: np.ndarray, lr: float, momentum: float, epochs: int, init_seed: int,
                 chunk_size: int = 50, device: str = "cpu", init_seeds: Optional[np.ndarray] = None, fast: Optional[bool] = None):
        self.kw = dict(model_spec=model_spec, init_seed=init_seed, fixed_x=fixed_x, fixed_y=fixed_y, extra_x=extra_x, extra_y=extra_y,
                       lr=lr, momentum=momentum, epochs=epochs, loss_fn=loss_fn, chunk_size=chunk_size, device=device,
                       init_seeds=init_seeds, fast=fast)

    def fit(self) -> np.ndarray:
        """功能：訓練並回傳 (n_models, n_params)。"""
        return train_models_batched(**self.kw)
