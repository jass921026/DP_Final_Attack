"""
檔案功能
--------
Balle, Cherubin, Hayes (IEEE S&P 2022) "Reconstructing Training Data with Informed Adversaries" 的一般重建攻擊（RecoNN）。
對手知道除一筆目標 z 之外的所有訓練資料 D-、訓練演算法 A、釋出模型參數 theta，要重建 z。
作法：用 A 在 D- ∪ {z_bar_i}（z_bar_i 來自對手的旁資訊集 D_bar）上訓練 k 個影子模型，
把「影子模型參數 -> 影子目標」當作訓練資料，訓練重建網路 phi，再把 phi 套到真正的釋出模型。

本檔提供：
    - fit_param_scaler / apply_scaler        參數標準化（論文 Sec. V-A(c)）
    - train_reconstructor                    訓練 RecoNN（論文 Sec. IV-B 步驟 2）
    - reconstruct                            套用 RecoNN（步驟 3）
    - build_attack_training_data             產生攻擊訓練資料 S（步驟 1），結果存成 RunRecord 供復用
    - run_informed_reconstruction            Algorithm 1 的完整流程與評估指標

使用方式
--------
    python run_balle.py --config configs/balle_mnist_notebook.json      # 完整流程
    python run_balle.py --config configs/balle_mnist_notebook.json --smoke   # 小規模快速檢查
或在程式中呼叫 run_informed_reconstruction(cfg)。
"""
import json
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from attacks.base import BaseAttack
from util import metrics
from util.data import make_loader
from util.record import RunRecord
from util.registry import build
from util.splits import make_disjoint_splits
from datasets.base import BaseDataset
from trainers.batched_gd import accuracy_of_flat_params, train_models_batched
from trainers.trainer import Trainer


def fit_param_scaler(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    功能
        逐座標標準化的統計量。論文 Sec. V-A(c)：we re-scale each coordinate in this representation to zero mean and unit variance;
        we found this pre-processing step to be important, as some of the parameters can be extremely small.
    Pseudo code
        mu_j  <- mean_i params[i, j]
        sd_j  <- std_i  params[i, j]       (sd_j = 0 時以 1 取代，避免除以 0)
    用法
        mu, sd = fit_param_scaler(shadow_params)
    """
    # [對應 Pseudo code]
    mu = params.mean(axis=0)
    sd = params.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(params: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """功能：套用標準化 (params - mu) / sd。用法：apply_scaler(P, mu, sd)。"""
    return ((params - mu) / sd).astype(np.float32)


RECON_LOSS_SPEC = {"module": "loss.combine", "type": "WeightedSum", "kwargs": {"terms": [
    {"weight": 1.0, "loss": {"module": "loss.regression", "type": "MAE", "kwargs": {"activation": "sigmoid"}}},
    {"weight": 1.0, "loss": {"module": "loss.regression", "type": "MSE", "kwargs": {"activation": "sigmoid"}}}]}}   # Sec. V-A(c)：MAE + MSE
CE_SPEC = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}                                      # 演算法 A 的損失


def train_reconstructor(
    params_train: np.ndarray, targets_train: np.ndarray,
    params_eval: np.ndarray, targets_eval: np.ndarray,
    recon_spec: Dict[str, Any], lr: float, batch_size: int, epochs: int, seed: int,
    device: str = "cpu", log_every: int = 10, optimizer: str = "torch_rmsprop",
    loss_spec: Dict[str, Any] = None,
) -> Tuple[torch.nn.Module, list]:
    """
    功能
        訓練重建網路 phi（RecoNN）。論文 Sec. IV-B 步驟 2：Train a RecoNN phi using S as examples of successful
        reconstructions: phi = R(S)。這裡只「組合」：建立模型、損失（loss/）、DataLoader、優化器設定，交給
        trainers.trainer.Trainer 執行訓練程序。MNIST 的訓練目標為 MAE + MSE（Sec. V-A(c)），優化器 RMSProp（Table VI）。
    Pseudo code
        輸入：S = {(theta_i, z_i)}（theta 已標準化）
        phi <- 隨機初始化（seed）
        for epoch in 1..epochs:
            隨機打亂 S 並切成大小 batch_size 的 mini-batch
            for each mini-batch B:
                z_hat <- sigmoid(phi(theta_B))
                loss  <- mean|z_hat - z_B| + mean(z_hat - z_B)^2        # MAE + MSE（loss.combine.WeightedSum）
                phi   <- RMSProp 更新(phi, grad loss)
        return phi
    用法
        phi, hist = train_reconstructor(P_tr, Z_tr, P_ev, Z_ev, spec, lr=1e-3, batch_size=128, epochs=100, seed=0)
    """
    # [對應 Pseudo code: phi <- 隨機初始化] 固定種子；輸入維度 = 參數數量，輸出維度 = 目標維度
    torch.manual_seed(seed)
    phi = build(recon_spec, in_dim=params_train.shape[1], out_dim=targets_train.shape[1])
    # [對應 Pseudo code: 損失、mini-batch] 損失來自 loss/；資料為 torch Dataset + DataLoader（每個 epoch 的打亂順序由 seed 決定）
    loss_fn = build(loss_spec or RECON_LOSS_SPEC)
    loader = make_loader(BaseDataset(params_train, targets_train.astype(np.float32)), None, batch_size, shuffle=True, seed=seed)
    xe = torch.as_tensor(params_eval, device=device)
    ye = torch.as_tensor(targets_eval, dtype=torch.float32, device=device)
    mae_fn = build({"module": "loss.regression", "type": "MAE", "kwargs": {"activation": "sigmoid"}})
    mse_fn = build({"module": "loss.regression", "type": "MSE", "kwargs": {"activation": "sigmoid"}})

    def eval_fn(model):
        out = model(xe)
        return {"eval_mae": float(mae_fn(out, ye)), "eval_mse": float(mse_fn(out, ye))}

    # [對應 Pseudo code: 迴圈與更新] 交給 Trainer；optimizer="optax_rmsprop" 複製官方 eps 位置（平方根內），"torch_rmsprop" 為標準版
    history = Trainer(phi, loss_fn, loader, {"name": optimizer, "lr": lr}, epochs, device, eval_fn=eval_fn, eval_every=log_every).fit()
    return phi, history


def reconstruct(phi: torch.nn.Module, params_scaled: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    功能
        把 RecoNN 套到釋出模型的（標準化後）參數，得到重建 z_hat。論文 Sec. IV-B 步驟 3：z_hat = phi(theta)。
    Pseudo code
        z_hat <- sigmoid(phi(theta_scaled))
    用法
        z_hat = reconstruct(phi, apply_scaler(released_params, mu, sd))
    """
    # [對應 Pseudo code]
    with torch.no_grad():
        out = torch.sigmoid(phi(torch.as_tensor(params_scaled, device=device)))
    return out.cpu().numpy()


def _init_seeds(A: Dict[str, Any], n: int, role: str):
    """
    功能
        決定每個模型的初始化種子。A["init"]="shared"（預設）：全部共用 init_seed（對手知道初始化，Sec. V-A(b)）。
        A["init"]="per_model"：Table II「Adversary does not know initial released model parameters」——
        影子模型與釋出模型各自隨機初始化，且兩組種子不重疊（釋出模型的初始化對對手是未見過的）。
    Pseudo code
        if init == shared: return None
        shadow:   seed_i = init_seed + 1 + i
        released: seed_i = init_seed + 1_000_000 + i
    用法
        init_seeds=_init_seeds(A, n_models, role="shadow" | "released")
    """
    if A.get("init", "shared") == "shared":
        return None
    base = A["init_seed"] + (1 if role == "shadow" else 1_000_000)
    return base + np.arange(n)


def _dp_for(A: Dict[str, Any], role: str):
    """功能：演算法 A 的 DP 設定（A["dp"] = {"clip", "sigma", "seed"}）；影子與釋出模型用不同的雜訊種子（各自獨立抽雜訊）。無 DP 回傳 None。"""
    dp = A.get("dp")
    if dp is None:
        return None
    return {**dp, "seed": int(dp.get("seed", 0)) + (0 if role == "shadow" else 7919)}


def dp_epsilon(A: Dict[str, Any], delta: float = 1e-5):
    """功能：全批次 Gaussian DP-GD 的精確 (eps, delta)（無子抽樣；T 步組合成 mu-GDP，mu = sqrt(T) / sigma；add/remove 鄰接）。無 DP 回傳 None。"""
    dp = A.get("dp")
    if dp is None:
        return None
    sigma = float(dp.get("sigma", 0.0))
    if sigma <= 0:
        return float("inf")
    import math
    from scipy.optimize import brentq
    from scipy.stats import norm
    mu = math.sqrt(A["epochs"]) / sigma
    f = lambda e: norm.cdf(-e / mu + mu / 2) - math.exp(min(e + norm.logcdf(-e / mu - mu / 2), 700)) - delta
    return 0.0 if f(0) <= 0 else float(brentq(f, 0, 1e9))


def build_attack_training_data(cfg: Dict[str, Any], x: np.ndarray, y: np.ndarray, splits: Dict[str, np.ndarray],
                               root: str = "runs") -> RunRecord:
    """
    功能
        產生並（若已存在則復用）影子模型-目標配對 S。論文 Sec. IV-B 步驟 1：
        For i = 1,...,k, train model theta_bar_i = A_{D-}(z_bar_i) on the fixed dataset plus the i-th shadow target
        from the adversary's side knowledge pool D_bar. S = {(theta_bar_i, z_bar_i)}_{i=1..k}。
    Pseudo code
        spec <- (資料設定, 切分, 模型, 演算法 A 的超參數, 影子數量 k)
        if RunRecord(spec) 已存在: return 它                      # 復用，不重訓
        for i in 1..k:  theta_bar_i <- A_{D-}(z_bar_i)             # 呼叫 batched_gd.train_models_batched
        存成 RunRecord(params = {theta_bar_i}, targets = {z_bar_i})
    用法
        rec = build_attack_training_data(cfg, ds.x, ds.y, splits)
        rec.arrays["params"], rec.arrays["targets"]
    """
    A = cfg["algorithm_A"]
    # [對應 Pseudo code: spec] 影響結果的一切都放進 spec，雜湊後決定能否復用
    spec = {"kind": "balle_shadow", "data": cfg["data"], "splits_seed": cfg["splits"]["seed"], "splits_sizes": cfg["splits"]["sizes"],
            "model": cfg["released_model"], "A": {**{k: A[k] for k in ("lr", "momentum", "epochs", "init_seed")}, "init": A.get("init", "shared")}, "loss": A.get("loss", CE_SPEC), "role": "shadow", **({"dp": A["dp"]} if A.get("dp") else {})}
    d = RunRecord.dir_for(root, spec, tag="balle_shadow")
    # [對應 Pseudo code: if 已存在 -> 復用]
    if RunRecord.exists_for_spec(root, spec, tag="balle_shadow"):
        return RunRecord.load(d)
    # [對應 Pseudo code: for i in 1..k]
    fixed_x, fixed_y = x[splits["fixed"]], y[splits["fixed"]]
    sh_x, sh_y = x[splits["shadow"]], y[splits["shadow"]]
    params = train_models_batched(cfg["released_model"], A["init_seed"], fixed_x, fixed_y, sh_x, sh_y,
                                  A["lr"], A["momentum"], A["epochs"], build(A.get("loss", CE_SPEC)), A.get("chunk_size", 50), cfg.get("device", "cpu"),
                                  init_seeds=_init_seeds(A, len(sh_x), role="shadow"), dp=_dp_for(A, "shadow"))
    # [對應 Pseudo code: 存成 RunRecord]
    return RunRecord.save(d, spec, {"params": params, "targets": sh_x, "labels": sh_y})


def run_informed_reconstruction(cfg: Dict[str, Any], root: str = "runs") -> Dict[str, Any]:
    """
    功能
        Balle 2022 Algorithm 1 的完整流程，並報告 MSE、NN oracle 與勝過 NN 的比例。
    Pseudo code (論文 Algorithm 1: Reconstruction attack with an informed adversary)
        procedure RECONSTRUCTION(A, R, D-, z; aux)
            theta <- A(D- ∪ {z})                    # 模型開發者訓練釋出模型
            z_hat <- R(theta, D-, A; aux)           # 對手執行攻擊（aux = 影子集 D_bar）
            return l(z, z_hat)                      # 重建誤差
        本檔對 1K 個測試目標各訓練一個釋出模型，回報平均表現（Sec. V-A(a)）。
    用法
        result = run_informed_reconstruction(json.load(open("configs/balle_mnist_notebook.json")))
    """
    t0 = time.time()
    device = cfg.get("device", "cpu")
    # [對應 資料與切分] 固定集 D-、影子集 D_bar、測試目標、評估集互不相交（Sec. V-A(a)）
    ds = build(cfg["data"])
    splits = make_disjoint_splits(len(ds), cfg["splits"]["sizes"], cfg["splits"]["seed"])

    # [對應 步驟 1 / Algorithm 1 對手端] 影子模型 -> 攻擊訓練資料 S
    S = build_attack_training_data(cfg, ds.x, ds.y, splits, root)

    # [對應 Algorithm 1 第一行 theta <- A(D- ∪ {z})] 對每個測試目標 z 訓練釋出模型
    A = cfg["algorithm_A"]
    fixed_x, fixed_y = ds.get(splits["fixed"])
    tgt_x, tgt_y = ds.get(splits["test_targets"])
    released = train_models_batched(cfg["released_model"], A["init_seed"], fixed_x, fixed_y, tgt_x, tgt_y,
                                    A["lr"], A["momentum"], A["epochs"], build(A.get("loss", CE_SPEC)), A.get("chunk_size", 50), device,
                                    init_seeds=_init_seeds(A, len(tgt_x), role="released"), dp=_dp_for(A, "released"))
    ev_x, ev_y = ds.get(splits["eval"])
    acc = accuracy_of_flat_params(cfg["released_model"], released, ev_x, ev_y, device)

    # [對應 步驟 2] 標準化參數並訓練 RecoNN
    R = cfg["reconstructor"]
    fit_on = S.arrays["params"] if R.get("fit_scaler_on_train_only", True) else np.concatenate([S.arrays["params"], released])
    mu, sd = fit_param_scaler(fit_on)
    phi, hist = train_reconstructor(apply_scaler(S.arrays["params"], mu, sd), S.arrays["targets"],
                                    apply_scaler(released, mu, sd), tgt_x, R["model"], R["lr"], R["batch_size"],
                                    R["epochs"], R["seed"], device, optimizer=R.get("optimizer", "torch_rmsprop"), loss_spec=R.get("loss"))

    # [對應 步驟 3 / Algorithm 1 第二行 z_hat <- R(...)] 重建
    z_hat = reconstruct(phi, apply_scaler(released, mu, sd), device)

    # [對應 Algorithm 1 第三行 l(z, z_hat) 與 Sec. V-B] 指標
    mse = metrics.mse_per_sample(z_hat, tgt_x)
    pool = np.concatenate([fixed_x, S.arrays["targets"]])           # 對手可用資料池 D- ∪ D_bar
    nn = metrics.nn_oracle_mse(tgt_x, pool)
    result = {
        "name": cfg.get("name"), "n_shadow": int(len(S.arrays["params"])), "n_targets": int(len(tgt_x)),
        "released_acc_mean": float(acc.mean()), "dp": A.get("dp"), "dp_epsilon_delta1e-5": dp_epsilon(A), "mse_mean": float(mse.mean()), "mse_median": float(np.median(mse)),
        "nn_oracle_mse_mean": float(nn.mean()), "frac_better_than_nn": metrics.frac_better_than_nn(mse, nn),
        "shadow_record": S.record_id, "history": hist, "seconds": round(time.time() - t0, 1),
    }
    out_dir = os.path.join(root, "results", f"{cfg.get('name', 'balle')}_{S.record_id}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, "reconstructions.npz"), z_hat=z_hat, targets=tgt_x, mse=mse, nn=nn)
    return result


class BalleReconstructionAttack(BaseAttack):
    """
    功能
        Balle 2022 重建攻擊的 BaseAttack 實作：run() = run_informed_reconstruction(cfg, root)。
    用法
        BalleReconstructionAttack(cfg, root="runs").run()
    """

    def run(self) -> Dict[str, Any]:
        """功能：執行 Algorithm 1 的完整流程並回傳指標。"""
        return run_informed_reconstruction(self.cfg, self.root)
