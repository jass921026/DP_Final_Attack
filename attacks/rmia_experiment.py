"""
檔案功能
--------
設定檔驅動的 RMIA 實驗（離線）：訓練成對模型（互補的兩半訓練集）-> 計算並快取 softmax 訊號（RunRecord）->
對每個目標模型、參考模型數 R、γ、a 執行 RMIA -> 彙整 AUC / TPR@FPR。同時在真實訊號上呼叫官方 run_rmia（γ=1）
比對分數，作為一致性檢查。訓練與訊號只做一次，依設定雜湊復用。

使用方式
--------
    python run_rmia.py --config configs/rmia_cifar10_small.json [--smoke]
"""
import importlib.util
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

import torch

from attacks.rmia import model_signals, rmia_attack, roc_summary, tune_offline_a
from util.data import make_loader
from util.record import RunRecord
from util.registry import build
from trainers.trainer import Trainer, random_hflip


def make_pair_memberships(n: int, num_pairs: int, seed: int) -> np.ndarray:
    """
    功能
        成員矩陣 (n, 2*num_pairs)：第 i 對的兩個模型各訓練在互補的一半資料上（官方 split_dataset_for_training 的平衡切分）。
    Pseudo code
        for pair i: perm <- RNG(seed, i).permutation(n); 模型 2i 的成員 <- perm 前半；模型 2i+1 <- 後半
    用法
        mem = make_pair_memberships(50000, 9, seed=0)
    """
    mem = np.zeros((n, 2 * num_pairs), dtype=bool)
    for i in range(num_pairs):
        perm = np.random.default_rng([seed, i]).permutation(n)
        mem[perm[: n // 2], 2 * i] = True
        mem[perm[n // 2:], 2 * i + 1] = True
    return mem


def build_signals(cfg: Dict[str, Any], root: str) -> RunRecord:
    """
    功能
        訓練所有模型並計算審計樣本與母體上的訊號；依 spec 雜湊復用。本函式只「組合」：資料集（torch Dataset）、
        模型、損失（loss/）、DataLoader 與 Trainer 都由設定檔建構。
    Pseudo code
        spec <- (資料、模型、損失、優化器、訓練、成員切分、審計抽樣設定)
        if RunRecord(spec) 存在: return
        mem <- make_pair_memberships;  audit_idx <- 依目標 0 的成員/非成員各取一半（官方 sample_auditing_dataset）
        for 每個模型 j: theta_j <- Trainer(model, loss, DataLoader(成員 j)).fit()
                         signals[:, j] <- softmax(theta_j, audit);  pop[:, j] <- softmax(theta_j, population)
    用法
        rec = build_signals(cfg, "runs")   # rec.arrays: signals, pop_signals, memberships, audit_idx, train_acc, test_acc
    """
    T = cfg["train"]
    loss_spec = T.get("loss", {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}})
    opt_spec = T.get("optimizer", {"name": "sgd", "lr": T["lr"], "momentum": T.get("momentum", 0.9), "weight_decay": T.get("weight_decay", 0.0)})
    spec = {"kind": "rmia_signals", "data": cfg["data"], "model": cfg["model"], "train": {"epochs": T["epochs"], "batch_size": T["batch_size"],
            "flip": T.get("flip", True), "loss": loss_spec, "optimizer": opt_spec, **({"dp": T["dp"]} if T.get("dp") else {})}, "models": cfg["models"], "audit": cfg["audit"], "population": cfg["population"], **({"universe": cfg["universe"]} if cfg.get("universe") else {})}
    d = RunRecord.dir_for(root, spec, tag="rmia_signals")
    if RunRecord.exists_for_spec(root, spec, tag="rmia_signals"):
        return RunRecord.load(d)
    # [對應 Pseudo code: 資料] 資料集 = train split（成員／非成員來源），母體 = test split；兩者都是 torch Dataset
    train_ds, pop_ds = build(cfg["data"]), build(cfg["data"], split="test")
    if cfg.get("universe"):                       # 只用訓練集的一個子集當「全部資料」（縮小每個模型的訓練集）；成員矩陣、審計集都在這個子集內
        from torch.utils.data import Subset
        u = np.random.default_rng(cfg["universe"]["seed"]).choice(len(train_ds), cfg["universe"]["size"], replace=False)
        train_ds = Subset(train_ds, u)
    n = len(train_ds)
    M = 2 * cfg["models"]["num_pairs"]
    mem_full = make_pair_memberships(n, cfg["models"]["num_pairs"], cfg["models"]["seed"])
    # [對應 Pseudo code: audit_idx]
    rng = np.random.default_rng(cfg["audit"]["seed"])
    size = cfg["audit"]["size"]
    if size < n:
        a_in = rng.choice(np.where(mem_full[:, 0])[0], size // 2, replace=False)
        a_out = rng.choice(np.where(~mem_full[:, 0])[0], size // 2, replace=False)
        audit_idx = np.concatenate([a_in, a_out])
    else:
        audit_idx = np.arange(n)
    pop_idx = np.random.default_rng(cfg["population"]["seed"]).choice(len(pop_ds), cfg["population"]["size"], replace=False)
    device = T.get("device", "cpu")
    fast = bool(T.get("fast", False))             # 選用的加速路徑：只省掉每個模型三次訊號呼叫之間來回搬移整個模型的時間；
                                                    # 預設 False，現有設定檔跟之前完全相同（原版）
    if fast and device == "cuda":
        torch.backends.cudnn.benchmark = True      # 同形狀的卷積重複很多次（每個模型都訓練），讓 cudnn 挑最快的演算法
    # 注意：曾經試過把推論 batch size 一起調大（例如 1000 -> 4096），實測在 MPS 上某些呼叫順序下會觸發嚴重變慢
    # （同一個 batch size、同一段程式碼，單獨跑是 4s，接在別的 batch size 呼叫之後卻變成 170 幾秒，原因不明，
    # 懷疑是 PyTorch MPS 後端 allocator 在切換不同形狀後的某種退化路徑）。不值得冒這個風險，所以 batch size
    # 維持原來的值，除非使用者自己用 signal_batch_size 覆寫（風險自負，建議先用 analysis 腳本量過再用在真正的實驗上）。
    sig_batch = T.get("signal_batch_size", 1000)
    sig = np.zeros((len(audit_idx), M), np.float32)
    pop = np.zeros((len(pop_idx), M), np.float32)
    tr_acc, te_acc = np.zeros(M), np.zeros(M)
    dp_info = np.zeros((M, 2))
    # 逐模型檢查點：中斷（如 Colab 斷線）後從已完成的模型接續；只在規格相同時才會讀到（目錄由 spec 雜湊決定）
    os.makedirs(d, exist_ok=True)
    part = os.path.join(d, "partial.npz")
    done = 0
    if os.path.exists(part):
        z = np.load(part)
        done = int(z["done"]); sig, pop, tr_acc, te_acc = z["sig"], z["pop"], z["tr_acc"], z["te_acc"]
        dp_info = z["dp_info"] if "dp_info" in z.files else dp_info
        print(f"resume from model {done}/{M}", flush=True)
    for j in range(done, M):
        t0 = time.time()
        keep = np.where(mem_full[:, j])[0]
        # [對應 Pseudo code: theta_j <- Trainer(...).fit()] 先固定種子再建模型（初始化由種子決定）
        torch.manual_seed(1000 + j)
        model = build(cfg["model"])
        if T.get("dp"):
            from trainers.dp_sgd import train_dp_sgd
            dpc = T["dp"]
            xk, yk = train_ds.__getitems__(list(keep))
            info = train_dp_sgd(model, xk, yk, T["epochs"], opt_spec["lr"], opt_spec.get("momentum", 0.9), T["batch_size"], dpc["clip"],
                                dpc.get("epsilon"), dpc.get("delta", 1e-5), device, 1000 + j, random_hflip if T.get("flip", True) else None)
            dp_info[j] = [info["sigma"], info["epsilon"]]
        else:
            loader = make_loader(train_ds, keep, T["batch_size"], shuffle=True, seed=1000 + j)
            Trainer(model, build(loss_spec), loader, opt_spec, T["epochs"], device, augment=random_hflip if T.get("flip", True) else None).fit()
        # [對應 Pseudo code: signals, pop]（fast=True 時模型留在 device，三次呼叫不來回搬移、也用更大的推論 batch；
        # 搬不搬本身不影響數值，但較大的推論 batch 會改變矩陣乘法內部加總順序，float32 下有 ~1e-7 量級的捨入差異，
        # 見 tests/test_rmia_fast.py）
        sig[:, j] = model_signals(model, train_ds, audit_idx, device, sig_batch, move=not fast)[:, 0]
        pop[:, j] = model_signals(model, pop_ds, pop_idx, device, sig_batch, move=not fast)[:, 0]
        p_tr = model_signals(model, train_ds, keep[:5000], device, sig_batch, move=not fast)[:, 0]
        if fast:
            model.cpu()
        tr_acc[j] = float((p_tr > 0.5).mean())                       # 以「真實類別機率>0.5」近似正確（二者在多數情形等價）
        te_acc[j] = float((pop[:, j] > 0.5).mean())
        print(f"model {j + 1}/{M} trained+signals in {time.time() - t0:.0f}s  train(p>.5) {tr_acc[j]:.3f}  test(p>.5) {te_acc[j]:.3f}", flush=True)
        np.savez(part, done=j + 1, sig=sig, pop=pop, tr_acc=tr_acc, te_acc=te_acc, dp_info=dp_info)
    return RunRecord.save(d, spec, {"signals": sig.astype(np.float64), "pop_signals": pop.astype(np.float64),
                                    "memberships": mem_full[audit_idx], "audit_idx": audit_idx, "train_acc": tr_acc, "test_acc": te_acc, "dp_sigma_eps": dp_info})


def _official_run_rmia(path: str):
    """功能：以檔案路徑載入官方 attacks.py（不需安裝）。用法：_official_run_rmia(cfg['official_attacks_path']).run_rmia(...)。"""
    spec = importlib.util.spec_from_file_location("official_rmia_attacks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rmia_experiment(cfg: Dict[str, Any], root: str = "runs") -> Dict[str, Any]:
    """
    功能
        整個實驗：build_signals -> 對每個 (目標, R, γ, a 模式) 攻擊並評估 -> 彙整（跨目標平均 ± 標準差）。
    Pseudo code
        rec <- build_signals(cfg)
        for target in cfg.attack.targets:
            for R in cfg.attack.num_ref:
                for gamma in cfg.attack.gamma:
                    a <- tune_offline_a(target, ...) 若 a_mode == "tune"，否則固定值
                    scores <- rmia_attack(...);  stats <- roc_summary(scores, memberships[:, target])
                若 gamma == 1 且 selection == master：另以官方 run_rmia 算分數，記錄最大絕對差
        彙整：每個 (R, gamma, a_mode) 在各目標上的平均與標準差
    用法
        run_rmia_experiment(json.load(open("configs/rmia_cifar10_small.json")))
    """
    t0 = time.time()
    rec = build_signals(cfg, root)
    A = rec.arrays
    at = cfg["attack"]
    off = _official_run_rmia(cfg["official_attacks_path"]) if at.get("check_official", True) and os.path.isfile(cfg.get("official_attacks_path", "")) else None
    rows: List[Dict[str, Any]] = []
    for tgt in at["targets"]:
        for R in at["num_ref"]:
            for gamma in at["gamma"]:
                for a_mode in at["a"]:
                    a = tune_offline_a(tgt, A["signals"], A["pop_signals"], A["memberships"], gamma, at.get("selection", "master"))[0] if a_mode == "tune" else float(a_mode)
                    s = rmia_attack(A["signals"], A["pop_signals"], A["memberships"], tgt, R, a, gamma, at.get("selection", "master"))
                    row = {"target": tgt, "R": R, "gamma": gamma, "a": a, "a_mode": str(a_mode),
                           **roc_summary(s, A["memberships"][:, tgt], fprs=(0.0, 0.0001, 0.001, 0.01, 0.1))}
                    if off is not None and gamma == 1.0 and at.get("selection", "master") == "master":
                        ref = off.run_rmia(tgt, A["signals"], A["pop_signals"], A["memberships"], R, a)
                        row["max_abs_diff_vs_official"] = float(np.abs(ref - s).max())
                    rows.append(row)
    # [對應 Pseudo code: 彙整]
    summary = []
    for key in sorted({(r["R"], r["gamma"], r["a_mode"]) for r in rows}):
        sel = [r for r in rows if (r["R"], r["gamma"], r["a_mode"]) == key]
        entry = {"R": key[0], "gamma": key[1], "a_mode": key[2], "n_targets": len(sel), "mean_a": float(np.mean([r["a"] for r in sel]))}
        for m in ("auc", "tpr@0", "tpr@0.0001", "tpr@0.001", "tpr@0.01", "tpr@0.1"):
            v = np.array([r[m] for r in sel])
            entry[m] = float(v.mean()); entry[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        diffs = [r["max_abs_diff_vs_official"] for r in sel if "max_abs_diff_vs_official" in r]
        if diffs:
            entry["max_abs_diff_vs_official"] = float(max(diffs))
        summary.append(entry)
    out = {"name": cfg["name"], "signals_record": rec.record_id, "n_audit": int(A["signals"].shape[0]), "n_pop": int(A["pop_signals"].shape[0]),
           "n_models": int(A["signals"].shape[1]), "mean_test_acc_proxy": float(A["test_acc"].mean()), "mean_train_acc_proxy": float(A["train_acc"].mean()),
           "summary": summary, "rows": rows, "seconds": round(time.time() - t0, 1)}
    d = os.path.join(root, "results", cfg["name"])
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "result.json"), "w"), indent=1)
    return out
