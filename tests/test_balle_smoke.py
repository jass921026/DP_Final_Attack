"""
檔案功能
--------
Balle 重建攻擊的端到端煙霧測試：用合成資料跑完 Algorithm 1（影子模型 -> RecoNN -> 重建 -> 指標），
檢查流程可執行、輸出欄位齊全、RunRecord 會被復用。它不驗證攻擊強度（那由真實資料的實驗負責）。
使用方式：pytest tests/test_balle_smoke.py -q
"""
import numpy as np

from attacks.balle_recon import run_informed_reconstruction


def _cfg():
    return {
        "name": "smoke_synth", "device": "cpu",
        "data": {"module": "datasets.synthetic_images", "type": "SyntheticImages", "kwargs": {"n": 800, "dim": 16, "n_classes": 4, "seed": 0}},
        "splits": {"seed": 0, "sizes": {"eval": 100, "test_targets": 20, "fixed": 60, "shadow": 200}},
        "released_model": {"module": "models.mlp", "type": "MLP", "kwargs": {"in_dim": 16, "hidden": [6], "out_dim": 4, "activation": "elu"}},
        "algorithm_A": {"lr": 0.1, "momentum": 0.9, "epochs": 20, "init_seed": 1, "chunk_size": 50},
        "reconstructor": {"model": {"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": [64, 64], "activation": "relu"}},
                          "lr": 0.001, "batch_size": 32, "epochs": 5, "seed": 0, "fit_scaler_on_train_only": True},
    }


def test_pipeline_runs_and_reuses_record(tmp_path):
    """[驗證 attacks/balle_recon.py: run_informed_reconstruction] 欄位齊全、數值合法；第二次執行復用影子 RunRecord（結果相同）。"""
    r1 = run_informed_reconstruction(_cfg(), root=str(tmp_path))
    for k in ("n_shadow", "n_targets", "released_acc_mean", "mse_mean", "nn_oracle_mse_mean", "frac_better_than_nn", "shadow_record"):
        assert k in r1
    assert r1["n_shadow"] == 200 and r1["n_targets"] == 20
    assert 0 <= r1["frac_better_than_nn"] <= 1 and np.isfinite(r1["mse_mean"])
    r2 = run_informed_reconstruction(_cfg(), root=str(tmp_path))
    assert r2["shadow_record"] == r1["shadow_record"]
