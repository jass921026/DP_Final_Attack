"""
檔案功能
--------
RMIA（attacks/rmia.py）的測試：
    1. 與官方 ml_privacy_meter master 的 attacks.run_rmia / tune_offline_a 逐元素一致（γ=1，selection="master"；以路徑載入官方檔案）
    2. γ 與論文版選取：γ 變大分數單調不增；paper 選取在每列皆有 >=R 個 OUT 時與 master 選取一致
    3. 評估函式與官方存下的結果（colab 的 attack_result_*.npz：官方算出的 auc / tpr@0.1 fpr / tpr@0 fpr）一致
使用方式：pytest tests/test_rmia.py -q
"""
import glob
import importlib.util
import logging
import os

import numpy as np
import pytest

from attacks.rmia import rmia_attack, roc_summary, select_out_signals, tune_offline_a

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "rmia")
OFFICIAL = os.path.join(ROOT, "ml_privacy_meter", "attacks.py")


def _official():
    spec = importlib.util.spec_from_file_location("official_rmia_attacks", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_signals(seed=0, n=300, m=200, n_models=12):
    """隨機但有結構的訊號：成員（在該模型訓練集中）的機率較高；模型成對，互補成員。"""
    rng = np.random.default_rng(seed)
    mem = np.zeros((n, n_models), dtype=bool)
    for p in range(n_models // 2):
        keep = rng.random(n) < 0.5
        mem[:, 2 * p], mem[:, 2 * p + 1] = keep, ~keep
    base = rng.beta(2, 2, size=(n, 1))
    sig = np.clip(base + 0.25 * mem + 0.1 * rng.standard_normal((n, n_models)), 1e-4, 1.0)
    pop = np.clip(rng.beta(2, 2, size=(m, 1)) + 0.1 * rng.standard_normal((m, n_models)), 1e-4, 1.0)
    return sig, pop, mem


@pytest.mark.skipif(not os.path.isfile(OFFICIAL), reason="official ml_privacy_meter checkout not found")
@pytest.mark.parametrize("num_ref,a,target", [(1, 0.3, 0), (2, 0.3, 3), (4, 0.6, 2), (5, 0.0, 1)])
def test_rmia_matches_official(num_ref, a, target):
    """[驗證 rmia_attack] 與官方 run_rmia 的分數完全相同（含 R=1、a=0、奇數目標）。"""
    sig, pop, mem = _fake_signals()
    ours = rmia_attack(sig, pop, mem, target, num_ref, a, gamma=1.0, selection="master")
    ref = _official().run_rmia(target, sig, pop, mem, num_ref, a)
    assert np.array_equal(ours, ref)


@pytest.mark.skipif(not os.path.isfile(OFFICIAL), reason="official ml_privacy_meter checkout not found")
def test_tune_offline_a_matches_official():
    """[驗證 tune_offline_a] 選出的 a 與官方相同。"""
    sig, pop, mem = _fake_signals(1)
    a_ours, _ = tune_offline_a(0, sig, pop, mem)
    a_ref, _, _ = _official().tune_offline_a(0, sig, pop, mem, logging.getLogger("t"))
    assert abs(a_ours - a_ref) < 1e-9


def test_gamma_monotone_and_selection_paper():
    """[驗證 γ 與 paper 選取] γ 越大分數越小（單調不增）；paper 選取 = 每個樣本依欄位順序的前 R 個 OUT 訊號（不足以 NaN 補齊）。"""
    sig, pop, mem = _fake_signals(2)
    s1 = rmia_attack(sig, pop, mem, 0, 2, 0.3, gamma=1.0)
    s2 = rmia_attack(sig, pop, mem, 0, 2, 0.3, gamma=2.0)
    assert (s2 <= s1 + 1e-12).all() and s2.mean() < s1.mean()
    R, target = 3, 0
    got = select_out_signals(sig, mem, target, R, "paper")
    cols = [i for i in range(sig.shape[1]) if i not in (target, 1)][: 2 * R]            # 目標 0 的配對模型是 1
    for i in range(len(sig)):
        want = [sig[i, c] for c in cols if not mem[i, c]][:R]
        want += [np.nan] * (R - len(want))
        assert np.allclose(got[i], want, equal_nan=True)


def test_roc_summary_matches_official_saved_results():
    """[驗證 roc_summary] 對官方存下的分數，AUC、TPR@0.1% FPR（官方欄位 one_tenth_fpr）、TPR@0% FPR（zero_fpr）與官方 npz 一致。"""
    files = sorted(glob.glob(os.path.join(ROOT, "colab_output", "speedyresnet_t10_rerun", "results_*_ref8", "exp", "attack_result_*.npz")))
    if not files:
        pytest.skip("official saved results not found")
    for f in files[:3]:
        z = np.load(f, allow_pickle=True)
        mine = roc_summary(z["scores"], z["memberships"], fprs=(0.0, 0.001))
        assert abs(mine["auc"] - float(z["auc"])) < 1e-9
        assert abs(mine["tpr@0.001"] - float(z["one_tenth_fpr"])) < 1e-9
        assert abs(mine["tpr@0"] - float(z["zero_fpr"])) < 1e-9
