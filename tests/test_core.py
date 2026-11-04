"""
檔案功能
--------
util/ 的單元測試（秒級，不需網路）：splits 的互不相交與確定性、RunRecord 的存取與防錯配、NN oracle 與暴力法一致。
使用方式：在 framework/ 下執行  pytest tests/test_core.py -q
"""
import numpy as np
import pytest

from util.metrics import frac_better_than_nn, mse_per_sample, nn_oracle_mse
from util.record import RunRecord, spec_hash
from util.splits import index_hash, make_disjoint_splits


def test_splits_disjoint_and_deterministic():
    """[驗證 util/splits.py: make_disjoint_splits] 區段互不相交、大小正確、同 seed 結果相同、不同 seed 不同。"""
    s1 = make_disjoint_splits(1000, {"a": 100, "b": 200, "c": 50}, seed=1)
    s2 = make_disjoint_splits(1000, {"a": 100, "b": 200, "c": 50}, seed=1)
    s3 = make_disjoint_splits(1000, {"a": 100, "b": 200, "c": 50}, seed=2)
    allidx = np.concatenate(list(s1.values()))
    assert len(allidx) == len(set(allidx.tolist())) == 350      # 互不相交
    assert all(np.array_equal(s1[k], s2[k]) for k in s1)         # 確定性
    assert not np.array_equal(s1["a"], s3["a"])                  # seed 有作用
    assert index_hash(s1["a"]) == index_hash(s2["a"]) != index_hash(s3["a"])


def test_splits_too_large_raises():
    """[驗證 make_disjoint_splits 步驟 3] 要求總數超過資料量時報錯。"""
    with pytest.raises(ValueError):
        make_disjoint_splits(10, {"a": 8, "b": 8}, seed=0)


def test_record_roundtrip_and_tamper(tmp_path):
    """[驗證 util/record.py] 存取一致；改動 spec 後 record_id 核對失敗；exists_for_spec 能找到相同 spec。"""
    spec = {"seed": 0, "epochs": 5}
    arrays = {"p": np.arange(6, dtype=np.float32).reshape(2, 3)}
    d = RunRecord.dir_for(str(tmp_path), spec, tag="t")
    RunRecord.save(d, spec, arrays)
    rec = RunRecord.load(d)
    assert rec.spec == spec and np.array_equal(rec.arrays["p"], arrays["p"]) and rec.record_id == spec_hash(spec)
    assert RunRecord.exists_for_spec(str(tmp_path), spec, tag="t")
    assert not RunRecord.exists_for_spec(str(tmp_path), {"seed": 1, "epochs": 5}, tag="t")
    open(f"{d}/spec.json", "w").write('{"seed":9,"epochs":5}')    # 竄改 spec
    with pytest.raises(ValueError):
        RunRecord.load(d)


def test_nn_oracle_matches_bruteforce():
    """[驗證 util/metrics.py: nn_oracle_mse] 與逐一暴力計算的最近鄰 MSE 相同（含跨 chunk 邊界）。"""
    rng = np.random.default_rng(0)
    t, p = rng.random((37, 12)), rng.random((90, 12))
    brute = np.array([((p - ti) ** 2).mean(1).min() for ti in t])
    assert np.allclose(nn_oracle_mse(t, p, chunk=10), brute)


def test_mse_and_fraction():
    """[驗證 mse_per_sample 與 frac_better_than_nn]"""
    a, b = np.zeros((3, 4)), np.ones((3, 4))
    assert np.allclose(mse_per_sample(a, b), 1.0)
    assert frac_better_than_nn(np.array([0.1, 0.5, 0.9]), np.array([0.5, 0.5, 0.5])) == pytest.approx(1 / 3)
