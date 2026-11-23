"""
檔案功能
--------
P12 資料集（datasets/p12.py）的檢查：形狀、遮罩與值的一致性、標準化只用訓練集統計、無 NaN。HW1 資料不在時跳過。
使用方式：pytest tests/test_p12.py -q
"""
import os

import numpy as np
import pytest

from datasets.p12 import DEFAULT_ROOT, P12

pytestmark = pytest.mark.skipif(not os.path.isdir(DEFAULT_ROOT), reason="HW1 PhysioNet12 data not available")


@pytest.fixture(scope="module")
def sets():
    return {(s, m): P12(s, m) for s in ("train", "test") for m in ("values", "values_mask")}


def test_shapes_and_labels(sets):
    tr, te = sets[("train", "values_mask")], sets[("test", "values_mask")]
    assert tr.x.shape == (3992, 13, 82) and te.x.shape == (3992, 13, 82)
    assert sets[("train", "values")].x.shape == (3992, 13, 41)
    assert set(np.unique(tr.y)) == {0, 1} and 0.10 < tr.y.mean() < 0.18


def test_no_nan_and_padding_is_zero(sets):
    for d in sets.values():
        assert np.isfinite(d.x).all()
    d = sets[("train", "values_mask")]
    pad = np.arange(13)[None, :] >= d.lengths[:, None]
    assert (d.x[pad] == 0).all()                                   # 填充格：值與遮罩都是 0


def test_mask_matches_values_and_is_sparse(sets):
    d = sets[("train", "values_mask")]
    m = d.x[..., 41:]
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert 0.1 < m[np.arange(13)[None, :] < d.lengths[:, None]].mean() < 0.5      # 大部分格子沒有真實觀測
    assert np.array_equal(d.x[..., :41], sets[("train", "values")].x)             # 兩種 mode 的值部分相同


def test_standardised_with_train_stats_only(sets):
    tr, te = sets[("train", "values")], sets[("test", "values")]
    assert np.array_equal(tr.mean, te.mean) and np.array_equal(tr.std, te.std)    # 測試集用的是訓練集的統計
    m = sets[("train", "values_mask")].x[..., 41:].astype(bool)
    v = tr.x
    for f in (15, 18):                                              # HR、MAP：觀測到的值平均約 0、標準差約 1
        assert abs(v[..., f][m[..., f]].mean()) < 1e-3 and abs(v[..., f][m[..., f]].std() - 1) < 1e-3
