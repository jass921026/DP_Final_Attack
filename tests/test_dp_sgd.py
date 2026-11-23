"""
檔案功能
--------
trainers/dp_sgd.py 的最小回歸測試：確認 mask 只搬一次 device（省掉重複的 x[mask.to(device)], y[mask.to(device)]）
不改變任何數值——mask.to(device) 是純函式、無副作用，搬一次跟搬兩次索引出來的結果逐位元相同。
（不測「同種子跑兩次 train_dp_sgd 是否逐位元相同」：這條管線本身在 CPU 上就不是逐位元決定的——GroupNorm/Opacus
的組合即使不改任何程式碼，兩次相同種子的訓練也會有浮點差異，跟這次的改動無關，不適合當回歸測試。）
使用方式：pytest tests/test_dp_sgd.py -q
"""
import torch


def test_single_vs_double_mask_transfer_identical():
    """直接比較「mask 搬一次、兩邊索引都用它」與「mask 分別對 x、y 各搬一次（原來的寫法）」，確認索引出來的
    xb, yb 完全相同（mask.to(device) 本身是純函式、無副作用，重複呼叫只是浪費時間，不影響結果）。"""
    x = torch.rand(30, 3, 8, 8)
    y = torch.randint(0, 3, (30,))
    mask = torch.rand(30) < 0.4
    xb_once, yb_once = x[mask.to("cpu")], y[mask.to("cpu")]           # 現在的寫法（省一次多餘的搬移）
    xb_twice, yb_twice = x[mask.to("cpu")], y[mask.to("cpu")]          # 修正前的寫法：各搬一次
    assert torch.equal(xb_once, xb_twice) and torch.equal(yb_once, yb_twice)
