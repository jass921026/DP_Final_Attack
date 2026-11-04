"""
檔案功能
--------
命令列入口：讀取設定檔，執行 Balle et al. 2022 的 informed-adversary 重建實驗（attacks/balle_recon.py 的 Algorithm 1），
並把指標印出、存到 runs/results/。

使用方式
--------
    python run_balle.py --config configs/balle_mnist_notebook.json
    python run_balle.py --config configs/balle_mnist_notebook.json --smoke     # 縮小規模檢查流程（不代表論文結果）
    python run_balle.py --config ... --root runs                               # 改變 RunRecord / 結果的存放根目錄
--smoke 只縮小：影子模型 300、測試目標 100、固定集 200、重建器 20 epochs。
"""
import argparse
import copy
import json

from attacks.balle_recon import BalleReconstructionAttack


def apply_smoke(cfg: dict) -> dict:
    """
    功能
        把設定縮小到幾分鐘內可跑完（只用於檢查流程）。
    用法
        cfg = apply_smoke(cfg)
    """
    # [對應 --smoke 說明] 逐項縮小，並在 name 上標記以免與正式結果混淆
    c = copy.deepcopy(cfg)
    c["name"] = c["name"] + "_smoke"
    c["splits"]["sizes"].update({"eval": 500, "test_targets": 100, "fixed": 200, "shadow": 300})
    c["reconstructor"]["epochs"] = 20
    return c


if __name__ == "__main__":
    # [對應 檔頭使用方式] 解析參數 -> 載入設定 -> 執行 -> 印出結果
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.smoke:
        cfg = apply_smoke(cfg)
    res = BalleReconstructionAttack(cfg, root=a.root).run()
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))
