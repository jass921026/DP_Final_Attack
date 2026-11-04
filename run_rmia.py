"""
檔案功能
--------
RMIA 實驗的命令列入口：讀設定檔、呼叫 attacks.rmia_experiment.run_rmia_experiment、印出彙整表。

使用方式
--------
    python run_rmia.py --config configs/rmia_cifar10_small.json            # 完整設定
    python run_rmia.py --config configs/rmia_cifar10_small.json --smoke    # 縮小（4 個模型、1 epoch、審計 2000）
"""
import argparse
import json

from attacks.rmia import RMIAAttack


def apply_smoke(cfg: dict) -> dict:
    """功能：縮小規模以快速檢查流程。用法：cfg = apply_smoke(cfg)。"""
    cfg = json.loads(json.dumps(cfg))
    cfg["name"] += "_smoke"
    cfg["train"]["epochs"] = 1
    cfg["models"]["num_pairs"] = 3
    cfg["audit"]["size"] = 2000
    cfg["population"]["size"] = 500
    cfg["attack"]["num_ref"] = [1, 2]
    cfg["attack"]["targets"] = [0, 1]
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.smoke:
        cfg = apply_smoke(cfg)
    res = RMIAAttack(cfg, a.root).run()
    print(f"models {res['n_models']}, audit {res['n_audit']}, pop {res['n_pop']}, train/test acc proxy {res['mean_train_acc_proxy']:.3f}/{res['mean_test_acc_proxy']:.3f}")
    print(f"{'R':>3} {'gamma':>5} {'a':>6} {'AUC':>14} {'TPR@1%':>14} {'TPR@0.1%':>14} {'TPR@0%':>10} {'maxdiff':>9}")
    for e in res["summary"]:
        print(f"{e['R']:>3} {e['gamma']:>5g} {e['a_mode']:>6} {e['auc']:.4f}±{e['auc_sd']:.4f} {e['tpr@0.01']:.4f}±{e['tpr@0.01_sd']:.4f} "
              f"{e['tpr@0.001']:.4f}±{e['tpr@0.001_sd']:.4f} {e['tpr@0']:.4f} {e.get('max_abs_diff_vs_official', float('nan')):>9.1e}")
    print(f"seconds {res['seconds']}")
