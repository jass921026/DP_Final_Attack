"""
檔案功能
--------
Robbing the Fed（imprint）FedSGD 實驗的命令列入口：讀設定檔、呼叫 attacks.imprint.run_imprint_fedsgd、印出結果表。

使用方式
--------
    python run_imprint.py --config configs/imprint_fedsgd_bins.json            # 完整設定
    python run_imprint.py --config configs/imprint_fedsgd_bins.json --smoke    # 縮小（8 張、2 個 bins 設定）
"""
import argparse
import json

from attacks.imprint import ImprintAttack


def apply_smoke(cfg: dict) -> dict:
    """功能：縮小規模以快速檢查流程。用法：cfg = apply_smoke(cfg)。"""
    cfg = json.loads(json.dumps(cfg))
    cfg["name"] += "_smoke"
    cfg["data"]["kwargs"]["n"] = 8
    cfg["sweep"]["num_bins"] = cfg["sweep"]["num_bins"][:2]
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
    res = ImprintAttack(cfg, a.root).run()
    print(f"{'bins':>6} {'noise':>8} {'hits':>5} {'theory':>7} {'perfect':>8} {'PSNR>30':>8} {'meanPSNR':>9} {'PSNR160':>8} {'medPSNR':>8} {'MSE':>8} {'IIP':>6}")
    for r in res["rows"]:
        print(f"{r['num_bins']:>6} {r['noise']:>8g} {r['n_hits']:>5} {r['theory_perfect']:>7.3f} {r['frac_perfect']:>8.3f} "
              f"{r['frac_psnr_gt30']:>8.3f} {r['mean_psnr']:>9.1f} {r['mean_psnr_cap160']:>8.1f} {r['median_psnr']:>8.1f} {r['mean_mse']:>8.4f} {r.get('iip', float('nan')):>6.3f}")
    print(f"seconds {res['seconds']}")
