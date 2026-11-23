"""
檔案功能
--------
P12（set-a 訓練、set-b 測試）上 iTransformer 的基線：比較只用值（41 個變數）與值加觀測遮罩（82 個變數）。
固定訓練輪數、報告最後一輪的測試 AUROC / AUPRC（不用測試集選輪數）。損失用 pos_weight 處理約 1:6 的類別不平衡。
使用方式
--------
    python -m analysis.p12_itransformer_baseline [--epochs 40] [--seeds 0 1 2]
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.iTransformer import Model
from util.registry import build


class Cfg:
    def __init__(self, enc_in):
        self.task_name, self.seq_len, self.pred_len, self.enc_in, self.num_class = "classification", 13, 0, enc_in, 1
        self.d_model, self.d_ff, self.n_heads, self.dropout, self.activation, self.e_layers = 32, 128, 2, 0.3, "gelu", 3


def run(mode, seed, epochs, device):
    tr = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "train", "mode": mode}})
    te = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "test", "mode": mode}})
    xtr, ytr = torch.as_tensor(tr.x).to(device), torch.as_tensor(tr.y).float().to(device)
    xte = torch.as_tensor(te.x).to(device)
    torch.manual_seed(seed)
    m = Model(Cfg(tr.x.shape[-1])).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    pw = torch.tensor(float((1 - tr.y.mean()) / tr.y.mean()), dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        for idx in torch.randperm(len(xtr), generator=g).split(16):
            idx = idx.to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xtr[idx], None, None, None)[:, 0], ytr[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(xte[i:i + 500], None, None, None)[:, 0] for i in range(0, len(xte), 500)]).sigmoid().cpu().numpy()
    return roc_auc_score(te.y, p), average_precision_score(te.y, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    for mode in ("values", "values_mask"):
        t0 = time.time()
        r = np.array([run(mode, s, a.epochs, a.device) for s in a.seeds])
        print(f"{mode:12s} AUROC {r[:, 0].mean():.4f}±{r[:, 0].std():.4f}  AUPRC {r[:, 1].mean():.4f}±{r[:, 1].std():.4f}  ({len(a.seeds)} seeds, {a.epochs} ep, {time.time() - t0:.0f}s)", flush=True)
