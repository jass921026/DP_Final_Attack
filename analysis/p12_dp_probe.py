"""
檔案功能
--------
P12 + iTransformer 的 DP-GD 前置檢查：初始參數下的逐樣本梯度範數（挑選 clip C），以及小批模型在不同 (C, lr, sigma) 下的效用與每步時間。
使用方式
--------
    python -m analysis.p12_dp_probe
"""
import time
import numpy as np
import torch
from torch.func import functional_call, grad, vmap
from sklearn.metrics import roc_auc_score
import torch.nn.utils as U

from trainers.batched_gd import train_models_batched
from util.registry import build

SPEC = {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}}
CE = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
perm = np.random.default_rng(0).permutation(len(ds.x))
fx, fy = ds.x[perm[:500]], ds.y[perm[:500]]
zx, zy = ds.x[perm[500:504]], ds.y[perm[500:504]]
ev = perm[-1000:]
torch.manual_seed(42); base = build(SPEC)
p = {k: v.detach() for k, v in base.named_parameters()}
ce = build(CE)
g = vmap(grad(lambda pp, x, y: ce.per_sample(functional_call(base, pp, (x[None],)), y[None]).sum()), in_dims=(None, 0, 0))(
    p, torch.as_tensor(fx), torch.as_tensor(fy))
nr = torch.sqrt(sum((v.reshape(len(fx), -1) ** 2).sum(1) for v in g.values())).numpy()
print("init per-sample grad norm: median %.3f p10 %.3f p90 %.3f" % (np.median(nr), *np.percentile(nr, [10, 90])), flush=True)


def util(P):
    m = build(SPEC); a = []
    for q in P:
        U.vector_to_parameters(torch.as_tensor(q), m.parameters()); m.eval()
        with torch.no_grad():
            a.append(roc_auc_score(ds.y[ev], torch.softmax(m(torch.as_tensor(ds.x[ev])), 1)[:, 1].numpy()))
    return float(np.mean(a))


for clip, lr, sigma in [(None, 0.03, 0), (1.0, 0.03, 0), (1.0, 0.3, 0), (1.0, 1.0, 0)]:
    t = time.time()
    dp = None if clip is None else {"clip": clip, "sigma": sigma, "seed": 0}
    P = train_models_batched(SPEC, 42, fx, fy, zx, zy, lr, 0.9, 100, build(CE), chunk_size=4, device=dev, dp=dp)
    print(f"clip {clip} lr {lr} sigma {sigma}: AUROC {util(P):.4f}  {time.time() - t:.0f}s for 4 models x100 steps", flush=True)
