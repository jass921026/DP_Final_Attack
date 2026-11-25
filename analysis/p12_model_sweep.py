"""
檔案功能
--------
使用者指示（2026-11-23，接續「之前都在空口說白話」的檢討）：重新嚴謹地跑「基本的」兩個攻擊——
imprint closed-form（`attacks.imprint.reconstruct`，單一 bin 相除）跟 gradient-only ReCoNN（MLP 學 (dW,db)
-> x_hat，跟被刪掉的 p12_grad_reconn.py / 現在 p12_wg_reconn.py 的 gradient_only 同一套邏輯），在 4 種 P12
victim 架構上都跑一遍：iTransformer / SUMMIT / MLP / GRU。流程：
    1. 先用 sigma=0（無 DP）確認兩個攻擊在這 4 個架構上都真的打得穿（closed-form 應該幾乎完美，
       gradient-only ReCoNN 應該明顯贏過瞎猜）。
    2. 確認後再 sweep 多個 DP sigma，觀察雜訊如何讓兩種攻擊退化（closed-form 的斷崖 vs ReCoNN 的漸進/斷崖）。
ImprintBlock 架構無關（插在最前面，下游接哪個分類器只是換 `net`），但 SUMMIT 需要 mode="values_mask"
（多一倍的遮罩通道，D=13*82=1066）才能建它的 key_padding_mask，其餘三個用 mode="values"（D=13*41=533）
——因此同一個 sigma、同一個攻擊，SUMMIT 跟其他三個模型用的不是同一個 D，比較時要注意。
指標沿用 `analysis/p12_wg_reconn.py` 的 `rank_metrics`（NN-oracle 池子 + 百分位排名，取代 population-mean
ratio）：closed-form 也改用同一套指標（它原本的 hit_rate/mse 不能跟 ReCoNN 的 rank_score 放在一起比）。

使用方式
--------
    python -m analysis.p12_model_sweep --model itransformer --sigmas 0
    python -m analysis.p12_model_sweep --model summit --sigmas 0 1e-4 3e-4 1e-3 3e-3 1e-2
    python -m analysis.p12_model_sweep --model mlp --sigmas 0 1e-4 ...
    python -m analysis.p12_model_sweep --model gru --sigmas 0 1e-4 ...
"""
import argparse
import json
import os

import numpy as np
import torch

from attacks.imprint import ImprintBlock, add_gradient_noise, compute_empirical_bins, reconstruct
from trainers.fl_client import dp_clipped_gradient
from util.gdp import gdp_epsilon
from util.registry import build

CE_SPEC = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}
SEQ_LEN = 41  # 這裡指 enc_in（變數數），seq_len(時間格)其實是 13；兩個名字容易混，底下都用 ENC_IN/N_BINS_T 分開命名
ENC_IN = 41

MODEL_SPECS = {
    "itransformer": {"spec": {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}}, "mode": "values"},
    "summit": {"spec": {"module": "models.summit_cls", "type": "SUMMITClassifier", "kwargs": {}}, "mode": "values_mask"},
    "mlp": {"spec": {"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": [256, 128], "activation": "relu"}}, "mode": "values"},
    "gru": {"spec": {"module": "models.gru_cls", "type": "GRUClassifier", "kwargs": {}}, "mode": "values"},
}


def get_data_shape(mode: str) -> tuple:
    return (13 * ENC_IN * (2 if mode == "values_mask" else 1),)


def build_victim(model_name, bins, gain, linfunc, bin_aux_x, data_shape):
    """
    功能
        建 ImprintBlock + 下游分類器。block 跟 net 之間插一個 nn.LayerNorm：ImprintBlock 的輸出量級由
        gain 決定、原本調校是給 iTransformer（自己內部每層都有 LayerNorm，對輸入量級不敏感）用的；直接接
        MLP/GRU（沒有任何內部正規化）時，block 的巨大輸出會讓 logits 爛到幾百~幾千，softmax 完全飽和，
        交叉熵梯度在 block 這一層變成全 0（實測：gW/gb 絕對值和都是 0.0），imprint 攻擊因此永遠 0 hits、
        退化成 prior——這不是「MLP/GRU 比較抗打」，是純粹的數值飽和假象。加 LayerNorm 不影響 closed-form
        攻擊的正確性：dL/dW_i 跟 dL/db_i 共享同一個「這個 bin 對下游損失的鏈式法則係數」，不管下游多複雜
        （包含 LayerNorm 這種跨 bin 的操作），這個係數對同一個 i 永遠相同，所以 dW_i/db_i = x_i 這個比例
        式不受影響（實測：加 LayerNorm 後 MLP 的 closed-form 還原誤差 ~1e-15，跟沒有數值問題時的 iTransformer
        一樣接近完美）。四個模型都統一加這一層，確保比較的是攻擊本身、不是數值穩定性的差異。
    """
    spec = MODEL_SPECS[model_name]["spec"]
    bins_list = compute_empirical_bins(bin_aux_x, data_shape, bins, linfunc=linfunc)
    torch.manual_seed(42)
    overrides = {"in_dim": data_shape[0], "out_dim": 2} if model_name == "mlp" else {}
    net = build(spec, **overrides)
    block = ImprintBlock(data_shape, bins, connection="linear", gain=gain, linfunc=linfunc, empirical_bins=bins_list)
    model = torch.nn.Sequential(block, torch.nn.LayerNorm(data_shape[0]), net)
    model.eval()
    return model, block


def rank_metrics(xhat: np.ndarray, x_true: np.ndarray, pool: np.ndarray):
    """功能：同 analysis/p12_wg_reconn.py 的 rank_metrics（NN-oracle 池子 + 百分位排名），見該檔docstring。"""
    d_pool = (xhat ** 2).sum(1, keepdims=True) + (pool ** 2).sum(1)[None, :] - 2 * xhat @ pool.T
    d_true = ((xhat - x_true) ** 2).sum(1)
    pct = ((d_pool < d_true[:, None]).sum(1) + 0.5 * (d_pool == d_true[:, None]).sum(1)) / len(pool)
    roll = np.roll(np.arange(len(x_true)), 1)
    d_null = ((xhat[roll] - x_true) ** 2).sum(1)
    pct_null = (d_pool[roll] < d_null[:, None]).sum(1) / len(pool)
    return float(1 - pct.mean()), float(1 - pct_null.mean())


def add_noise_np(feats, sigma, clip, seed):
    if sigma <= 0:
        return feats
    rng = np.random.default_rng(seed)
    return (feats + rng.normal(0.0, sigma * clip, feats.shape)).astype(np.float32)


def train_reconn(feat_aux, x_aux, feat_test, sigma, clip, hidden, epochs, lr, bs, seed, device, D, val_frac=0.2):
    """功能：gradient-only ReCoNN（跟 p12_wg_reconn.py 的 train_reconn 同一套流程，去掉 weight 相關分支與診斷，
    這裡的輸入永遠全部是梯度特徵，每個 epoch 重新加一次雜訊）。"""
    n = len(feat_aux)
    nval = int(n * val_frac)
    idx = np.random.default_rng(seed).permutation(n)
    i_tr, i_val = idx[nval:], idx[:nval]
    zmu = x_aux[i_tr].mean(0)

    def noisy(feat, s):
        return add_noise_np(feat, sigma, clip, seed=s)

    feat_val = noisy(feat_aux[i_val], 999)
    mu_f, sd_f = feat_aux.mean(0), feat_aux.std(0) + 1e-8
    Xval = torch.as_tensor((feat_val - mu_f) / sd_f, dtype=torch.float32, device=device)
    Yval = torch.as_tensor(x_aux[i_val] - zmu, dtype=torch.float32, device=device)

    phi = build({"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": hidden, "activation": "relu"}},
                in_dim=feat_aux.shape[1], out_dim=D).to(device)
    opt = torch.optim.AdamW(phi.parameters(), lr=lr, weight_decay=1e-4)

    best, best_state = (1e9, 0), None
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        feat_tr = noisy(feat_aux[i_tr], 10_000 + ep)
        Xtr = torch.as_tensor((feat_tr - mu_f) / sd_f, dtype=torch.float32, device=device)
        Ytr = torch.as_tensor(x_aux[i_tr] - zmu, dtype=torch.float32, device=device)
        phi.train()
        perm = torch.randperm(len(i_tr), generator=g)
        for s in range(0, len(i_tr), bs):
            b = perm[s:s + bs]
            e = phi(Xtr[b]) - Ytr[b]
            loss = e.abs().mean() + (e ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(phi.parameters(), 5.0)
            opt.step()
        phi.eval()
        with torch.no_grad():
            va = float(((phi(Xval) - Yval) ** 2).mean())
        if va < best[0]:
            best, best_state = (va, ep), {k: v.clone() for k, v in phi.state_dict().items()}
    phi.load_state_dict(best_state); phi.eval()

    feat_te = noisy(feat_test, 1000)
    Xte = torch.as_tensor((feat_te - mu_f) / sd_f, dtype=torch.float32, device=device)
    with torch.no_grad():
        xhat = phi(Xte).cpu().numpy() + zmu
    return np.clip(xhat, -20.0, 20.0), best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--n-aux", type=int, default=1500); ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-bin-aux", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=16); ap.add_argument("--linfunc", default="avg")
    ap.add_argument("--gain", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--thr-sd", type=float, default=3.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--hidden", type=int, nargs="+", default=[1000, 1000]); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out_tag = a.out or f"p12_model_sweep_{a.model}"

    mode = MODEL_SPECS[a.model]["mode"]
    data_shape = get_data_shape(mode)
    D = data_shape[0]

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": mode, "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ds.x))
    i_bin_aux = perm[:a.n_bin_aux]
    i_aux = perm[a.n_bin_aux:a.n_bin_aux + a.n_aux]
    i_test = perm[a.n_bin_aux + a.n_aux:a.n_bin_aux + a.n_aux + a.n_test]
    bin_aux_x = torch.as_tensor(ds.x[i_bin_aux], dtype=torch.float32)
    x_aux = torch.as_tensor(ds.x[i_aux], dtype=torch.float32)
    x_test = torch.as_tensor(ds.x[i_test], dtype=torch.float32)
    pool = np.concatenate([bin_aux_x.numpy(), x_aux.numpy()])

    model, block = build_victim(a.model, a.bins, a.gain, a.linfunc, bin_aux_x, data_shape)
    loss_fn = build(CE_SPEC)

    clean, g_norms = [], []
    for idx in np.concatenate([i_aux, i_test]):
        x = torch.as_tensor(ds.x[idx:idx + 1], dtype=torch.float32)
        y = torch.zeros(1, dtype=torch.long)
        g, norm = dp_clipped_gradient(model, loss_fn, x, y, a.clip)
        clean.append(g); g_norms.append(float(norm.item()))
    frac_clipped = float(np.mean(np.array(g_norms) > a.clip))
    n_aux = len(i_aux)
    clean_aux, clean_test = clean[:n_aux], clean[n_aux:]

    def flatten(gs, noise_seed_base=None, sigma=0.0):
        feats = np.zeros((len(gs), block.linear0.weight.numel() + block.linear0.bias.numel()), np.float32)
        for i, g in enumerate(gs):
            gg = add_gradient_noise(g, sigma * a.clip, seed=(noise_seed_base + i), distribution="gaussian") if sigma > 0 and noise_seed_base is not None else g
            feats[i] = torch.cat([gg[0].reshape(-1), gg[1].reshape(-1)]).numpy()
        return feats

    rows = []
    for sigma in a.sigmas:
        eps = gdp_epsilon(1.0 / sigma, 1e-5) if sigma > 0 else float("inf")
        row = {"model": a.model, "sigma": sigma, "epsilon": eps, "D": D}

        # [imprint closed-form]：跟 p12_imprint_shrinkage.py 一樣的單一 bin 相除 + 門檻 + fallback 成 prior
        prior = bin_aux_x.mean(0)
        cf_xhat = np.zeros((len(i_test), D), np.float32)
        for i in range(len(i_test)):
            g = add_gradient_noise(clean_test[i], sigma * a.clip, seed=2000 + i, distribution="gaussian") if sigma > 0 else clean_test[i]
            thr = a.thr_sd * 2 * sigma * a.clip if sigma > 0 else 0.0
            rec, n_hits = reconstruct(g, 1, data_shape, bias_threshold=thr, reduction=None, structure=block.structure, clip_range=(-20.0, 20.0))
            x_i = x_test[i:i + 1]
            cf_xhat[i] = (rec[((rec - x_i) ** 2).mean(1).argmin()] if n_hits > 0 else prior).numpy()
        cf_rank, cf_null = rank_metrics(cf_xhat, x_test.numpy(), pool)
        row["closed_form_rank_score"], row["closed_form_rank_null"] = cf_rank, cf_null

        # [gradient-only ReCoNN]
        Gf_aux = flatten(clean_aux)
        Gf_test = flatten(clean_test)
        xhat_recon, best_ep = train_reconn(Gf_aux, x_aux.numpy(), Gf_test, sigma, a.clip, a.hidden, a.epochs, a.lr, a.bs, a.seed, a.device, D)
        rc_rank, rc_null = rank_metrics(xhat_recon, x_test.numpy(), pool)
        row["reconn_rank_score"], row["reconn_rank_null"], row["reconn_best_epoch"] = rc_rank, rc_null, best_ep

        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
        rows.append(row)

    d = os.path.join("runs", "results", out_tag)
    os.makedirs(d, exist_ok=True)
    json.dump({"args": vars(a), "frac_clipped": frac_clipped, "rows": rows}, open(os.path.join(d, f"result_seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
