"""
檔案功能
--------
取代被刪掉的 `p12_grad_reconn.py`（純梯度輸入版）。使用者要求：重新建立一個「weight + gradient 一起當輸入」
的 ReCoNN，判斷梯度本身到底有沒有可以利用的資訊。

問題：之前那版的 ImprintBlock 權重（`linear0.weight`）對所有病人都是同一個固定矩陣（伺服器自己設計、不隨
病人變化），把它一起餵進網路等於餵一個常數特徵，對迴歸沒有任何幫助——沒辦法真正測出「weight 這個資訊有沒有
用」。這一版讓 weight 真的會變：伺服器（攻擊者）每個病人隨機指派一個 `linfunc="fourier"` 的探測方向
（`mode` ∈ MODES，隨機、跟病人的真實資料 x 無關——攻擊者看不到 x 才要去猜），所以 weight 本身仍然不帶任何
跟 x 相關的資訊（這是刻意設計、也是下面「weight-only」那組的用途：驗證它的 ratio 應該剛好是 1.0，當 sanity
check），但「知道這次用的是哪個 weight」可能幫助網路更準確地解讀對應的梯度（不同探測方向的訊號/雜訊比不同）。

三個 ReCoNN 版本，同一批資料、同一個 sigma 掃描，一次比較：
    weight-only   : 輸入只有 weight（flatten），驗證沒有 x 的資訊時 rank_score 應該 ≈ 0.5（sanity check）
    gradient-only : 輸入只有 (dW,db)，weight 隨機混在一起（比上一版更難，因為不同病人用的探測方向不同）
    weight+gradient: 兩者都給——如果這組明顯比 gradient-only 好，代表「知道用的是哪個 weight」這個 side
                     information 真的能幫網路更有效地利用梯度；如果跟 gradient-only 打平，代表梯度本身的
                     可用資訊跟探測方向無關，加 weight 沒有意義。

指標（跟使用者討論後換掉 population-mean ratio）：改用 NN-oracle 式的池子 + 百分位排名（`rank_metrics`，跟
`analysis/p12_reconn.py` 的 rank_auc 同一套算法）——population mean 當基準太弱，NN-oracle（不訓練任何東西，
直接在攻擊者手上的輔助資料池裡找最像的）才是誠實的門檻。rank_score=1 代表完美還原，0.5 代表跟「亂猜」沒
差別（沒有可用資訊），<0.5 代表比亂猜還差。额外還有一個 shuffle ablation（把 gradient 那一截特徵在病人間
打散，重算 rank_score）跟輸入敏感度（saliency），用來直接檢查「梯度裡真的有沒有流入跟病人對應的資訊」，
不是只看最終分數。

使用方式
--------
    python -m analysis.p12_wg_reconn --sigma 0 --sigmas 0 1e-4 3e-4 1e-3 3e-3 1e-2 --modes 1 2 3 4 5
"""
import argparse
import json
import os

import numpy as np
import torch

from attacks.imprint import ImprintBlock, compute_empirical_bins
from trainers.fl_client import dp_clipped_gradient
from util.gdp import gdp_epsilon
from util.registry import build

ITR_SPEC = {"module": "models.itransformer_cls", "type": "ITransformerClassifier", "kwargs": {}}
CE_SPEC = {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}
DATA_SHAPE = (13 * 41,)
D = DATA_SHAPE[0]


def build_victims(bins, gain, modes, bin_aux_x):
    """功能：每個 mode 各建一個 ImprintBlock+iTransformer（固定隨機初始化 seed=42，跟其他 P12 imprint 腳本一致），
    bins 用該 mode 自己的經驗分位點（不同 mode 下 h(x) 的分布不同，不能共用同一組 bins）。"""
    victims = {}
    for mode in modes:
        bins_list = compute_empirical_bins(bin_aux_x, DATA_SHAPE, bins, linfunc="fourier", mode=mode)
        torch.manual_seed(42)
        net = build(ITR_SPEC)
        block = ImprintBlock(DATA_SHAPE, bins, connection="linear", gain=gain, linfunc="fourier", mode=mode, empirical_bins=bins_list)
        model = torch.nn.Sequential(block, net)
        model.eval()
        victims[mode] = (model, block)
    return victims


def clean_features(victims, modes_assigned, loss_fn, clip, x_all):
    """
    功能
        每個病人用它被指派到的 mode 對應的 victim 算一次 DP-clipped 梯度，回傳
        (weight_flat, grad_flat)：weight_flat 是那個 mode 的 ImprintBlock.linear0.weight（flatten），
        grad_flat 是 (dW,db) flatten（跟之前 p12_grad_reconn.py 一樣）。
    """
    n = len(x_all)
    Wf = np.zeros((n, victims[modes_assigned[0]][1].linear0.weight.numel()), np.float32)
    Gf = np.zeros((n, victims[modes_assigned[0]][1].linear0.weight.numel() + victims[modes_assigned[0]][1].linear0.bias.numel()), np.float32)
    norms = np.zeros(n)
    for i in range(n):
        model, block = victims[modes_assigned[i]]
        xi, yi = x_all[i:i + 1], torch.zeros(1, dtype=torch.long)
        g, norm = dp_clipped_gradient(model, loss_fn, xi, yi, clip)
        Wf[i] = block.linear0.weight.detach().reshape(-1).numpy()
        Gf[i] = torch.cat([g[0].reshape(-1), g[1].reshape(-1)]).numpy()
        norms[i] = float(norm.item())
    return Wf, Gf, norms


def add_noise_np(feats, sigma, clip, seed):
    """功能：對梯度特徵逐筆加 N(0,(sigma*clip)^2)（weight 特徵不是梯度，不加噪——攻擊者自己選的，本來就知道）。"""
    if sigma <= 0:
        return feats
    rng = np.random.default_rng(seed)
    return (feats + rng.normal(0.0, sigma * clip, feats.shape)).astype(np.float32)


def rank_metrics(xhat: np.ndarray, x_true: np.ndarray, pool: np.ndarray):
    """
    功能
        NN-oracle 式的池子 + 百分位排名指標（跟 `analysis/p12_reconn.py` 的 rank_auc 算法一致，取代population-mean
        ratio）。對每個測試病人：拿攻擊者自己手上的輔助資料池（pool，等於「什麼都不訓練、直接在池子裡找最像的」
        這個 NN-oracle baseline 會用的同一個池子），算「真正的病人」在「重建結果 xhat 跟池子裡每一個成員的距離」
        這個排序裡排第幾百分位——xhat 離真正病人越近（比池子裡幾乎所有人都近），百分位越低，rank_score 越高。
    Pseudo code
        d_pool[i,j]  <- ||xhat_i - pool_j||^2                      # xhat 到池子裡每個人的距離
        d_true[i]    <- ||xhat_i - x_true_i||^2                    # xhat 到「真正答案」的距離
        pct[i]       <- 池子裡比真正答案更接近 xhat_i 的比例        # 0 = xhat 比池子裡任何人都更像真正答案；0.5 = 跟亂猜一樣
        rank_score   <- 1 - mean(pct)                               # 1 = 完美；0.5 = 沒有資訊；0 = 比亂猜還差
        null：把 xhat 跟「隔壁病人」的重建結果對調一次（roll by 1）重算一次 pct，驗證沒資訊時 rank_score_null≈0.5
    用法
        score, null = rank_metrics(xhat, x_test_true, pool)
    """
    d_pool = (xhat ** 2).sum(1, keepdims=True) + (pool ** 2).sum(1)[None, :] - 2 * xhat @ pool.T
    d_true = ((xhat - x_true) ** 2).sum(1)
    pct = ((d_pool < d_true[:, None]).sum(1) + 0.5 * (d_pool == d_true[:, None]).sum(1)) / len(pool)
    roll = np.roll(np.arange(len(x_true)), 1)
    d_null = ((xhat[roll] - x_true) ** 2).sum(1)
    pct_null = (d_pool[roll] < d_null[:, None]).sum(1) / len(pool)
    return float(1 - pct.mean()), float(1 - pct_null.mean())


def gradient_flow_diagnostics(phi, Xte, n_grad, zmu, x_test_true, device, shuffle_seed=0):
    """
    功能
        量「梯度資訊有沒有真的流進網路的輸出」，兩個指標：
        1. shuffle ablation：把 gradient 那一截特徵在病人之間打散（weight 那一截、標準化統計量都不變），
           重新跑一次 phi，比較打散前後的 median MSE。打散後退化越多，代表梯度裡「跟這個病人對應」的部分
           越重要（不是單純在用 weight/mode 猜一個跟 mode 相關的常數）。
        2. saliency：|d(sum_j x_hat_j)/d(input_feature_i)|，對 weight 那一截跟 gradient 那一截分別取平均，
           衡量網路對兩種輸入的「局部敏感度」（跟 1 不同：這個不看打散後對不對，只看網路有沒有在用這個方向的梯度）。
    Pseudo code
        shuffled_X <- Xte；shuffled_X[:, n_grad:] <- Xte[打散的病人順序, n_grad:]
        shuffle_mse <- median(((phi(shuffled_X)+zmu - x_true)^2).mean(1))
        saliency <- |d phi(Xte).sum() / d Xte|.mean(0)；weight_sal <- saliency[:n_grad].mean()；grad_sal <- saliency[n_grad:].mean()
    用法
        diag = gradient_flow_diagnostics(phi, Xte, n_grad, zmu, x_test_true, device)
    """
    with torch.no_grad():
        xhat_real = (phi(Xte).cpu().numpy() + zmu)

    if n_grad < Xte.shape[1]:                       # 有 gradient 那一截才做 shuffle ablation
        perm = torch.as_tensor(np.random.default_rng(shuffle_seed).permutation(Xte.shape[0]), device=device)
        Xshuf = Xte.clone()
        Xshuf[:, n_grad:] = Xte[perm][:, n_grad:]
        with torch.no_grad():
            xhat_shuf = (phi(Xshuf).cpu().numpy() + zmu)
    else:
        xhat_shuf = xhat_real                         # weight-only：沒有 gradient 那一截，打散等於沒做

    Xsal = Xte.clone().requires_grad_(True)
    phi(Xsal).sum().backward()
    sal = Xsal.grad.abs().mean(0).detach().cpu().numpy()
    weight_sal = float(sal[:n_grad].mean()) if n_grad > 0 else 0.0
    grad_sal = float(sal[n_grad:].mean()) if n_grad < len(sal) else 0.0
    return {"xhat_real": xhat_real, "xhat_shuffled_grad": xhat_shuf,
            "saliency_weight_part": weight_sal, "saliency_grad_part": grad_sal}


def train_reconn(feat_aux, x_aux, feat_test, sigma_feat_is_grad, sigma, clip, hidden, epochs, lr, bs, seed, device, val_frac=0.2,
                 x_test_true=None):
    """
    功能
        訓練一個 ReCoNN：輸入可以是 weight-only / gradient-only / weight+gradient 的任何一種特徵向量，流程完全
        相同（跟被刪掉的 p12_grad_reconn.py 一樣：每個 epoch 對「梯度那部分」的特徵重新抽一次雜訊，weight 那部分
        不加噪；AdamW + weight_decay + grad clip 穩定訓練；留出集挑最佳 epoch）。
    Pseudo code
        feat_aux = (weight_part, grad_part) 串接（weight_part 可能是空的，長度 0）
        每個 epoch：grad_part 重新加一次 N(0,(sigma*clip)^2)，weight_part 原樣
        訓練 MLP phi(標準化後的 feat) -> x - prior，MAE+MSE，留出集挑最佳
    """
    n_grad = sigma_feat_is_grad    # feat_aux 裡，從這個 column 開始到底是梯度特徵（會加噪），之前的是 weight（不加噪）
    n = len(feat_aux)
    nval = int(n * val_frac)
    idx = np.random.default_rng(seed).permutation(n)
    i_tr, i_val = idx[nval:], idx[:nval]
    mu_f, sd_f = feat_aux.mean(0), feat_aux.std(0) + 1e-8
    zmu = x_aux[i_tr].mean(0)

    phi = build({"module": "models.mlp", "type": "MLP", "kwargs": {"hidden": hidden, "activation": "relu"}},
                in_dim=feat_aux.shape[1], out_dim=D).to(device)
    opt = torch.optim.AdamW(phi.parameters(), lr=lr, weight_decay=1e-4)

    def noisy(feat, s):
        if n_grad >= feat.shape[1]:
            return feat
        out = feat.copy()
        out[:, n_grad:] = add_noise_np(feat[:, n_grad:], sigma, clip, seed=s)
        return out

    feat_val = noisy(feat_aux[i_val], 999)
    Xval = torch.as_tensor((feat_val - mu_f) / sd_f, device=device)
    Yval = torch.as_tensor(x_aux[i_val] - zmu, dtype=torch.float32, device=device)

    best, best_state = (1e9, 0), None
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        feat_tr = noisy(feat_aux[i_tr], 10_000 + ep)
        Xtr = torch.as_tensor((feat_tr - mu_f) / sd_f, device=device)
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
    Xte = torch.as_tensor((feat_te - mu_f) / sd_f, device=device)
    with torch.no_grad():
        xhat = phi(Xte).cpu().numpy() + zmu
    diag = gradient_flow_diagnostics(phi, Xte, n_grad, zmu, x_test_true, device) if x_test_true is not None else None
    return np.clip(xhat, -20.0, 20.0), best[1], diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aux", type=int, default=1500); ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-bin-aux", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=16); ap.add_argument("--gain", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--hidden", type=int, nargs="+", default=[1000, 1000]); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="p12_wg_reconn")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ds = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "all", "mode": "values", "flat": True}})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ds.x))
    i_bin_aux = perm[:a.n_bin_aux]
    i_aux = perm[a.n_bin_aux:a.n_bin_aux + a.n_aux]
    i_test = perm[a.n_bin_aux + a.n_aux:a.n_bin_aux + a.n_aux + a.n_test]
    bin_aux_x = torch.as_tensor(ds.x[i_bin_aux], dtype=torch.float32)
    x_aux = torch.as_tensor(ds.x[i_aux], dtype=torch.float32)
    x_test = torch.as_tensor(ds.x[i_test], dtype=torch.float32)

    victims = build_victims(a.bins, a.gain, a.modes, bin_aux_x)
    loss_fn = build(CE_SPEC)

    # [mode 指派] 跟病人的 x 無關（攻擊者看不到 x），只跟病人在資料集裡的位置、seed 有關——這是刻意要讓 weight
    # 不帶任何 x 的資訊，才能用 weight-only 那組當 sanity check
    mode_rng = np.random.default_rng(a.seed + 777)
    modes_aux = mode_rng.choice(a.modes, size=len(i_aux))
    modes_test = mode_rng.choice(a.modes, size=len(i_test))

    Wf_aux, Gf_aux, norms_aux = clean_features(victims, modes_aux, loss_fn, a.clip, x_aux)
    Wf_test, Gf_test, norms_test = clean_features(victims, modes_test, loss_fn, a.clip, x_test)
    frac_clipped = float(np.mean(np.concatenate([norms_aux, norms_test]) > a.clip))

    # [NN-oracle 池子] 跟 p12_reconn.py 的 rank_auc 用同一個慣例：攻擊者自己手上能拿到的輔助資料（bin 估計用的
    # + 訓練 ReCoNN 用的 aux），不含測試病人。NN-oracle baseline 本身（不訓練任何東西，直接在這個池子裡找最像的）
    # 的 rank_score 定義上就是 0.5（因為池子不含測試病人的真實鄰居資訊，等於沒有額外資訊時的基準）。
    pool = np.concatenate([bin_aux_x.numpy(), x_aux.numpy()])

    rows = []
    for sigma in a.sigmas:
        row = {"sigma": sigma, "epsilon": gdp_epsilon(1.0 / sigma, 1e-5) if sigma > 0 else float("inf")}
        for name, feat_aux, feat_test, n_grad_start in (
            ("weight_only", Wf_aux, Wf_test, Wf_aux.shape[1]),                                   # 全部是 weight，沒有梯度部分（n_grad_start = 結尾，noisy() 不加噪）
            ("gradient_only", Gf_aux, Gf_test, 0),                                               # 全部是梯度
            ("weight_plus_gradient", np.concatenate([Wf_aux, Gf_aux], 1), np.concatenate([Wf_test, Gf_test], 1), Wf_aux.shape[1]),
        ):
            _, best_ep, diag = train_reconn(feat_aux, x_aux.numpy(), feat_test, n_grad_start, sigma, a.clip, a.hidden, a.epochs, a.lr, a.bs, a.seed, a.device,
                                            x_test_true=x_test.numpy())
            rank_score, rank_null = rank_metrics(diag["xhat_real"], x_test.numpy(), pool)
            rank_shuf, _ = rank_metrics(diag["xhat_shuffled_grad"], x_test.numpy(), pool)
            row[f"{name}_rank_score"] = rank_score                 # 1=完美，0.5=沒有資訊，0=比亂猜還差
            row[f"{name}_rank_null"] = rank_null                   # sanity check：應該 ≈0.5
            row[f"{name}_rank_shuffled_grad"] = rank_shuf          # 把 gradient 那一截在病人間打散後的 rank_score（資訊流入的直接證據：越接近 0.5 代表被打散的那部分資訊越重要）
            row[f"{name}_best_epoch"] = best_ep
            row[f"{name}_saliency_weight"] = diag["saliency_weight_part"]
            row[f"{name}_saliency_grad"] = diag["saliency_grad_part"]
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
        rows.append(row)

    d = os.path.join("runs", "results", a.out)
    os.makedirs(d, exist_ok=True)
    json.dump({"args": vars(a), "frac_clipped": frac_clipped, "rows": rows}, open(os.path.join(d, f"result_seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
