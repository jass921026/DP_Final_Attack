"""
檔案功能
--------
Robbing the Fed（Fowl, Geiping, Czaja, Goldblum, Goldstein; ICLR 2022）的 imprint 攻擊，FedSGD 版本。
惡意伺服器在模型最前面插入一個 imprint 區塊（k 個「bin」，每個 bin 是 ReLU(h(x) - c_i)），
客戶端照常回傳梯度後，伺服器只靠梯度相鄰列的差分就能直接解出落在單一 bin 的輸入影像。
本檔包含：
    ImprintBlock               惡意區塊（cumulative 結構，connection = addition | linear）
    SparseImprintBlock         sparse 結構（FedAvg 用，不需差分）
    add_gradient_noise / iip_score  本地差分隱私雜訊與 IIP 度量
    invert_cumulative          由梯度差分解出影像（論文 Eq. 4；對應官方 ImprintAttacker + invert_fc_layer）
    select_hits                命中數多於批次時的篩選（官方 reduce_hits）
    reconstruct                從客戶端梯度到重建影像的完整伺服器端流程
    best_match_psnr            評估：每張真實影像取最接近的重建影像
    expected_perfect_fraction  理論：完美還原比例（Proposition 1 的主項）
    run_imprint_fedsgd         設定檔驅動的實驗：掃 bins 數，輸出指標
本檔不依賴 breaching；tests/test_imprint.py 以官方程式碼為基準驗證一致性。

使用方式
--------
    python run_imprint.py --config configs/imprint_fedsgd_bins.json [--smoke]
    from attacks.imprint import ImprintBlock, reconstruct
"""
import json
import math
import os
import time
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import laplace

from attacks.base import BaseAttack
from util.registry import build
from models.base import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImprintBlock(BaseModel):
    """
    功能
        放在模型最前面的惡意區塊。所有列 W_i 相同（都是線性查詢 h），偏置 b_i = -c_i，c_i 是 h(x) 分布的 i/k 分位數
        （等質量 bins）。ReLU 後 bin i 對 h(x) > c_i 的影像啟動（cumulative 結構）。
    Pseudo code (論文 Sec. 3, "Imprint module")
        h <- 線性函數（預設 DCT-2 第 mode 個係數 "fourier"，分布近似 Laplace(0, 1/sqrt(2))）
        c_i <- F^{-1}(i/k),  i = 1..k-1;  c_0 = -10 (近似 -inf)
        W*_i <- gain * h ;  b*_i <- -gain * c_i                       # k 列相同
        forward(x):  a <- ReLU(W* x + b*)
                     connection=addition: out <- x + mean_i(a_i)       # 不改變輸入的形狀，後續網路照常運作
                     connection=linear:   out <- W2 a + b2,  W2 = 1/gain, b2 = -mean(c)
    用法
        block = ImprintBlock((3,224,224), num_bins=128, connection="addition", gain=1e-3, linfunc="fourier", mode=32)
        net = torch.nn.Sequential(block, resnet)        # 參數順序：block.linear0.weight 是 index 0，bias 是 index 1
    """
    structure = "cumulative"

    def __init__(self, data_shape: Tuple[int, ...], num_bins: int, connection: str = "linear",
                 gain: float = 1e-3, linfunc: str = "fourier", mode: int = 0, bin_dist: Optional[str] = None,
                 empirical_bins: Optional[List[float]] = None):
        super().__init__()
        self.bin_dist = bin_dist        # None：官方行為（fourier -> Laplace，其他 -> 常態）；"laplace"/"normal"：明確指定（Table 1 的 Mean+Laplacian）
        self.data_shape = tuple(data_shape)
        self.data_size = int(np.prod(data_shape))
        self.num_bins = num_bins
        self.connection = connection
        self.linear0 = torch.nn.Linear(self.data_size, num_bins)
        # [對應 Pseudo code: c_i 與 W*, b*] 分位點、線性查詢與偏置；empirical_bins 給定時跳過常態/Laplace 假設，
        # 改用輔助資料上 h(x) 的經驗分位點（P12 的特徵分布既非常態也非 Laplace，見 compute_empirical_bins）
        self.bins = list(empirical_bins) if empirical_bins is not None else self._get_bins(linfunc)
        with torch.no_grad():
            self.linear0.weight.data = self._init_linear_function(linfunc, mode) * gain
            self.linear0.bias.data = self._make_biases() * gain
        # [對應 Pseudo code: connection=linear]
        if connection == "linear":
            self.linear2 = torch.nn.Linear(num_bins, self.data_size)
            with torch.no_grad():
                self.linear2.weight.data = torch.ones_like(self.linear2.weight.data) / gain
                self.linear2.bias.data -= float(np.mean(self.bins))
        self.nonlin = torch.nn.ReLU()

    @torch.no_grad()
    def _init_linear_function(self, linfunc: str, mode: int) -> torch.Tensor:
        """功能：k 列相同的線性查詢 h。fourier：DCT-2 第 mode 個係數（含官方的 *4 與 max(mode,0.33) 正規化）。"""
        K, N = self.num_bins, self.data_size
        if linfunc == "avg":
            return torch.ones(K, N) / N
        if linfunc == "fourier":
            return torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
        if linfunc == "randn":      # 官方：高斯隨機向量，強制平均 0、標準差 1/sqrt(N)（Table 1 的 Random + Normal）
            w = torch.randn(N).repeat(K, 1)
            std, mu = torch.std_mean(w[0])
            return (w - mu) / std / math.sqrt(N)
        raise ValueError(f"unsupported linfunc {linfunc!r}")

    def _make_biases(self) -> torch.Tensor:
        """功能：偏置 b_i = -c_i（Pseudo code 的 b*；尚未乘 gain）。子類別可覆寫。"""
        return -torch.as_tensor(self.bins, dtype=torch.float32)

    def _get_bins(self, linfunc: str) -> List[float]:
        """功能：等質量分位點 c_i。fourier -> Laplace(0, 1/sqrt 2)；其他 -> 標準常態（論文 Eq. 3 的 b_i = -Phi^{-1}(i/k)）。"""
        bins = [-10.0]
        for i in range(1, self.num_bins):
            use_laplace = (self.bin_dist == "laplace") if self.bin_dist else ("fourier" in linfunc)
            if use_laplace:
                bins.append(float(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i / self.num_bins)))
            else:
                bins.append(NormalDist().inv_cdf(i / self.num_bins))
        return bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能：前向傳播（見 Pseudo code 的 forward）。用法：block(x)，x 形狀 (B,*data_shape)。"""
        x_in = x
        a = self.nonlin(self.linear0(x.flatten(start_dim=1)))
        if self.connection == "linear":
            out = self.linear2(a)
        else:   # addition
            out = x_in.flatten(start_dim=1) + a.mean(dim=1, keepdim=True)
        return out.unflatten(dim=1, sizes=self.data_shape)


def compute_empirical_bins(aux_x: torch.Tensor, data_shape: Tuple[int, ...], num_bins: int,
                            linfunc: str = "avg", mode: int = 0) -> List[float]:
    """
    功能
        用輔助資料（攻擊者手上、不含目標的樣本）估計線性查詢 h(x) 的經驗等質量分位點，取代 ImprintBlock 預設的
        常態/Laplace 假設。P12 的標準化特徵約 2/3 是補 0（遮罩），h(x) 的分布在 0 附近有尖峰，不服從那兩個假設；
        ImageNet 的假設成立是因為標準化像素經 DCT（fourier）後近似 Laplace，一般線性查詢近似常態。
    Pseudo code
        h <- 與 ImprintBlock 相同的線性查詢（avg/fourier/randn，gain 無關，分位點對 gain 縮放不變）
        c_i <- empirical quantile(h(aux_x), i/num_bins),  i = 1..num_bins-1;  c_0 = -1e4（近似 -inf）
    用法
        bins = compute_empirical_bins(aux_x, (13, 41), num_bins=16, linfunc="avg")
        block = ImprintBlock((13, 41), 16, linfunc="avg", empirical_bins=bins)
    """
    tmp = ImprintBlock(data_shape, num_bins, linfunc=linfunc, mode=mode, gain=1.0)   # 只借用 h 的權重；gain 不影響分位點的相對位置
    w = tmp.linear0.weight.data[0]
    h = aux_x.reshape(aux_x.shape[0], -1).float() @ w
    return [-1e4] + [float(torch.quantile(h, i / num_bins)) for i in range(1, num_bins)]


class SparseImprintBlock(ImprintBlock):
    """
    功能
        sparse 結構的 imprint 區塊（論文 Appendix「sparse variant」，用於 FedAvg）：每個 bin 只對 c_i < h(x) < c_{i+1} 的影像
        有非零輸出（以 Hardtanh(0, gain) 構成，等同兩個 ReLU 相減），所以梯度的每一列直接對應單一 bin，不需要相鄰列差分。
    Pseudo code (官方 breaching SparseImprintBlock)
        m_j <- F^{-1}(j/(k+2)), j = 1..k+1;   bin_sizes_j <- m_{j+1} - m_j;   c_j <- m_{j+1}   (官方丟掉第一個分位點)
        W*_j <- gain * h / bin_size_j ;  b*_j <- -gain * c_j / bin_size_j
        forward: a <- Hardtanh(0, gain)(W* x + b*);  其餘連接同 ImprintBlock
    用法
        block = SparseImprintBlock((3,224,224), 128, "linear", gain=1.0, linfunc="fourier", mode=32)
    """
    structure = "sparse"

    def __init__(self, data_shape, num_bins, connection="linear", gain=1.0, linfunc="fourier", mode=0, bin_dist=None):
        super().__init__(data_shape, num_bins, connection, gain, linfunc, mode, bin_dist)
        # [對應 Pseudo code: forward 的 Hardtanh]
        self.nonlin = torch.nn.Hardtanh(min_val=0, max_val=gain)

    def _get_bins(self, linfunc: str) -> List[float]:
        """功能：k+1 個等質量分位點，記錄 bin 寬度，回傳去掉第一個的分位點（對應 Pseudo code 前兩行）。"""
        bins, mass = [], 0.0
        for _ in range(self.num_bins + 1):
            mass += 1 / (self.num_bins + 2)
            if (self.bin_dist == "laplace") if self.bin_dist else ("fourier" in linfunc):
                bins.append(float(laplace(loc=0, scale=1 / math.sqrt(2)).ppf(mass)))
            else:
                bins.append(NormalDist(mu=0, sigma=1).inv_cdf(mass))
        self.bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        return bins[1:]

    @torch.no_grad()
    def _init_linear_function(self, linfunc: str, mode: int) -> torch.Tensor:
        """功能：每一列除以該 bin 的寬度（Pseudo code 的 W*_j）。"""
        w = super()._init_linear_function(linfunc, mode)
        for i in range(w.shape[0]):
            w[i] = w[i] / self.bin_sizes[i]
        return w

    def _make_biases(self) -> torch.Tensor:
        """功能：b_j = -c_j / bin_size_j（Pseudo code 的 b*_j）。"""
        return torch.tensor([-b / w for b, w in zip(self.bins, self.bin_sizes)], dtype=torch.float32)


def invert_cumulative(weight_grad: torch.Tensor, bias_grad: torch.Tensor,
                      bias_threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    功能
        由 imprint 層的梯度還原落在單一 bin 的輸入。論文 Eq. 4 與官方 ImprintAttacker + invert_fc_layer。
    Pseudo code (論文 Eq. 4)
        對 cumulative 結構：  dW_i <- grad W_i - grad W_{i+1};  db_i <- grad b_i - grad b_{i+1}
                              （列 i 累加了 h(x) > c_i 的所有影像，相鄰列相減 = 落在 [c_i, c_{i+1}] 的影像）
        有效 bin：            db_i != 0                       （官方；bias_threshold > 0 時改用 |db_i| > bias_threshold）
        還原：                x_hat_i <- dW_i / db_i          （bin 內只有一張影像時 = 該影像本身）
    用法
        rec, wd, bd, valid = invert_cumulative(grads[0], grads[1])    # rec: (n_valid, D)
    說明
        論文 Eq. 4 寫 W_l - W_{l+1}（bin 編號隨 h 遞增時，列 l 包含較多影像，所以前一列減後一列）；官方迴圈與本函式算的是
        g_i - g_{i-1}，即論文式的相反號。權重差與偏置差同時反號，所以比值 dW/db（還原結果）完全相同；
        tests/test_imprint.py 以官方原始運算驗證兩者逐元素一致。
    """
    # [對應 Pseudo code: dW, db] 相鄰列差分（第 0 列保留）
    wd, bd = weight_grad.clone(), bias_grad.clone()
    wd[1:] = weight_grad[1:] - weight_grad[:-1]
    bd[1:] = bias_grad[1:] - bias_grad[:-1]
    # [對應 Pseudo code: 有效 bin]
    valid = (bd.abs() > bias_threshold) if bias_threshold > 0 else (bd != 0)
    # [對應 Pseudo code: x_hat = dW / db]
    rec = wd[valid] / bd[valid, None]
    return rec, wd, bd, valid


def shrinkage_reconstruct(weight_grad: torch.Tensor, bias_grad: torch.Tensor, noise_std: float, prior: torch.Tensor,
                          prior_var: float, structure: str = "cumulative") -> torch.Tensor:
    """
    功能
        closed-form `invert_cumulative` 的貝氏收縮版：不挑「一個命中的 bin」做除法、其餘 bin 整個丟掉，而是把全部
        k 個 bin 的 (dW_i, db_i) 一起當作對同一個 x 的 k 次帶權線性量測，做加權最小平方 + 收縮到已知先驗（population
        mean）的聯合估計。雜訊蓋過某個 bin 的訊號時，那個 bin 的權重（db_i^2）本身就很小，會自動被壓低，不需要
        像 `invert_cumulative` 的 `bias_threshold` 那樣用硬門檻整個丟棄——這是跟使用者討論「post-processing 換 MSE」
        時提的 Wiener filter / ridge 估計，解析解、不用訓練。
    Pseudo code
        目標：對同一個 x，每個 bin i 給出一次觀測 dW_i = x * db_i + N(0, noise_std^2)（cumulative 結構下，batch=1 時
              相鄰列差分後每個有效 bin 都是對同一張輸入的獨立量測；batch>1 時個別 bin 混了不同樣本，這裡仍把它們
              當成對「這批資料的某個聚合量」的量測，近似處理，不做跨樣本拆分）。
        先驗：x ~ N(prior, prior_var)（prior、prior_var 來自輔助資料，population mean / variance）
        最小化：sum_i (dW_i - db_i*x)^2 / noise_std^2 + (x - prior)^2 / prior_var
        解（對 x 微分 = 0）：lambda <- noise_std^2 / prior_var
                            x_hat <- (sum_i db_i*dW_i + lambda*prior) / (sum_i db_i^2 + lambda)
    用法
        x_hat = shrinkage_reconstruct(grads[0], grads[1], sigma*clip, prior, prior_var)   # x_hat: (D,)
    """
    if structure == "sparse":
        wd, bd = weight_grad, bias_grad
    else:
        wd, bd = weight_grad.clone(), bias_grad.clone()
        wd[1:] = weight_grad[1:] - weight_grad[:-1]
        bd[1:] = bias_grad[1:] - bias_grad[:-1]
    lam = (noise_std ** 2) / prior_var if noise_std > 0 else 0.0
    num = (bd[:, None] * wd).sum(0) + lam * prior
    den = (bd ** 2).sum() + lam
    return num / den


def select_hits(rec: torch.Tensor, wd: torch.Tensor, valid: torch.Tensor, n: int,
                reduction: str = "weight", pad: bool = True) -> torch.Tensor:
    """
    功能
        命中數與批次大小不同時的處理（官方 ImprintAttacker.reduce_hits）。命中數 < n：補零（pad）；> n：留下 n 筆。
    Pseudo code
        if n_hits <= n:  補零到 n（pad=True）
        else:            reduction="weight":  留下 |mean_d dW_i| 最小的 n 個有效 bin
                         reduction="bias":    留下 |db_i| 最小的 n 個（本函式未提供，需傳 bd）
                         reduction=None:      全部保留
    用法
        rec = select_hits(rec, wd, valid, n=64, reduction="weight")
    """
    if n >= rec.shape[0]:
        if pad and n > rec.shape[0]:
            rec = torch.cat([rec, rec.new_zeros(n - rec.shape[0], *rec.shape[1:])])
        return rec
    if reduction == "weight":
        keep = torch.topk(wd.mean(dim=1)[valid].abs(), n, largest=False)[1]
        return rec[keep]
    return rec


def reconstruct(grads: List[torch.Tensor], n: int, data_shape: Tuple[int, ...], weight_idx: int = 0, bias_idx: int = 1,
                bias_threshold: float = 0.0, reduction: str = "weight", structure: str = "cumulative",
                clip_range: Optional[Tuple[float, float]] = None) -> Tuple[torch.Tensor, int]:
    """
    功能
        伺服器端完整流程：取出 imprint 層的權重/偏置梯度 -> 差分反演 -> 篩選 -> 重塑並夾到合法像素範圍。
        回傳 (重建影像 (n,C,H,W)，已正規化空間；有效命中數)。
    Pseudo code (論文 Sec. 3 + 官方 ImprintAttacker.reconstruct)
        gW, gb <- grads[weight_idx], grads[bias_idx]
        x_hat, dW, db, valid <- invert_cumulative(gW, gb)
        x_hat <- select_hits(x_hat, ...)
        x_hat <- reshape(x_hat, data_shape);  clamp 到 [(0-mean)/std, (1-mean)/std]
    用法
        rec, n_hits = reconstruct(grads, 64, (3,224,224))
    """
    if structure == "sparse":       # sparse：每列直接對應單一 bin，不做差分
        wd, bd = grads[weight_idx].clone(), grads[bias_idx].clone()
        valid = (bd.abs() > bias_threshold) if bias_threshold > 0 else (bd != 0)
        rec = wd[valid] / bd[valid, None]
    else:
        rec, wd, bd, valid = invert_cumulative(grads[weight_idx].clone(), grads[bias_idx].clone(), bias_threshold)
    n_hits = int(valid.sum())
    rec = select_hits(rec, wd, valid, n, reduction)
    rec = rec.reshape(rec.shape[0], *data_shape)
    if clip_range is not None:            # 一般（非像素）資料：夾到給定的 (下界, 上界)，而非 ImageNet 的正規化像素範圍
        lo, hi = clip_range
        return rec.clamp(lo, hi), n_hits
    mean = torch.tensor(IMAGENET_MEAN, dtype=rec.dtype)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=rec.dtype)[:, None, None]
    return torch.max(torch.min(rec, (1 - mean) / std), -mean / std), n_hits


def best_match_psnr(rec01: torch.Tensor, true01: torch.Tensor, perfect_mse: float = 1e-4) -> Tuple[Dict[str, float], torch.Tensor]:
    """
    功能
        評估：每張真實影像取與它「像素 MSE 最小」的重建影像，算 PSNR。論文 Appendix A.5 是用 LPIPS 特徵相似度加線性指派
        配對（已知差異：本函式不做一對一指派，也不用 LPIPS）。PSNR 上限 120（完全相同時），所以平均 PSNR 會被完美還原的
        影像拉高，只能當參考；frac_perfect（MSE < perfect_mse）較不受影響。
    Pseudo code
        for each true image t:  m_t <- min_j MSE(t, rec_j);  psnr_t <- -10 log10(max(m_t, 1e-12))
        frac_perfect <- mean[ m_t < perfect_mse ]
    用法
        stats, psnr = best_match_psnr(rec01, true01)
    """
    a, b = true01.flatten(1).double(), rec01.flatten(1).double()
    d = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T).clamp_min(0) / a.shape[1]   # 不建立 n*n*D 張量
    best = d.min(1).values
    psnr = -10 * torch.log10(best.clamp_min(1e-12))
    psnr160 = -10 * torch.log10(best.clamp_min(1e-16))     # 上限 160：論文報告過 PSNR 161.36（機器精度的複製），沒有 120 的上限
    return dict(mean_psnr=float(psnr.mean()), median_psnr=float(psnr.median()), frac_perfect=float((best < perfect_mse).float().mean()),
                frac_psnr_gt30=float((psnr > 30).float().mean()), mean_psnr_cap160=float(psnr160.mean()), mean_mse=float(best.mean())), psnr


def add_gradient_noise(grads: List[torch.Tensor], scale: float, seed: int = 0, distribution: str = "laplacian") -> List[torch.Tensor]:
    """
    功能
        客戶端的本地差分隱私防禦：對回傳梯度的每個元素加獨立雜訊（論文 Sec. 5 / Appendix Fig. 20；官方
        breaching 的 local_diff_privacy.gradient_noise，預設 Laplace(0, scale)）。
    Pseudo code
        for each gradient tensor g:  g <- g + Laplace(0, scale)   （distribution="gaussian" 時為 N(0, scale^2)）
    用法
        noisy = add_gradient_noise(grads, 1e-3, seed=0)
    """
    if scale <= 0:
        return grads
    gen = torch.Generator().manual_seed(seed)
    out = []
    for g in grads:
        u = torch.rand(g.shape, generator=gen, dtype=torch.float64).clamp(1e-12, 1 - 1e-12) - 0.5
        if distribution == "laplacian":     # 反函數法：Laplace(0, b) = -b * sign(u) * ln(1 - 2|u|)
            noise = -scale * torch.sign(u) * torch.log1p(-2 * u.abs())
        else:
            noise = torch.randn(g.shape, generator=gen, dtype=torch.float64) * scale
        out.append(g + noise.to(g.dtype))
    return out


def iip_score(rec01: torch.Tensor, true01: torch.Tensor, true_labels: torch.Tensor,
              mates_x: torch.Tensor, mates_y: torch.Tensor) -> float:
    """
    功能
        Image Identifiability Precision（Yin et al., 2021；論文 Appendix A.5 的像素空間版本）：重建影像若比
        「驗證集中同類別的任何其他影像」更接近它對應的真圖（歐氏距離），就算識別成功。對應的重建影像取像素 MSE 最小者。
    Pseudo code
        for 每張真圖 t_i（類別 c_i）:
            r_i <- argmin_j MSE(t_i, rec_j)
            ok_i <- ||r_i - t_i|| < min_{v in 驗證集, 類別 c_i, v != t_i} ||r_i - v||
        return mean(ok)
    用法
        iip = iip_score(rec01, true01, labels, mates_x, mates_y)
    """
    a, b = true01.flatten(1).double(), rec01.flatten(1).double()
    d = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T).clamp_min(0)
    best = d.argmin(1)
    ok = []
    m = mates_x.flatten(1).double()
    for i in range(len(a)):
        r = b[best[i]]
        d_true = ((r - a[i]) ** 2).sum()
        mm = m[mates_y == true_labels[i]]
        d_mates = ((mm - r) ** 2).sum(1).min() if len(mm) else torch.tensor(float("inf"))
        ok.append(bool(d_true < d_mates))
    return float(np.mean(ok))


def expected_perfect_fraction(n: int, k: int) -> float:
    """
    功能
        理論：n 張影像、k 個等質量 bin，單獨落在某個 bin 的影像比例的主項（論文 Proposition 1 的第一項）。
        論文的完整公式還含殘差項 r(n,k)（尾端質量與多數影像同 bin 的修正），本函式不含，所以只是近似。
    Pseudo code
        一張影像單獨在它的 bin 的機率 = (1 - 1/k)^(n-1)
    用法
        expected_perfect_fraction(64, 156)   # 論文 Fig.1a 說約 156 bins 時超過 50%
    """
    return (1.0 - 1.0 / k) ** (n - 1)


def run_imprint_fedsgd(cfg: Dict[str, Any], root: str = "runs") -> Dict[str, Any]:
    """
    功能
        設定檔驅動的 FedSGD 實驗：對 sweep 的每個 bins 數，建模型、客戶端算梯度、伺服器反演、算指標，寫出結果。
    Pseudo code
        x, y <- 資料集 (n 張)
        for k in cfg.sweep.num_bins:
            model <- Sequential(ImprintBlock(k), ResNet18)                 # 伺服器下發的惡意模型
            g <- fedsgd_gradient(model, normalize(x), y)                    # 誠實客戶端回傳的梯度
            x_hat <- reconstruct(g)                                         # 伺服器端反演
            指標 <- best_match_psnr(x_hat, x);  理論 <- expected_perfect_fraction(n, k)
        寫出 runs/results/<name>/result.json
    用法
        run_imprint_fedsgd(json.load(open("configs/imprint_fedsgd_bins.json")))
    """
    from trainers.fl_client import fedavg_update, fedsgd_gradient
    t0 = time.time()
    # [對應 Pseudo code: x, y]
    ds = build(cfg["data"])
    x01 = torch.as_tensor(ds.x)
    y = torch.as_tensor(ds.y)
    n = len(x01)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    x_norm = (x01 - mean) / std
    loss_fn = build(cfg.get("loss", {"module": "loss.cross_entropy", "type": "CrossEntropy", "kwargs": {}}))      # 客戶端的訓練損失
    blk = cfg["block"]
    dp_sigmas = cfg["sweep"].get("dp_sigma")                           # DP-FedSGD：高斯機制的噪音倍率 sigma（雜訊 = sigma * clip，除以 n）
    noises = cfg["sweep"].get("noise", [0.0])
    thr_sd = cfg.get("attack", {}).get("bias_threshold_sd", 0.0)       # 0 = 官方（bias_grad != 0）；>0 = 雜訊感知門檻（倍數 * 2*scale）
    mates = None
    if cfg.get("iip", False):
        mx, my = ds.class_mates()
        mates = (torch.as_tensor(mx), torch.as_tensor(my))
    rows = []
    for k in cfg["sweep"]["num_bins"]:
        # [對應 Pseudo code: model]
        net = build(cfg["model"])
        BlockCls = SparseImprintBlock if blk.get("type", "ImprintBlock") == "SparseImprintBlock" else ImprintBlock
        block = BlockCls(tuple(x01.shape[1:]), k, blk["connection"], blk["gain"], blk["linfunc"], blk["mode"], blk.get("bin_dist"))
        model = torch.nn.Sequential(block, net)
        model.eval()
        # [對應 Pseudo code: g] 誠實客戶端回傳的東西：FedSGD 的梯度，或 FedAvg 的參數差 (p_local - p_server)
        cl = cfg.get("client", {"type": "fedsgd"})
        if cl["type"] == "fedavg":
            if cl.get("dtype") == "float64":
                # 更新量 (~lr*grad) 遠小於參數量級時，float32 相減會吃掉精度；本地更新改用 float64 計算，再轉回 float32 給攻擊端
                import copy
                m64 = copy.deepcopy(model).double()
                clean = [u.float() for u in fedavg_update(m64, loss_fn, x_norm.double(), y, cl["steps"], cl["per_step"], cl["lr"])]
            else:
                clean = fedavg_update(model, loss_fn, x_norm, y, cl["steps"], cl["per_step"], cl["lr"])
        elif dp_sigmas is not None:
            from trainers.fl_client import dp_clipped_gradient
            clean, g_norms = dp_clipped_gradient(model, loss_fn, x_norm, y, cl["clip"])
        else:
            clean = fedsgd_gradient(model, loss_fn, x_norm, y, cfg.get("micro_batch", 8))
        for sigma in (dp_sigmas if dp_sigmas is not None else noises):
            if dp_sigmas is not None:
                from util.gdp import gdp_epsilon
                eps = gdp_epsilon(1.0 / sigma, cfg.get("delta", 1e-5)) if sigma > 0 else float("inf")      # 單次高斯機制 mu = 1/sigma
                scale = sigma * cl["clip"] / n
                grads = add_gradient_noise(clean, scale, seed=cfg.get("noise_seed", 0), distribution="gaussian") if sigma > 0 else clean
                sigma_dp, sigma = sigma, scale                                                            # 之後的 sigma 一律是加在梯度上的雜訊尺度
            else:
                grads = add_gradient_noise(clean, sigma, seed=cfg.get("noise_seed", 0))
            # [對應 Pseudo code: x_hat 與 指標]
            thr = thr_sd * 2 * sigma if sigma > 0 else 0.0
            rec, n_hits = reconstruct(grads, n, tuple(x01.shape[1:]), bias_threshold=thr, reduction=cfg.get("attack", {}).get("reduction", "weight"),
                                     structure=block.structure)
            rec01 = (rec * std + mean).clamp(0, 1)
            stats, _ = best_match_psnr(rec01, x01)
            row = {"num_bins": k, "noise": sigma, "n_hits": n_hits, "theory_perfect": expected_perfect_fraction(n, k), **stats}
            if dp_sigmas is not None:
                row.update({"dp_sigma": sigma_dp, "epsilon": eps, "clip": cl["clip"], "grad_norm_median": float(g_norms.median()),
                            "frac_clipped": float((g_norms > cl["clip"]).float().mean())})
            if mates is not None:
                row["iip"] = iip_score(rec01, x01, y, mates[0], mates[1])
            rows.append(row)
    out = {"name": cfg["name"], "n_images": n, "rows": rows, "seconds": round(time.time() - t0, 1)}
    d = os.path.join(root, "results", cfg["name"])
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "result.json"), "w"), indent=1)
    return out


class ImprintAttack(BaseAttack):
    """
    功能
        Robbing the Fed（imprint）的 BaseAttack 實作：run() = run_imprint_fedsgd(cfg, root)
        （客戶端類型 FedSGD / FedAvg 由 cfg["client"] 決定）。
    用法
        ImprintAttack(cfg, root="runs").run()
    """

    def run(self) -> Dict[str, Any]:
        """功能：執行 bins / 雜訊掃描並回傳各列指標。"""
        return run_imprint_fedsgd(self.cfg, self.root)
