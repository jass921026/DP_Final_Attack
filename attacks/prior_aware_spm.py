"""
已知差異（待修,2026-11-25 記）
----------------------------
本檔搬自第三方後續論文的 repo（`h-0-0/info_reconstruction_bound`，arXiv 2610.13768），不是 Hayes et al. 2023
原作者的程式（原作者沒有釋出 code）。跟 Hayes 2023 論文本身的 Algorithm 2（prior-aware attack）有出入：
    1. 論文的分數是 Σ_t ⟨clip_C(∇loss(θ_t, z_i)), ḡ_t⟩，用真實模型在每步 θ_t 上算出的候選梯度，輸出是候選池上的
       argmax；本檔是全維度高斯似然比 + softmax posterior mean（對應論文 App. G 的變體，論文沒報告結果）。
    2. `analysis/mnist_spm_repro.py` / `analysis/p12_spm_vs_balle.py` 用的是「釋出 = 裁剪過的資料本身 + 雜訊」的理想化
       設定（後續論文為了對照他們自己的理論界所做的簡化），不是真實模型的梯度；真實梯度版在
       `analysis/p12_gradient_spm.py`（單步、θ₀、D⁻=∅）。
    3. 多步（T>1）版沒有處理 θ_t 隨訓練軌跡變動、也沒有扣除已知 batch 梯度的步驟。
所以這裡的數字不能直接當成「Hayes 2023 的結果」引用。

續篇(子抽樣/多步版,2026-11-25 補)
--------------------------------
原本只搬了單次全批次釋出（q=1, T=1）。這裡補上論文/repo 的 Poisson 子抽樣、多步版（`gen_release_sub` +
`gamma_prv_mean` 的 q<1 積分 + `solve_sigma`），讓 SPM 能在「T=100 步」這個跟 Balle weight-ReCoNN 同等級的
設定下測，是更對等的比較（Balle 本來就是 100 步 DP-GD；原本的 SPM 單步版雖然已經用 mu-GDP 換算成同一個 ε
比過，但「幾步釋出」這個維度本身沒有對齊）。

檔案功能
--------
Hayes, Balle, Mahloujifar（NeurIPS 2023,arXiv 2302.07225）App. G 的 shadow posterior-mean（SPM）攻擊，從
`h-0-0/info_reconstruction_bound`（"From Pixels to Privacy-Sensitive Features" 的官方 code,`src/attacks/spm.py`）
搬過來、改寫成本專案風格。只搬「單次全批次釋出」（q=1, T=1）這一段（對應論文 Algorithm 2 的 prior-aware 攻擊，
但這裡的似然是全維度的高斯似然比，不是只對候選集打一個標量分數）；論文裡的 Poisson 子抽樣版（q<1, T>1,
`gen_release_sub`）沒有搬,需要時可以再補。

威脅模型(跟我們的 Imprint/Balle weight-ReCoNN 都不一樣,是第三種):攻擊者知道候選集合 D_shadow(有限的候選
池,不是連續空間)、知道 DP 機制的 clip C 跟雜訊標準差、觀測到的「梯度」就是被裁剪過的候選本身(見下方
pseudo code 的 "idealized release")——這是論文用來測「攻擊方法本身的統計極限」的簡化設定,不實際訓練模型:
把某個候選資料點 z 直接當成它自己的「per-example 梯度貢獻」(z 本身去中心化、裁剪),等於是在問「如果有一個
機制忠實地回報了裁剪過的目標點 + 高斯雜訊,攻擊者最多能從候選池裡挑多準」,跟 Imprint/Balle 需要真的訓練
一個分類器不同。

Pseudo code(論文 App. G,q=1,T=1 的情況)
    候選 z_j ∈ D_shadow(每個候選 j),中心化 u_j = z_j - mu(mu = D_shadow 的平均)
    裁剪 c_j = a_j * u_j,a_j = min(1, C / ||u_j||)                              # clip factor
    釋出(對某個真正的目標 z*)：G = c_* + N(0, sigma^2 C^2 I)                     # sigma 由 gamma(bits) 反推
    對每個候選算 log-likelihood ratio：L_j = (2<G,c_j> - ||c_j||^2) / (2 sigma^2 C^2)
    posterior 權重 w_j = softmax_j(L_j)
    重建 û = sum_j w_j * (c_j / a_j)                                            # posterior mean(換回未裁剪座標)

使用方式
--------
    from attacks.prior_aware_spm import nm_of_gamma, spm_attack
    sigma = nm_of_gamma(gamma_bits=8.01)                    # 這個 gamma(bits)對應的雜訊標準差倍率
    u_mean, u_map = spm_attack(targets_u, candidates_u, clip=0.3, noise_std=sigma*0.3)
"""
import math
from typing import Tuple

import numpy as np
import torch

LN2 = math.log(2.0)


def gamma_prv_mean(T: int, q: float, sigma: float, n_quad: int = 256) -> float:
    """
    功能
        T 步、取樣率 q、雜訊倍率 sigma 的 Poisson 子抽樣高斯機制洩漏多少 bits（nats）。q=1 時有解析解
        T/(2 sigma^2)；q<1 用 Gauss-Hermite 積分算 KL(P||Q)，P=(1-q)N(0,sigma^2)+q N(1,sigma^2)，Q=N(0,sigma^2)。
    Pseudo code
        log(P/Q)(x) = log((1-q) + q*exp((2x-1)/(2 sigma^2)))
        KL = (1-q) E_{N(0,sigma^2)}[log(P/Q)] + q E_{N(1,sigma^2)}[log(P/Q)]
        gamma(nats) = T * KL
    """
    if q <= 0.0:
        return 0.0
    s2 = sigma ** 2
    if q >= 1.0:
        return T * 0.5 / s2
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    sqrt2, sqrtpi = math.sqrt(2.0), math.sqrt(math.pi)
    log1mq, logq = math.log(1.0 - q), math.log(q)

    def log_ratio(x):
        return np.logaddexp(log1mq, logq + (2.0 * x - 1.0) / (2.0 * s2))

    def gauss_expect(mu):
        xs = mu + sqrt2 * sigma * nodes
        return float(np.sum(weights * log_ratio(xs)) / sqrtpi)

    return T * ((1.0 - q) * gauss_expect(0.0) + q * gauss_expect(1.0))


def solve_sigma(T: int, q: float, gamma_bits: float) -> float:
    """功能：二分搜尋使 T 步、取樣率 q 恰好洩漏 gamma_bits 個 bit 的雜訊倍率 sigma。"""
    lo, hi = 1e-3, 1e4
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if gamma_prv_mean(T, q, mid) / LN2 > gamma_bits else (lo, mid)
    return math.sqrt(lo * hi)


def nm_of_gamma(gamma_bits: float) -> float:
    """功能：單次全批次釋出（q=1, T=1）要洩漏 gamma_bits 個 bit 時,所需的雜訊標準差倍率 sigma。
    Pseudo code
        gamma(nats) = 1 / (2 sigma^2)  =>  sigma = sqrt(1 / (2 gamma(nats))),gamma(nats) = gamma_bits * ln2
    """
    return math.sqrt(0.5 / (gamma_bits * LN2))


def clip_candidates(u: torch.Tensor, clip: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    功能
        把一批「已去中心化」的候選/目標向量裁剪到範數 <= clip。
    Pseudo code
        a_j <- min(1, clip / ||u_j||)；c_j <- a_j * u_j
    用法
        c, a, norm2 = clip_candidates(u, clip=0.3)   # c=裁剪後向量, a=裁剪係數, norm2=||c||^2
    """
    norm = u.norm(dim=-1).clamp_min(1e-12)
    a = torch.clamp(clip / norm, max=1.0)
    c = a.unsqueeze(-1) * u
    return c, a, (c * c).sum(-1)


def gen_release_single(clipped_targets: torch.Tensor, noise_std: float, seed: int) -> torch.Tensor:
    """功能：單次全批次釋出 G = c_* + N(0, noise_std^2 I)（見檔頭 pseudo code）。"""
    g = torch.Generator(device=clipped_targets.device).manual_seed(seed)
    return clipped_targets + torch.randn(clipped_targets.shape, generator=g, device=clipped_targets.device) * noise_std


def gen_release_sub(clipped_target: torch.Tensor, noise_std: float, seed: int, T: int, q: float) -> torch.Tensor:
    """
    功能
        單一目標的 T 步 Poisson 子抽樣釋出：每一步以機率 q「參與」（這一步的釋出裡加上這個目標），不管有沒有
        參與都加一次雜訊；攻擊者看得到全部 T 步的釋出，但看不到哪幾步真的有參與（見檔頭 pseudo code 的
        mixture likelihood，`q<1` 時 attack() 要用這個）。
    Pseudo code
        for t in 1..T: 1_t ~ Bernoulli(q)；G_t <- 1_t * c_* + N(0, noise_std^2 I)
    """
    d = clipped_target.numel()
    g = torch.Generator(device=clipped_target.device).manual_seed(seed)
    participate = (torch.rand(T, generator=g, device=clipped_target.device) < q).float()
    noise = torch.randn(T, d, generator=g, device=clipped_target.device) * noise_std
    return participate[:, None] * clipped_target[None, :] + noise


def spm_attack(releases: torch.Tensor, cand_clipped: torch.Tensor, cand_norm2: torch.Tensor, cand_clip_factor: torch.Tensor,
               noise_std: float, q: float = 1.0, chunk: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能
        對每個釋出的 release(可以是 (n,1,d) 的單步、或 (n,T,d) 的多步)，在候選集 cand_clipped 上算
        posterior（softmax 的 log-likelihood,q<1 時用混合似然),回傳 posterior mean（還原到未裁剪座標）與
        MAP（最高分候選本身）。
    Pseudo code
        L_tj <- (2<G_t,c_j> - ||c_j||^2) / (2 noise_std^2)           # 每一步 t、每個候選 j
        q=1：score_j <- sum_t L_tj；q<1：score_j <- sum_t log((1-q) + q*exp(L_tj))（不知道哪幾步真的參與）
        w   <- softmax_j(score)
        u_mean <- sum_j w_j * (c_j / a_j)；u_map <- c_{argmax} / a_{argmax}
    用法
        u_mean, u_map = spm_attack(G[:,None,:], cand, cand_norm2, cand_a, noise_std)                 # 單步
        u_mean, u_map = spm_attack(G_T, cand, cand_norm2, cand_a, noise_std, q=0.02)                 # T 步子抽樣
    """
    n_targets, n_cand, d = releases.shape[0], cand_clipped.shape[0], cand_clipped.shape[-1]
    scores = torch.empty(n_targets, n_cand, dtype=torch.float64, device=releases.device)
    log1mq, logq = (math.log(1.0 - q), math.log(q)) if q < 1.0 else (None, None)
    for i in range(0, n_cand, chunk):
        cand = cand_clipped[i:i + chunk]
        dots = torch.einsum("ntd,cd->ntc", releases, cand).double()                      # (n_targets, T, chunk)
        llr = (2.0 * dots - cand_norm2[i:i + chunk][None, None, :].double()) / (2.0 * noise_std ** 2)
        if q >= 1.0:
            scores[:, i:i + chunk] = llr.sum(1)
        else:
            scores[:, i:i + chunk] = torch.logaddexp(torch.full_like(llr, log1mq), logq + llr).sum(1)
    weights = torch.softmax(scores, 1)
    j_map = torch.argmax(scores, 1)
    unclip = 1.0 / cand_clip_factor
    u_mean = torch.zeros(n_targets, d, dtype=torch.float32, device=releases.device)
    u_map = torch.zeros(n_targets, d, dtype=torch.float32, device=releases.device)
    for i in range(0, n_cand, chunk):
        cand = cand_clipped[i:i + chunk]
        u_mean += (weights[:, i:i + chunk].float() * unclip[i:i + chunk][None]) @ cand
        hit = (j_map >= i) & (j_map < i + chunk)
        if hit.any():
            idx = j_map[hit] - i
            u_map[hit] = cand[idx] * unclip[i:i + chunk][idx][:, None]
    return u_mean, u_map
