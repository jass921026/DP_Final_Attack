"""
檔案功能
--------
Gaussian DP（mu-GDP）換算：mu 與 (epsilon, delta) 的精確對應，以及由目標 epsilon 反推高斯機制的噪音倍率。
    mu-GDP  <=>  delta(eps) = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2)
    單次高斯機制（敏感度 C、雜訊 sigma*C，add/remove 鄰接）：mu = 1/sigma；T 步全批次組合：mu = sqrt(T)/sigma。

使用方式
--------
    from util.gdp import gdp_epsilon, sigma_for_epsilon
    gdp_epsilon(mu=1 / 0.5, delta=1e-5)               # sigma=0.5 的單次高斯機制
    sigma_for_epsilon(eps=10, delta=1e-5, steps=1)    # 反推 sigma
"""
import math

from scipy.optimize import brentq
from scipy.stats import norm


def gdp_delta(eps: float, mu: float) -> float:
    """功能：mu-GDP 在給定 eps 下的精確 delta。"""
    return float(norm.cdf(-eps / mu + mu / 2) - math.exp(min(eps + norm.logcdf(-eps / mu - mu / 2), 700)))


def gdp_epsilon(mu: float, delta: float = 1e-5) -> float:
    """功能：mu-GDP 對應的 eps（給定 delta）；mu 為無窮大時回傳 inf。"""
    if not math.isfinite(mu):
        return float("inf")
    return 0.0 if gdp_delta(0.0, mu) <= delta else float(brentq(lambda e: gdp_delta(e, mu) - delta, 0, mu * mu + 100 * mu + 10))


def sigma_for_epsilon(eps: float, delta: float = 1e-5, steps: int = 1) -> float:
    """功能：使 steps 次組合後恰為 (eps, delta)-DP 的噪音倍率 sigma（mu = sqrt(steps)/sigma）。"""
    mu = brentq(lambda m: gdp_delta(eps, m) - delta, 1e-6, 1e5)
    return math.sqrt(steps) / mu
