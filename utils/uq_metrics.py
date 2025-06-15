import numpy as np
from scipy.special import erf
from math import sqrt, pi

SQRT2 = sqrt(2.0)
SQRT2PI = sqrt(2.0 * pi)

def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / SQRT2PI

def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / SQRT2))

def nll_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    s2 = sigma ** 2
    return 0.5 * (np.log(2.0 * pi * s2) + (y - mu) ** 2 / s2)

def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / sqrt(pi))

def ece_regression(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                   n_bins: int = 10, adaptive: bool = True) -> float:
    sigma = np.maximum(sigma, 1e-6)
    p = _Phi((y - mu) / sigma)
    if adaptive:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    N = len(p)
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        mask = (p >= left) & (p < right) if i < n_bins - 1 else (p >= left) & (p <= right)
        if not mask.any():
            continue
        acc_bin = np.mean((p[mask] <= right) & (p[mask] >= left))
        conf_bin = (left + right) / 2.0
        ece += np.abs(acc_bin - conf_bin) * mask.mean()
    return ece

def cece(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
         q_bins: int = 3, t_bins: int = 3) -> float:
    n = len(mu)
    if n < q_bins * t_bins * 30:
        return np.nan
    q_idx = np.digitize(
        mu,
        np.quantile(mu, np.linspace(0.0, 1.0, q_bins + 1))[1:-1],
        right=False,
    )
    t_rel = np.arange(n) / n
    t_idx = np.digitize(
        t_rel,
        np.linspace(0.0, 1.0, t_bins + 1)[1:-1],
        right=False,
    )

    eces = []
    for i in range(q_bins):
        for j in range(t_bins):
            mask = (q_idx == i) & (t_idx == j)
            if mask.sum() < 30:
                continue
            eces.append(
                ece_regression(mu[mask], sigma[mask], y[mask],
                               n_bins=10, adaptive=True)
            )
    return float(np.mean(eces)) if eces else np.nan

def picp(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, z: float) -> float:
    return np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma))

def sharpness(sigma: np.ndarray, z: float) -> float:
    return np.mean(2.0 * z * sigma)

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

def error_uncert_corr(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    err = np.abs(mu - y)
    return _spearman(err, sigma)

def aurc(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    err = np.abs(mu - y)
    order = np.argsort(sigma)
    err_sorted = err[order]
    risk_cumsum = np.cumsum(err_sorted)
    cover_range = np.arange(1, len(err) + 1)
    risk_curve = risk_cumsum / cover_range
    auc = np.trapz(risk_curve, cover_range / len(err))
    return auc / np.max(err)

def selective_return_curve(
    returns: np.ndarray,
    sigma: np.ndarray,
    cover_grid: np.ndarray = np.linspace(1.0, 0.5, 11),
) -> np.ndarray:
    order = np.argsort(sigma)
    returns_sorted = returns[order]
    N = len(returns)
    out = []
    for c in cover_grid:
        k = int(np.ceil(c * N))
        out.append(np.mean(returns_sorted[:k]))
    return np.array(out)

def selective_return_auc(curve: np.ndarray, cover_grid: np.ndarray) -> float:
    return float(np.trapz(curve, cover_grid) / (cover_grid[0] - cover_grid[-1]))
