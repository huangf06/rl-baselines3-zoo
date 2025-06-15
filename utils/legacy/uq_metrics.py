import numpy as np
from scipy.stats import spearmanr

# ---------- basic metrics ----------
def ece_regression(mu, y, n_bins=15):
    errs = np.abs(mu - y)
    conf = 1.0 / (errs + 1e-8)                  # higher = more confident
    bins = np.linspace(conf.min(), conf.max(), n_bins+1)
    ece  = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - 1.0 / (errs[m].mean()+1e-8))
    return ece

def nll_gaussian(mu, sigma, y, eps=1e-3):
    sig2 = np.maximum(sigma, eps) ** 2
    return 0.5 * np.log(2 * np.pi * sig2) + (y - mu) ** 2 / (2 * sig2)

def picp(mu, sigma, y, z):
    lower, upper = mu - z*sigma, mu + z*sigma
    return ((y >= lower) & (y <= upper)).mean()

def sharpness(sigma, z):
    return np.mean(2*z*sigma)

def error_uncert_corr(mu, sigma, y):
    if np.allclose(sigma, 0):
        return np.nan
    return spearmanr(np.abs(mu - y), sigma).correlation