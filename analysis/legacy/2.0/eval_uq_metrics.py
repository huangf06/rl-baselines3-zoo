#!/usr/bin/env python
"""
Compute UQ metrics from cached q_values_clean.csv
• metrics_raw.csv    — each (algo, seed) row
• metrics_summary.csv— weighted over all samples (recommended)
"""

import os, sys
import numpy as np
import pandas as pd
from utils.uq_metrics import (
    ece_regression,
    nll_gaussian,
    picp,
    sharpness,
    error_uncert_corr,
)

IN_CSV       = "analysis/results/q_values_clean.csv"
OUT_RAW_CSV  = "analysis/results/metrics_raw.csv"
OUT_SUM_CSV  = "analysis/results/metrics_summary.csv"

df = pd.read_csv(IN_CSV)
records = []

# z-scores for 50 % / 90 % Gaussian intervals
z50, z90 = 0.674, 1.645

# ---------- 1. per-seed metrics ----------
for algo in df["algo"].unique():
    for seed in df[df["algo"] == algo]["seed"].unique():
        sub = df[(df["algo"] == algo) & (df["seed"] == seed)]
        mu, sigma, y = sub["Q_mean"].to_numpy(), sub["Q_std"].to_numpy(), sub["MC_return"].to_numpy()
        sigma_safe   = np.maximum(sigma, 1e-3)

        m = dict(
            algo = algo,
            seed = int(seed),
            rmse = np.sqrt(np.mean((mu - y) ** 2)),
            mae  = np.mean(np.abs(mu - y)),
            ece  = ece_regression(mu, y, n_bins=10),
            nll  = nll_gaussian(mu, sigma_safe, y).mean(),
        )

        if np.allclose(sigma, 0):
            m.update(cover50=np.nan, cover90=np.nan,
                     sharp50=np.nan, sharp90=np.nan, rho=np.nan)
        else:
            m.update(
                cover50 = picp(mu, sigma_safe, y, z50),
                cover90 = picp(mu, sigma_safe, y, z90),
                sharp50 = sharpness(sigma_safe, z50),
                sharp90 = sharpness(sigma_safe, z90),
                rho     = error_uncert_corr(mu, sigma_safe, y)
            )
        records.append(m)

raw_df = pd.DataFrame(records)
os.makedirs(os.path.dirname(OUT_RAW_CSV), exist_ok=True)
raw_df.to_csv(OUT_RAW_CSV, index=False)
print(f"Raw metrics saved  -> {OUT_RAW_CSV}")

# ---------- 2. weighted summary over all samples ----------
summary_rows = []
for algo in df["algo"].unique():
    sub  = df[df["algo"] == algo]
    mu   = sub["Q_mean"].to_numpy()
    sig  = sub["Q_std"].to_numpy()
    y    = sub["MC_return"].to_numpy()
    sig_safe = np.maximum(sig, 1e-3)

    summary_rows.append(dict(
        algo     = algo,
        rmse     = np.sqrt(np.mean((mu - y) ** 2)),
        mae      = np.mean(np.abs(mu - y)),
        ece      = ece_regression(mu, y, n_bins=10),
        nll      = nll_gaussian(mu, sig_safe, y).mean(),
        cover50  = picp(mu, sig_safe, y, z50),
        cover90  = picp(mu, sig_safe, y, z90),
        sharp50  = sharpness(sig_safe, z50),
        sharp90  = sharpness(sig_safe, z90),
        rho      = error_uncert_corr(mu, sig_safe, y)
    ))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUM_CSV, index=False)
print(f"Summary saved      -> {OUT_SUM_CSV}")

print("\n=== Weighted mean over ALL samples ===")
print(summary_df.round(4).to_string(index=False))
