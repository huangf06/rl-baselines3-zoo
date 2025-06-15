#!/usr/bin/env python
"""
Compute UQ metrics from q_values_clean.csv and save:
• metrics_raw.csv     — per (algo, seed)
• metrics_summary.csv — aggregated across all samples
• sr_curves.npz       — selective-return curves for plotting
"""

import os
import sys
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.uq_metrics import (
    nll_gaussian,
    crps_gaussian,
    ece_regression,
    cece,
    picp,
    sharpness,
    error_uncert_corr,
    aurc,
    selective_return_curve,
    selective_return_auc,
)

# Config
IN_CSV  = "analysis/results/q_values_clean.csv"
OUT_RAW = "analysis/results/metrics_raw.csv"
OUT_SUM = "analysis/results/metrics_summary.csv"
OUT_CURVES = "analysis/results/sr_curves.npz"

Z50 = 0.674  # ~50% interval
Z90 = 1.645  # ~90% interval
COVER_GRID = np.linspace(1.0, 0.5, 11)

# Load Q-value predictions
df = pd.read_csv(IN_CSV)
assert set(["Q_mean", "Q_std", "MC_return", "algo", "seed"]).issubset(df.columns)

raw_rows = []
sr_data = {}

# Compute metrics per (algo, seed)
for algo in df["algo"].unique():
    sr_data[algo] = []
    for seed in df[df["algo"] == algo]["seed"].unique():
        sub = df[(df["algo"] == algo) & (df["seed"] == seed)]

        mu = sub["Q_mean"].to_numpy()
        sig = np.clip(sub["Q_std"].to_numpy(), a_min=1e-6, a_max=None)
        y = sub["MC_return"].to_numpy()

        # Selective-return curve
        sr_curve = selective_return_curve(y, sig, cover_grid=COVER_GRID)
        sr_auc = selective_return_auc(sr_curve, COVER_GRID)

        m = dict(
            algo=algo,
            seed=int(seed),
            rmse=np.sqrt(np.mean((mu - y) ** 2)),
            mae=np.mean(np.abs(mu - y)),
            nll=np.mean(nll_gaussian(mu, sig, y)),
            crps=np.mean(crps_gaussian(mu, sig, y)),
            ece=ece_regression(mu, sig, y, n_bins=15, adaptive=True),
            cece=cece(mu, sig, y),
            cover50=picp(mu, sig, y, z=Z50),
            cover90=picp(mu, sig, y, z=Z90),
            sharp50=sharpness(sig, z=Z50),
            sharp90=sharpness(sig, z=Z90),
            rho=error_uncert_corr(mu, sig, y),
            aurc=aurc(mu, sig, y),
            sr_auc=sr_auc,
            sr_90=sr_curve[np.where(COVER_GRID == 0.9)[0][0]],
        )
        raw_rows.append(m)
        sr_data[algo].append(sr_curve)

# Save raw metric table
raw_df = pd.DataFrame(raw_rows)
os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
raw_df.to_csv(OUT_RAW, index=False)
print(f"[✓] Raw metrics → {OUT_RAW}")

# Save summary table (aggregated over all samples)
summary_rows = []
for algo in df["algo"].unique():
    sub = df[df["algo"] == algo]
    mu = sub["Q_mean"].to_numpy()
    sig = np.clip(sub["Q_std"].to_numpy(), a_min=1e-6, a_max=None)
    y = sub["MC_return"].to_numpy()

    summary_rows.append(dict(
        algo=algo,
        rmse=np.sqrt(np.mean((mu - y) ** 2)),
        mae=np.mean(np.abs(mu - y)),
        nll=np.mean(nll_gaussian(mu, sig, y)),
        crps=np.mean(crps_gaussian(mu, sig, y)),
        ece=ece_regression(mu, sig, y, n_bins=15, adaptive=True),
        cece=cece(mu, sig, y),
        cover50=picp(mu, sig, y, z=Z50),
        cover90=picp(mu, sig, y, z=Z90),
        sharp50=sharpness(sig, z=Z50),
        sharp90=sharpness(sig, z=Z90),
        rho=error_uncert_corr(mu, sig, y),
        aurc=aurc(mu, sig, y),
        sr_auc=np.mean([row["sr_auc"] for row in raw_rows if row["algo"] == algo]),
        sr_90=np.mean([row["sr_90"] for row in raw_rows if row["algo"] == algo]),
    ))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUM, index=False)
print(f"[✓] Summary metrics → {OUT_SUM}")

# Save selective-return curves for plotting
np.savez_compressed(
    OUT_CURVES,
    cover_grid=COVER_GRID,
    **{f"{algo}": np.vstack(sr_data[algo]) for algo in sr_data}
)
print(f"[✓] Selective-return curves → {OUT_CURVES}")

# Print summary table
print("\n=== Summary (mean over all samples) ===")
print(summary_df.round(4).to_string(index=False))