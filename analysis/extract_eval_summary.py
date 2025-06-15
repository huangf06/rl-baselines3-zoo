#!/usr/bin/env python3
"""
Extract evaluation results (timesteps vs. mean reward)
from all trained models, organized by algorithm and seed.
"""

import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

ALGOS = ["dqn", "bootstrapped_dqn", "qrdqn", "mc_dropout"]
ENV_ID = "LunarLander-v3"
ENV_SUBDIR = f"{ENV_ID}_1"
EVAL_FILE = "evaluations.npz"
SEEDS_YAML = "seeds_master.yml"

ROOT_DIRS = {
    "dqn":               "/var/scratch/fhu100/rlzoo-logs/legacy/lunar_dqn",
    "bootstrapped_dqn":  "/var/scratch/fhu100/rlzoo-logs/legacy/lunar_dqn",
    "qrdqn":             "/var/scratch/fhu100/rlzoo-logs/legacy/lunar_dqn",
    "mc_dropout":        "/var/scratch/fhu100/rlzoo-logs/legacy/lunar_mc_dropout",
}
SUBFOLDERS = {
    "dqn": "dqn",
    "bootstrapped_dqn": "bootstrapped_dqn",
    "qrdqn": "qrdqn",
    "mc_dropout": "mcdropout_dqn",
}
OUTPUT_CSV = "analysis/results/eval_summary.csv"

def load_seeds(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [int(e["value"]) for e in data["seeds"]]

records = []

for algo in ALGOS:
    for seed in tqdm(load_seeds(SEEDS_YAML), desc=f"{algo:<18}"):
        eval_path = Path(ROOT_DIRS[algo]) / f"seed_{seed}" / SUBFOLDERS[algo] / ENV_SUBDIR / EVAL_FILE
        if not eval_path.exists():
            print(f"[skip] {eval_path} not found")
            continue

        try:
            data = np.load(eval_path)
            timesteps = data["timesteps"]
            rewards = data["results"].mean(axis=1)
            for t, r in zip(timesteps, rewards):
                records.append({
                    "algo": algo,
                    "seed": seed,
                    "timesteps": int(t),
                    "ep_rew_mean": float(r),
                })
        except Exception as e:
            print(f"[error] failed to read {eval_path}: {e}")

df = pd.DataFrame(records)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[✓] saved summary CSV → {OUTPUT_CSV}")
