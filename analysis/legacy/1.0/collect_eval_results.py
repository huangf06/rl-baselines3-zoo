import os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_LOG_DIR = Path("/var/scratch/fhu100/rlzoo-logs/lunar_dqn")
ALGO = "bootstrapped_dqn"
ENV_NAME = "LunarLander-v3"
ENV_DIR_NAME = f"{ENV_NAME}_1"
SEED_LIST = [101, 307, 911, 1747, 2029, 2861, 3253, 4099, 7919, 9011]

records = []

for seed in SEED_LIST:
    eval_file = (
        ROOT_LOG_DIR / f"seed_{seed}" / ALGO / ENV_DIR_NAME / "evaluations.npz"
    )
    if not eval_file.exists():
        print(f"[Warning] Not found: {eval_file}")
        continue

    data = np.load(eval_file, allow_pickle=True)
    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    for i in range(len(timesteps)):
        records.append({
            "seed": seed,
            "timestep": timesteps[i],
            "mean_reward": results[i].mean(),
            "std_reward": results[i].std(),
            "mean_ep_length": ep_lengths[i].mean(),
            "std_ep_length": ep_lengths[i].std(),
        })

df = pd.DataFrame(records)
df = df.sort_values(by=["seed", "timestep"])
output_path = Path("results") / f"{ALGO}_{ENV_NAME.lower()}_eval.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")
