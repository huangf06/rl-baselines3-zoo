#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from rl_zoo3.utils import get_saved_hyperparams, create_test_env

# ───────── config ─────────────────────────────────────────────────────────────
DATASET_PATH = "analysis/results/uq_dataset_clean.npz"
ENV_ID       = "LunarLander-v3"
ENV_SUBDIR   = f"{ENV_ID}_1"
MODEL_FILE   = "rl_model_100000_steps"
OUT_CSV      = "analysis/results/q_values_clean.csv"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MC_SAMPLES   = 20

MODEL_CLASSES = {
    "dqn":               DQN,
    "bootstrapped_dqn":  DQN,
    "qrdqn":             QRDQN,
    "mc_dropout":        DQN,
}
ROOT_DIRS = {
    "dqn":               "/var/scratch/fhu100/rlzoo-logs/lunar_dqn",
    "bootstrapped_dqn":  "/var/scratch/fhu100/rlzoo-logs/lunar_dqn",
    "qrdqn":             "/var/scratch/fhu100/rlzoo-logs/lunar_dqn",
    "mc_dropout":        "/var/scratch/fhu100/rlzoo-logs/lunar_mc_dropout",
}
SUBFOLDERS = {
    "dqn": "dqn",
    "bootstrapped_dqn": "bootstrapped_dqn",
    "qrdqn": "qrdqn",
    "mc_dropout": "mcdropout_dqn",
}
# ──────────────────────────────────────────────────────────────────────────────

dset = np.load(DATASET_PATH)
states, actions, returns, seeds_arr = dset["state"], dset["action"], dset["return_"], dset["seed"]

records = []

for algo, cls in MODEL_CLASSES.items():
    for seed in np.unique(seeds_arr):
        mask = seeds_arr == seed
        if not mask.any():
            continue

        model_dir = os.path.join(
            ROOT_DIRS[algo], f"seed_{seed}", SUBFOLDERS[algo], ENV_SUBDIR
        )
        model_path = os.path.join(model_dir, MODEL_FILE + ".zip")
        stats_dir  = os.path.join(model_dir, ENV_ID)

        if not os.path.exists(model_path):
            print(f"[skip] {model_path} missing")
            continue

        print(f"Loading {algo:<14} | seed {seed}")
        hyper, _ = get_saved_hyperparams(stats_dir)
        hyper.setdefault("normalize", False)

        env = create_test_env(
            ENV_ID, n_envs=1, stats_path=stats_dir,
            seed=int(seed), should_render=False, hyperparams=hyper
        )

        model = cls.load(
            model_path, env=env, device=DEVICE,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "exploration_schedule": lambda _: 0.0,
            },
        )
        env.close()

        for s, a, ret in tqdm(zip(states[mask], actions[mask], returns[mask]),
                              total=mask.sum(), desc=f"{algo}-{seed}", leave=False):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if algo == "bootstrapped_dqn":
                    q_all = model.policy.q_net(s_t)[0, :, a]
                elif algo == "qrdqn":
                    q_all = model.policy.quantile_net(s_t)[0, a, :]
                elif algo == "mc_dropout":
                    model.policy.train()
                    samples = [model.policy.q_net(s_t)[0, a].item() for _ in range(MC_SAMPLES)]
                    model.policy.eval()
                    q_all = torch.tensor(samples, device=DEVICE)
                else:  # vanilla dqn
                    q_all = model.policy.q_net(s_t)[0, a].unsqueeze(0)

            records.append((
                q_all.mean().item(),
                q_all.std(unbiased=False).item() if q_all.numel() > 1 else 0.0,
                ret, algo, seed
            ))

df = pd.DataFrame(records, columns=["Q_mean", "Q_std", "MC_return", "algo", "seed"])
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"[✓] Q-values saved → {OUT_CSV}")
