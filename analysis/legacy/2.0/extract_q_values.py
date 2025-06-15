"""
Extract Q-value predictions from trained models and save them once to CSV.
Run this only when you change models or collect a new dataset.
"""

import os, sys, numpy as np, pandas as pd, torch
from tqdm import tqdm
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from rl_zoo3.utils import get_saved_hyperparams, create_test_env

# ------------------------------------------------- configuration
DATASET_PATH = "analysis/results/uq_dataset_clean.npz"
ROOT_DIR     = "/var/scratch/fhu100/rlzoo-logs/lunar_dqn"
ENV_ID       = "LunarLander-v3"
OUT_CSV      = "analysis/results/q_values_clean.csv"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CLASSES = {"dqn": DQN, "bootstrapped_dqn": DQN, "qrdqn": QRDQN}
ENV_SUBDIR    = {k: f"{ENV_ID}_1" for k in MODEL_CLASSES}  # all use *_1
# -------------------------------------------------

# load cached transitions (state, action, MC return, algo, seed)
data      = np.load(DATASET_PATH)
states    = data["state"]
actions   = data["action"]
returns   = data["return_"]
algos_arr = data["algo"]
seeds_arr = data["seed"]

records = []

for algo in np.unique(algos_arr):
    for seed in np.unique(seeds_arr):
        mask = (algos_arr == algo) & (seeds_arr == seed)
        if not mask.any():
            continue

        subdir  = ENV_SUBDIR[algo]
        model_dir  = os.path.join(ROOT_DIR, f"seed_{seed}", algo, subdir)
        model_path = os.path.join(model_dir, "rl_model_100000_steps")
        stats_path = os.path.join(model_dir, ENV_ID)
        if not os.path.exists(model_path + ".zip"):
            print(f"[skip] {model_path}.zip missing")
            continue

        print(f"Loading {algo} | seed {seed}")
        hyper, _ = get_saved_hyperparams(stats_path)
        hyper.setdefault("normalize", False)

        env = create_test_env(ENV_ID, n_envs=1, stats_path=stats_path,
                              seed=int(seed), should_render=False,
                              hyperparams=hyper)
        model = MODEL_CLASSES[algo].load(
            model_path, env=env, device=DEVICE,
            custom_objects={"learning_rate": 0.0,
                            "lr_schedule": lambda _: 0.0,
                            "exploration_schedule": lambda _: 0.0})
        env.close()

        for s, a, ret in tqdm(zip(states[mask], actions[mask], returns[mask]),
                              total=mask.sum(), desc=f"{algo}-{seed}", leave=False):
            with torch.no_grad():
                s_t = torch.as_tensor(s, dtype=torch.float32,
                                      device=DEVICE).unsqueeze(0)
                # locate value net
                if hasattr(model, "q_net"):
                    q_out = model.q_net(s_t)
                elif hasattr(model.policy, "q_net"):
                    q_out = model.policy.q_net(s_t)
                elif hasattr(model.policy, "quantile_net"):
                    q_out = model.policy.quantile_net(s_t)
                else:
                    raise AttributeError("No Q network found.")

                if algo == "bootstrapped_dqn":      # [B,H,A]
                    q_all = q_out[0, :, a]
                elif algo == "qrdqn":               # [B,A,Nq]
                    q_all = q_out[0, a, :]
                else:                               # vanilla DQN
                    q_all = q_out[0, a].unsqueeze(0)

                q_mean = q_all.mean().item()
                q_std  = q_all.std().item() if q_all.numel() > 1 else 0.0

            records.append((q_mean, q_std, ret, algo, seed))

# save
df = pd.DataFrame(records, columns=["Q_mean", "Q_std", "MC_return",
                                    "algo", "seed"])
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"cached Q-values -> {OUT_CSV}")
