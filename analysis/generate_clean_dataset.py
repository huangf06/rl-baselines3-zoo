#!/usr/bin/env python
"""
Collect offline trajectories (state, action, MC return) for four algorithms.

Compatible with:
  • Gymnasium >= 0.28   (step -> 5 outputs)
  • Gym <= 0.26         (step -> 4 outputs)
"""

import os
import yaml
import numpy as np
from tqdm import tqdm
import torch
import gymnasium as gym

from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from rl_zoo3.utils import get_saved_hyperparams, create_test_env

# ───────── config ─────────────────────────────────────────────────────────────
ALGOS          = ["dqn", "bootstrapped_dqn", "qrdqn", "mc_dropout"]
ENV_ID         = "LunarLander-v3"
ENV_SUBDIR     = f"{ENV_ID}_1"
MODEL_FILE     = "rl_model_100000_steps"
GAMMA          = 0.99
EVAL_EPISODES  = 30
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH    = "analysis/results/uq_dataset_clean.npz"
SEEDS_YAML     = "seeds_master.yml"

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
MODEL_CLS = {
    "dqn": DQN,
    "bootstrapped_dqn": DQN,
    "qrdqn": QRDQN,
    "mc_dropout": DQN,
}
# ──────────────────────────────────────────────────────────────────────────────

def load_seeds(path: str) -> list[int]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return [int(e["value"]) for e in data["seeds"]]

def mc_returns(rewards, gamma):
    g, out = 0.0, []
    for r in reversed(rewards):
        g = r + gamma * g
        out.insert(0, g)
    return out

def reset_env(env):
    res = env.reset()
    return res[0] if isinstance(res, tuple) else res

def step_env(env, act):
    res = env.step(act)
    if len(res) == 5:                                  # gymnasium API
        obs, reward, terminated, truncated, _ = res
        done = terminated or truncated
    else:                                              # legacy gym
        obs, reward, done, _ = res
    reward_val = reward.item() if isinstance(reward, np.ndarray) else float(reward)
    return obs, reward_val, bool(done)

# storage
states, actions, returns, algos, seeds = [], [], [], [], []

for algo in ALGOS:
    for seed in tqdm(load_seeds(SEEDS_YAML), desc=algo, leave=False):
        model_dir = os.path.join(
            ROOT_DIRS[algo],
            f"seed_{seed}",
            SUBFOLDERS[algo],
            ENV_SUBDIR,
        )
        model_zip = os.path.join(model_dir, MODEL_FILE + ".zip")
        stats_dir = os.path.join(model_dir, ENV_ID)

        if not os.path.exists(model_zip):
            print(f"[skip] {model_zip} not found")
            continue

        hyper, _ = get_saved_hyperparams(stats_dir)
        hyper.setdefault("normalize", False)

        env = create_test_env(
            ENV_ID, n_envs=1, stats_path=stats_dir,
            seed=seed, should_render=False, hyperparams=hyper
        )

        model = MODEL_CLS[algo].load(
            model_zip, env=env, device=DEVICE,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "exploration_schedule": lambda _: 0.0,
            },
        )

        for _ in range(EVAL_EPISODES):
            obs = reset_env(env)
            done = False
            ep_obs, ep_act, ep_rew = [], [], []

            while not done:
                act, _ = model.predict(obs, deterministic=True)
                ep_obs.append(obs[0].copy())
                ep_act.append(int(act[0]))
                obs, r, done = step_env(env, act)
                ep_rew.append(r)

            mc = mc_returns(ep_rew, GAMMA)
            states.extend(ep_obs)
            actions.extend(ep_act)
            returns.extend(mc)
            algos.extend([algo] * len(ep_obs))
            seeds.extend([seed] * len(ep_obs))

        env.close()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.savez_compressed(
    OUTPUT_PATH,
    state=np.asarray(states,  dtype=np.float32),
    action=np.asarray(actions, dtype=np.int32),
    return_=np.asarray(returns, dtype=np.float32),
    algo=np.asarray(algos),
    seed=np.asarray(seeds, dtype=np.int32),
)
print(f"[✓] dataset saved → {OUTPUT_PATH}")
