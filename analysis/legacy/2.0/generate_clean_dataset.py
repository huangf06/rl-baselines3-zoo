import os
import numpy as np
import torch
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_zoo3.utils import get_saved_hyperparams, create_test_env
import gymnasium as gym
from tqdm import tqdm
import yaml

def load_seeds(yaml_path="seeds_master.yml"):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["value"] for entry in data["seeds"]]

# === CONFIG ===
ALGOS = ["dqn", "bootstrapped_dqn", "qrdqn"]
ENV_ID = "LunarLander-v3"
SEEDS = load_seeds()
DEVICE = "cuda"
EVAL_EPISODES = 30
GAMMA = 0.99
ROOT_DIR = "/var/scratch/fhu100/rlzoo-logs/lunar_dqn"
OUTPUT_PATH = "analysis/results/uq_dataset_clean.npz"
ENV_SUBDIR = f"{ENV_ID}_1"

# === Utility: compute MC return
def compute_mc_return(rewards, gamma):
    G = 0.0
    mc_returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        mc_returns.insert(0, G)
    return mc_returns

# === Main ===
states, actions, returns, algos, seeds = [], [], [], [], []

for algo in ALGOS:
    for seed in tqdm(SEEDS, desc=f"Processing {algo}"):
        # Paths
        model_dir = os.path.join(ROOT_DIR, f"seed_{seed}", algo, ENV_SUBDIR)
        model_path = os.path.join(model_dir, "rl_model_100000_steps")
        stats_path = os.path.join(model_dir, ENV_ID)

        # Load environment
        hyperparams, _ = get_saved_hyperparams(stats_path)
        if "normalize" not in hyperparams:
            hyperparams["normalize"] = False

        env = create_test_env(
            env_id=ENV_ID,
            n_envs=1,
            stats_path=stats_path,
            seed=int(seed),
            log_dir=None,
            should_render=False,
            hyperparams=hyperparams,
        )

        # Load model
        ModelCls = {"dqn": DQN, "bootstrapped_dqn": DQN, "qrdqn": QRDQN}[algo]
        model = ModelCls.load(
            model_path,
            env=env,
            device=DEVICE,
            buffer_size=1,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "exploration_schedule": lambda _: 0.0,
            },
        )

        # Rollout collection
        for _ in range(EVAL_EPISODES):
            obs = env.reset()
            done = [False]
            ep_obs, ep_act, ep_rew = [], [], []

            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                ep_obs.append(obs[0].copy())
                ep_act.append(action[0])
                obs, reward, done, _ = env.step(action)
                ep_rew.append(reward[0])

            mc_ret = compute_mc_return(ep_rew, GAMMA)
            states.extend(ep_obs)
            actions.extend(ep_act)
            returns.extend(mc_ret)
            algos.extend([algo] * len(ep_obs))
            seeds.extend([seed] * len(ep_obs))

        env.close()

# === Save dataset
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.savez_compressed(
    OUTPUT_PATH,
    state=np.array(states, dtype=np.float32),
    action=np.array(actions, dtype=np.int32),
    return_=np.array(returns, dtype=np.float32),
    algo=np.array(algos),
    seed=np.array(seeds, dtype=np.int32),
)
print(f"Saved clean UQ dataset to: {OUTPUT_PATH}")
