# run_uq_analysis.py

import os
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import DQN, QRDQN
from stable_baselines3.common.vec_env import VecEnvWrapper
from rl_zoo3.utils import get_saved_hyperparams, create_test_env
from utils.uq_metrics import compute_ece

# === CONFIG ===
ALGO = "bootstrapped_dqn"  # "dqn", "bootstrapped_dqn", "qr_dqn"
ENV_ID = "LunarLander-v3"
SEED = 101
DEVICE = "cuda"
EVAL_EPISODES = 30
ROOT_DIR = "/var/scratch/fhu100/rlzoo-logs/lunar_dqn"
ENV_DIR_NAME = f"{ENV_ID}_1"
MODEL_DIR = os.path.join(ROOT_DIR, f"seed_{SEED}", ALGO, ENV_DIR_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, "rl_model_100000_steps.zip")
STATS_PATH = os.path.join(MODEL_DIR, ENV_ID)

# === SCENARIOS ===
SCENARIOS = [
    ("clean", False, 0.0),
    ("noise01", False, 0.1),
    ("po", True, 0.0),
    ("noise01_po", True, 0.1),
]

# === Noise and PO Wrappers ===

class VecObsNoise(VecEnvWrapper):
    def __init__(self, venv, noise_std: float):
        super().__init__(venv)
        self.noise_std = noise_std
    def reset(self):
        obs = self.venv.reset()
        return self._add_noise(obs)
    def step_async(self, actions):
        self.venv.step_async(actions)
    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self._add_noise(obs), rews, dones, infos
    def _add_noise(self, obs):
        noisy = obs + self.noise_std * np.random.randn(*obs.shape)
        return np.clip(noisy, -1.0, 1.0).astype(obs.dtype)

class PartialObsWrapper(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        return self._partial(obs)
    def step_async(self, actions):
        self.venv.step_async(actions)
    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self._partial(obs), rews, dones, infos
    def _partial(self, obs):
        obs[..., 2:] = 0  # only keep first 2 dims (x, y)
        return obs

# === Main Loop ===
def get_q(model, obs, action):
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        q_vals = model.q_net(obs_tensor)
        return q_vals[0, action].item()

results = []

for name, use_po, noise_std in SCENARIOS:
    print(f"\n== Scenario: {name} ==")

    # Load hyperparams
    hyperparams, _ = get_saved_hyperparams(STATS_PATH)

    # Create env
    env = create_test_env(
        env_id=ENV_ID,
        n_envs=1,
        stats_path=STATS_PATH,
        seed=SEED,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
    )
    if use_po:
        env = PartialObsWrapper(env)
    if noise_std > 0:
        env = VecObsNoise(env, noise_std)

    # Load model
    ModelCls = {"dqn": DQN, "qr_dqn": QRDQN, "bootstrapped_dqn": DQN}[ALGO]
    model = ModelCls.load(
        MODEL_PATH,
        env=env,
        device=DEVICE,
        buffer_size=1,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
        },
    )
    GAMMA = model.gamma

    # Collect (Q, G)
    q_list, g_list = [], []

    for ep in range(EVAL_EPISODES):
        obs = env.reset()
        done = [False]
        rewards, q_seq = [], []
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            q_seq.append(get_q(model.policy, obs, action[0]))
            obs, reward, done, _ = env.step(action)
            rewards.append(reward[0])
        # MC return
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        q_list.extend(q_seq)
        g_list.extend(returns)
        print(f"Episode {ep+1}: {len(q_seq)} steps")
    env.close()

    # Save and evaluate
    df = pd.DataFrame({"Q_value": q_list, "MC_return": g_list})
    os.makedirs("results/dqns", exist_ok=True)
    df.to_csv(f"results/dqns/{ALGO}_{name}.csv", index=False)
    ece = compute_ece(df["Q_value"], df["MC_return"], n_bins=10)
    print(f"{name:10s} | steps: {len(df)} | ECE: {ece:.4f}")
    results.append((name, len(df), ece))

# Final summary
summary = pd.DataFrame(results, columns=["scenario", "n_steps", "ECE"])
print("\n=== Final Summary ===")
print(summary.to_string(index=False))
