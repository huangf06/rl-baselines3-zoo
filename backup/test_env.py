import torch
import gymnasium as gym
import ale_py
from stable_baselines3.dqn import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())

# Test Gym Atari env
try:
    env = make_atari_env("SeaquestNoFrameskip-v4", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()
    print("✅ Gymnasium Atari environment created successfully.")
except Exception as e:
    print("❌ Failed to create environment:", e)

# Test basic model creation
try:
    model = DQN("CnnPolicy", env, verbose=0)
    print("✅ DQN model created successfully.")
except Exception as e:
    print("❌ Failed to create DQN model:", e)

env.close()

