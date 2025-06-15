import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# 构建环境：使用图像观测 + human 渲染
def make_env():
    env = gym.make("MiniGrid-DoorKey-8x8", render_mode="human")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)  # 提取图像作为 obs
    return env

# 包装为 VecEnv
env = DummyVecEnv([make_env])

# 创建模型
model = DQN(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    tau=1.0,
    exploration_fraction=0.1,
    tensorboard_log="./logs/doorkey_dqn/"
)

# 开始训练
model.learn(total_timesteps=50000)
model.save("doorkey_dqn_baseline")

# 评估性能（不渲染）
print("\nEvaluating agent...")
eval_env = DummyVecEnv([lambda: make_env()])
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# 运行一局演示（带渲染）
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)