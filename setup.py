import os
import shutil

from setuptools import setup, find_packages

with open(os.path.join("rl_zoo3", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

# Copy hyperparams files for packaging
shutil.copytree("hyperparams", os.path.join("rl_zoo3", "hyperparams"))

long_description = """
# RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

See https://github.com/DLR-RM/rl-baselines3-zoo
"""
install_requires = [
    "sb3_contrib>=2.6.0,<3.0",
    "gymnasium>=0.29.1,<1.2.0",
    "huggingface_sb3>=3.0,<4.0",
    "tqdm",
    "rich",
    "optuna>=3.0",
    "pyyaml>=5.1",
    "pytablewriter~=1.2",
    "shimmy~=2.0",
]
plots_requires = ["seaborn", "rliable~=1.2.0", "scipy~=1.10"]
test_requires = [
    # for MuJoCo envs v4:
    "mujoco>=2.3,<4",
    # install parking-env to test HER
    "highway-env>=1.10.1,<1.11.0",
]

setup(
    name="rl-baselines3-zoo",
    version="2.0.0",
    description="A training framework for Stable-Baselines3 with Bootstrapped DQN integration",
    author="RL Zoo Team",
    author_email="",
    url="https://github.com/DLR-RM/rl-baselines3-zoo",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy>=1.21.0",
        "torch>=1.11.0",
        "stable-baselines3>=2.0.0",
        "pyyaml>=5.1.2",
        "tensorboard>=2.9.1",
        "rich>=12.0.0",
        "opencv-python>=4.5.5.64",
        "matplotlib>=3.5.1",
        "pandas>=1.4.2",
        "seaborn>=0.11.2",
        "cloudpickle>=2.0.1",
        "optuna>=2.10.0",
        "sb3-contrib>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# Remove copied files after packaging
shutil.rmtree(os.path.join("rl_zoo3", "hyperparams"))
