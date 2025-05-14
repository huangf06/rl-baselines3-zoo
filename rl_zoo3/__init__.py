"""RL Zoo3: A training framework for Stable-Baselines3."""

import os

# isort: off

import rl_zoo3.gym_patches  # noqa: F401

# isort: on

from rl_zoo3.utils import (
    ALGOS,
    create_test_env,
    get_latest_run_id,
    get_saved_hyperparams,
    get_trained_models,
    get_wrapper_class,
    linear_schedule,
)

from rl_zoo3.train import train
from rl_zoo3.enjoy import enjoy
from rl_zoo3.load_from_hub import load_from_hub
from rl_zoo3.exp_manager import ExperimentManager

# Register Bootstrapped DQN
from rl_zoo3.bootstrapped_dqn import BootstrappedDQN
ALGOS["bootstrapped_dqn"] = BootstrappedDQN

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "ALGOS",
    "create_test_env",
    "get_latest_run_id",
    "get_saved_hyperparams",
    "get_trained_models",
    "get_wrapper_class",
    "linear_schedule",
]
