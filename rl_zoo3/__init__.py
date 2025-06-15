"""RL Zoo3: A training framework for Stable-Baselines3."""

import os

# isort: off
import rl_zoo3.gym_patches  # noqa: F401
# isort: on

# Import algorithms first to avoid circular imports
from rl_zoo3.algos import ALGOS

# Then import utilities
from rl_zoo3.utils import (
    create_test_env,
    get_latest_run_id,
    get_saved_hyperparams,
    get_trained_models,
    get_wrapper_class,
    linear_schedule,
)

# Finally import main functionality
from rl_zoo3.train import train
from rl_zoo3.enjoy import enjoy
from rl_zoo3.load_from_hub import load_from_hub
from rl_zoo3.exp_manager import ExperimentManager

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
