"""Bootstrapped DQN — a clean, single‑file implementation fully compatible with
Stable‑Baselines3 ≥ 2.6 **and** rl‑baselines3‑zoo.

Highlights
==========
* **Multi‑head Q‑network** (shared trunk + H linear heads).
* **ReplayBuffer with bootstrap masks** — mask & data always use **identical indices**.
* **Head‑switch callback** — random head per env at rollout start.
* **Correct ε‑greedy** in `_sample_action()` so each env follows its current head.
* **Vectorised loss** — one pass over heads, no Python loops.

Usage (inside rl‑baselines3‑zoo)
--------------------------------
```bash
python -m rl_zoo3.train --algo bootstrapped_dqn --env LunarLander-v3 \
  --hyperparams "n_heads:10,bootstrap_prob:0.9,buffer_size:100000,\
                 learning_rate:1e-4,target_update_interval:4000,\
                 gradient_steps:1,exploration_fraction:0.05,\
                 exploration_final_eps:0.05" --n_envs 8
```

Quick smoke‑test (stand‑alone)
------------------------------
```bash
pip install "stable-baselines3>=2.6" gymnasium[box2d] torch
python bootstrapped_dqn.py --timesteps 5e4
```
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy

# -----------------------------------------------------------------------------
# Typing helpers
# -----------------------------------------------------------------------------
Schedule = Callable[[float], float]


class BootstrappedSamples(NamedTuple):
    """Extension of ReplayBufferSamples with bootstrap mask."""

    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    mask: torch.Tensor  # [batch, n_heads]


# -----------------------------------------------------------------------------
# ReplayBuffer with aligned bootstrap masks
# -----------------------------------------------------------------------------
class BootstrapMaskBuffer(ReplayBuffer):
    """ReplayBuffer that stores a Bernoulli mask for every transition."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: Union[torch.device, str] = "auto",
        *,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_heads: int = 10,
        bootstrap_prob: float = 0.5,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.n_heads = n_heads
        self.bootstrap_prob = float(bootstrap_prob)
        self.masks = np.zeros((buffer_size, n_heads), dtype=np.bool_)

    # ---------------------- helpers ----------------------
    def _sample_indices(self, batch_size: int) -> np.ndarray:  # pylint: disable=arguments-differ
        upper = self.buffer_size if self.full else self.pos
        return np.random.randint(0, upper, size=batch_size)

    # ---------------------- API --------------------------
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:  # pylint: disable=invalid-name
        # Ensure 2D input
        obs = np.atleast_2d(obs)
        next_obs = np.atleast_2d(next_obs)
        action = np.atleast_2d(action)
        reward = np.atleast_2d(reward)
        done = np.atleast_2d(done)
        n_envs = obs.shape[0]
        
        # Generate masks for each env
        new_masks = np.random.binomial(1, self.bootstrap_prob, size=(n_envs, self.n_heads))
        
        # Ensure each transition has at least one active mask
        zero_mask_rows = ~new_masks.any(axis=1)
        if zero_mask_rows.any():
            random_heads = np.random.randint(0, self.n_heads, size=zero_mask_rows.sum())
            new_masks[zero_mask_rows, random_heads] = 1
        
        # Add to buffer using parent's method
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Store masks using correct indexing
        indices = (self.pos - n_envs + np.arange(n_envs)) % self.buffer_size
        self.masks[indices] = new_masks

    # mypy: override
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> BootstrappedSamples:  # type: ignore[override]
        # Use parent's sampling logic
        indices = self._sample_indices(batch_size)
        data = self._get_samples(indices, env=env)
        # Convert mask to tensor immediately
        masks = torch.as_tensor(self.masks[indices], device=self.device)
        
        # Verify mask alignment
        assert (masks == 1).any(dim=1).all(), "Each sample must have at least one active mask"
        
        return BootstrappedSamples(
            observations=data.observations,
            next_observations=data.next_observations,
            actions=data.actions,
            rewards=data.rewards,
            dones=data.dones,
            mask=masks,  # Now a tensor
        )


# -----------------------------------------------------------------------------
# Network & Policy
# -----------------------------------------------------------------------------
class MultiHeadQNet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        n_heads: int,
        n_envs: int = 1,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.action_dim = action_dim
        self.n_envs = n_envs
        
        # Create shared MLP trunk properly
        if net_arch is None:
            net_arch = [256, 256]
            
        # Build trunk layers
        trunk_layers = []
        last_dim = observation_dim
        for layer_size in net_arch[:-1]:
            trunk_layers.append(nn.Linear(last_dim, layer_size))
            trunk_layers.append(activation_fn())
            last_dim = layer_size
        # Add final layer
        trunk_layers.append(nn.Linear(last_dim, net_arch[-1]))
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Create independent linear heads
        self.heads = nn.ModuleList([
            nn.Linear(net_arch[-1], action_dim)
            for _ in range(n_heads)
        ])
        
        # Register current heads buffer
        self.register_buffer(
            "_current_heads",
            torch.zeros(n_envs, dtype=torch.long)
        )
        # Initialize randomly
        self._current_heads.random_(0, n_heads)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Get shared features [batch_size, trunk_out_dim]
        features = self.trunk(obs)
        
        # Get Q-values for each head [batch_size, n_heads, n_actions]
        # TODO: Consider using einsum for better performance with large n_heads
        q_values = torch.stack([
            head(features) for head in self.heads
        ], dim=1)
        
        return q_values
    
    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get the action from an observation (in training mode using the policy).
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            Selected actions
        """
        # Get Q-values for all heads [batch_size, n_heads, n_actions]
        q_values = self(obs)
        
        # Select Q-values for current heads [batch_size, n_actions]
        current_q_values = q_values[torch.arange(len(self._current_heads), device=obs.device), self._current_heads]
        
        if deterministic:
            # Take best action for each environment
            actions = current_q_values.argmax(dim=1)
        else:
            # Per-environment ε-greedy
            random_mask = torch.rand(len(self._current_heads), device=obs.device) < getattr(self, "exploration_rate", 0.0)
            actions = torch.empty(len(self._current_heads), dtype=torch.long, device=obs.device)
            
            # Greedy actions for non-random environments
            if (~random_mask).any():
                actions[~random_mask] = current_q_values[~random_mask].argmax(dim=1)
            
            # Random actions for random environments
            if random_mask.any():
                actions[random_mask] = torch.randint(
                    0, self.action_dim, 
                    size=(random_mask.sum(),), 
                    device=obs.device
                )
        
        return actions
    
    def set_training_mode(self, mode: bool) -> None:
        """Set the training mode for all submodules.
        
        Required by SB3's DQN policy for target network management.
        """
        self.train(mode)
    
    def switch_heads(self) -> None:
        """Randomly select new heads for each environment."""
        self._current_heads.random_(0, self.n_heads)


class MultiHeadQPolicy(DQNPolicy):
    """Policy wrapper that plugs the multi‑head Q‑network into SB3."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        *,
        n_heads: int = 10,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.n_heads = n_heads
        if net_arch is None:
            net_arch = [256, 256]
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs or {},
            **kwargs,
        )

    def make_q_net(self) -> nn.Module:
        # Create feature extractor first
        self.features_extractor = self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        # Get n_envs from the environment if available
        n_envs = getattr(self.env, "num_envs", 1) if hasattr(self, "env") else 1
        # Create Q network with feature extractor's output dimension
        return MultiHeadQNet(
            observation_dim=self.features_extractor.features_dim,
            action_dim=self.action_space.n,
            n_heads=self.n_heads,
            n_envs=n_envs,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # More precise dimension check for single environment
        if (obs.ndim == len(self.observation_space.shape) and 
            getattr(self, "n_envs", 1) == 1):
            obs = obs.unsqueeze(0)
        
        # Extract features first
        features = self.features_extractor(obs)
        
        # Get Q-values for all heads [batch_size, n_heads, n_actions]
        q_values = self.q_net(features)
        
        # Get current head indices for each environment
        current_heads = self.q_net._current_heads  # Access from q_net
        
        # Select Q-values for current heads [batch_size, n_actions]
        current_q_values = q_values[torch.arange(len(current_heads)), current_heads]
        
        if deterministic:
            # Take best action for each environment
            actions = current_q_values.argmax(dim=1)
        else:
            # Per-environment ε-greedy
            random_mask = torch.rand(len(current_heads), device=self.device) < self.exploration_rate
            actions = torch.empty(len(current_heads), dtype=torch.long, device=self.device)
            
            # Greedy actions for non-random environments
            if (~random_mask).any():
                actions[~random_mask] = current_q_values[~random_mask].argmax(dim=1)
            
            # Random actions for random environments
            if random_mask.any():
                actions[random_mask] = torch.randint(
                    0, self.action_space.n, 
                    size=(random_mask.sum(),), 
                    device=self.device
                )
        
        return actions, actions  # Return same actions for buffer


# -----------------------------------------------------------------------------
# Callback — switch heads every rollout
# -----------------------------------------------------------------------------
class HeadSwitchCallback(BaseCallback):
    """Randomly selects a new head for every env at rollout start."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def _on_rollout_start(self) -> None:
        # Only switch heads for online network
        self.model.policy.q_net.switch_heads()
        # Remove target network head switching

    def _on_step(self) -> bool:
        """Required by BaseCallback. Returns True to continue training."""
        return True


# -----------------------------------------------------------------------------
# Bootstrapped DQN algorithm
# -----------------------------------------------------------------------------
class BootstrappedDQN(DQN):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MultiHeadQPolicy,
        "CnnPolicy": MultiHeadQPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        n_heads: int = 10,
        bootstrap_prob: float = 0.65,
        learning_rate: Union[float, Schedule] = 6.3e-4,
        buffer_size: int = 50000,
        learning_starts: int = 0,
        batch_size: int = 128,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = -1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.12,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        # Store parameters before super().__init__
        self.n_heads = n_heads
        self.bootstrap_prob = bootstrap_prob

        # --- 修正 policy_kwargs ---
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs = dict(policy_kwargs)  # 避免引用传递
        policy_kwargs["n_heads"] = n_heads

        # Initialize parent class with _init_setup_model=False
        super().__init__(
            policy=policy,
            env=env,
            _init_setup_model=False,  # Delay model setup
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
        )
        
        # Create head switch callback
        self.head_switch_cb = HeadSwitchCallback(n_heads=self.n_heads)
        
        # Initialize callbacks list
        self.callbacks = []
        
        # Now setup the model
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        super()._setup_model()
        
        # Create bootstrap mask buffer
        self.replay_buffer = BootstrapMaskBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            n_heads=self.n_heads,
            bootstrap_prob=self.bootstrap_prob,
            handle_timeout_termination=False,
        )
        
        # Initialize current heads in policy
        self.policy.q_net._current_heads = torch.zeros(
            self.n_envs, dtype=torch.long, device=self.device
        ).random_(0, self.n_heads)
        self.policy.q_net_target._current_heads = self.policy.q_net._current_heads.clone()
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BootstrappedDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "BootstrappedDQN":
        # Ensure head switch callback is in the list
        if self.head_switch_cb not in self.callbacks:
            self.callbacks.append(self.head_switch_cb)
        
        # Sync exploration rate before training
        self.policy.exploration_rate = self.exploration_rate
        
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    # ---------------- ε‑greedy sampler ----------------
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[np.ndarray] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:  # noqa: D401
        if self.num_timesteps < learning_starts:
            actions = np.array([self.action_space.sample() for _ in range(n_envs)])
            return actions, actions

        random_mask = np.random.rand(n_envs) < self.exploration_rate
        actions = np.empty(n_envs, dtype=int)
        
        # Handle all-random case first
        if random_mask.all():
            actions = np.random.randint(0, self.action_space.n, size=n_envs)
            return actions, actions

        # ---------- greedy branch ----------
        if (~random_mask).any():
            # Get observations for non-random actions
            obs = torch.as_tensor(self._last_obs[~random_mask]).to(self.device).float()
            
            # More robust dimension check
            if (obs.ndim == len(self.observation_space.shape) and 
                obs.shape[0] != (~random_mask).sum()):
                obs = obs.unsqueeze(0)
            
            q = self.policy.q_net(obs)                                   # [n_sel, H, A]
            heads = self.policy.q_net._current_heads[~random_mask]       # Use policy's heads
            q_sel = q[torch.arange(q.size(0), device=self.device), heads]
            actions[~random_mask] = q_sel.argmax(dim=1).cpu().numpy()

        # ---------- random branch ----------
        if random_mask.any():
            # Vectorized random action generation
            actions[random_mask] = np.random.randint(
                0, self.action_space.n, size=random_mask.sum()
            )

        return actions, actions

    # ---------------- core training loop ----------------
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # noqa: D401
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: List[float] = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            mask = data.mask.float()                                     # [B, H]
            
            # Skip step if no active masks (numerical stability)
            if mask.sum() == 0:
                continue

            with torch.no_grad():
                next_q_online = self.policy.q_net(data.next_observations)         # [B, H, A]
                next_act = next_q_online.argmax(dim=2, keepdim=True)              # [B, H, 1]
                next_q_target = self.policy.q_net_target(data.next_observations)  # [B, H, A]
                next_q = torch.gather(next_q_target, 2, next_act).squeeze(2)      # [B, H]
                rewards = data.rewards.view(-1, 1)                                # [B, 1]
                dones = data.dones.view(-1, 1)                                    # [B, 1]
                target_q = rewards + (1 - dones) * self.gamma * next_q            # [B, H]

            # current Q
            cur_q_all = self.policy.q_net(data.observations)                      # [B, H, A]
            actions = data.actions.view(-1, 1, 1).expand(-1, self.n_heads, 1)     # [B, H, 1]
            cur_q = torch.gather(cur_q_all, 2, actions).squeeze(2)                # [B, H]

            # Ensure dimensions match
            assert cur_q.shape == target_q.shape, f"Shape mismatch: cur_q {cur_q.shape} vs target_q {target_q.shape}"
            td_err = F.smooth_l1_loss(cur_q, target_q, reduction="none")         # [B, H]
            loss = (td_err * mask).sum() / mask.sum().clamp_min(1.0)

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            losses.append(loss.item())
            self._n_updates += 1

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    self.policy.q_net.parameters(),
                    self.policy.q_net_target.parameters(),
                    self.tau,
                )
                # Sync current heads to target network
                self.policy.q_net_target._current_heads = self.policy.q_net._current_heads.clone()

        if losses:  # Only record if we had any valid steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # Sync exploration rate before prediction
        self.policy.exploration_rate = self.exploration_rate
        return super().predict(
            observation=observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    # ---------------- save / load helpers ----------------
    def _excluded_save_params(self) -> List[str]:
        """Return a list of parameters that should not be saved."""
        excluded = super()._excluded_save_params()
        excluded.extend([
            "_current_heads",  # Exclude current head indices
            "head_switch_cb",  # Exclude callback object with thread locks
            "policy.features_extractor",  # Exclude feature extractor (will be recreated)
        ])
        return excluded

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """Return the parameters to save."""
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []


# -----------------------------------------------------------------------------
# rl‑baselines3‑zoo registration
# -----------------------------------------------------------------------------
ALGOS: Dict[str, Any] = {"bootstrapped_dqn": BootstrappedDQN}

# -----------------------------------------------------------------------------
# Local smoke‑test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000, help="total env steps")
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")
    model = BootstrappedDQN(
        "MlpPolicy",
        env,
        n_heads=10,
        bootstrap_prob=0.65,
        learning_rate=6.3e-4,
        buffer_size=50_000,
        gradient_steps=-1,
        batch_size=128,
        exploration_fraction=0.12,
        exploration_final_eps=0.1,
        target_update_interval=250,
        verbose=1,        
    )
    model.learn(total_timesteps=args.timesteps)
    save_path = Path("bootdqn_lunar.zip")
    model.save(save_path)
    print(f"Training finished → model saved to {save_path.resolve()}")
    env.close() 