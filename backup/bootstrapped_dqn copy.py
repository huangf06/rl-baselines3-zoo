"""Bootstrapped DQN implementation for rl-baselines3-zoo."""

from typing import Any, Dict, List, Optional, Type, Union, Tuple, Callable, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, ReplayBufferSamples
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize

# Define Schedule type
Schedule = Callable[[float], float]

class BootstrappedSamples(NamedTuple):
    """Extended ReplayBufferSamples that includes bootstrap mask and indices."""
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    mask: torch.Tensor  # Shape: [batch, n_heads]
    idx: torch.Tensor   # Shape: [batch]

class BootstrapMaskBuffer(ReplayBuffer):
    """Wrapper around ReplayBuffer that adds bootstrap mask functionality."""
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_heads: int = 10,
        bootstrap_prob: float = 0.5,
    ):
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
        self.bootstrap_prob = bootstrap_prob
        # Initialize empty mask buffer
        self.mask_buf = np.zeros((buffer_size, n_heads), dtype=bool)
    
    def _generate_masks(self, batch_size: int) -> torch.Tensor:
        """Generate bootstrap masks for a batch.
        
        Args:
            batch_size: Number of masks to generate
            
        Returns:
            Tensor of shape [batch_size, n_heads] containing boolean masks
        """
        indices = self._sample_indices(batch_size)
        return torch.as_tensor(self.mask_buf[indices], device=self.device)
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Add a new transition to the buffer."""
        # Generate bootstrap mask for the current position
        mask = np.random.rand(self.n_heads) < self.bootstrap_prob
        # Ensure at least one head is active
        if not mask.any():
            mask[np.random.randint(self.n_heads)] = True
        # Ensure not all heads are active
        if mask.all():
            mask[np.random.randint(self.n_heads)] = False
        self.mask_buf[self.pos] = mask
        
        # Convert infos to list of dicts if needed
        if isinstance(infos, tuple):
            infos = [{"TimeLimit.truncated": False} for _ in range(len(infos))]
        elif not isinstance(infos, list):
            infos = [{"TimeLimit.truncated": False}]
        elif len(infos) == 0:
            infos = [{"TimeLimit.truncated": False}]
        
        super().add(obs, next_obs, action, reward, done, infos)

    def reset(self):
        """Reset the buffer.
        
        Note: The mask buffer is cleared on reset to avoid sampling stale masks.
        """
        super().reset()
        self.mask_buf[:] = False

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """Sample indices from the buffer."""
        upper_bound = self.buffer_size if self.full else self.pos
        return np.random.randint(0, upper_bound, size=batch_size)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> BootstrappedSamples:
        """Sample a batch of experiences with bootstrap masks.
        
        Args:
            batch_size: Number of transitions to sample
            env: Optional VecNormalize wrapper for observation normalization
            
        Returns:
            BootstrappedSamples with mask and indices
        """
        # Get samples from parent class
        samples = super().sample(batch_size, env)
        
        # Generate bootstrap masks
        mask = self._generate_masks(batch_size)
        
        # Add masks to samples
        return BootstrappedSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            mask=mask,
            idx=torch.arange(batch_size, device=mask.device)
        )

class MultiHeadNetwork(nn.Module):
    """Q-Network with multiple heads and dueling architecture."""
    def __init__(
        self,
        features_extractor: nn.Module,
        shared_mlp: nn.Module,
        q_heads: nn.ModuleList,
        dueling: bool = True,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.shared_mlp = shared_mlp
        self.q_heads = q_heads
        self.dueling = dueling
        
        if dueling:
            # Find the last Linear layer in shared_mlp to get the correct output dim
            shared_dim = None
            if isinstance(shared_mlp, nn.Identity):
                shared_dim = features_extractor.features_dim
            elif isinstance(shared_mlp, nn.Sequential):
                for layer in reversed(shared_mlp):
                    if isinstance(layer, nn.Linear):
                        shared_dim = layer.out_features
                        break
            if shared_dim is None:
                raise RuntimeError("Could not determine shared_mlp output dimension for dueling heads.")
            self.value_heads = nn.ModuleList([
                nn.Linear(shared_dim, 1)
                for _ in range(len(q_heads))
            ])
            self.advantage_heads = nn.ModuleList([
                nn.Linear(shared_dim, q_heads[0].out_features)
                for _ in range(len(q_heads))
            ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if obs.dtype != torch.float32:
            obs = obs.float()
        
        if obs.ndim == 3 or obs.ndim == 1:
            obs = obs.unsqueeze(0)
            
        # Only normalize if input is actually an image (4D tensor with channels)
        if obs.ndim == 4:
            obs = obs / 255.0
            
        features = self.features_extractor(obs)
        shared_features = self.shared_mlp(features)
        
        if self.dueling:
            # Dueling DQN architecture
            q_values = []
            for i in range(len(self.q_heads)):
                value = self.value_heads[i](shared_features)
                advantage = self.advantage_heads[i](shared_features)
                # Combine value and advantage streams
                q_values.append(value + advantage - advantage.mean(dim=1, keepdim=True))
            return torch.stack(q_values, dim=1)
        else:
            # Standard DQN architecture
            return torch.stack([head(shared_features) for head in self.q_heads], dim=1)

    def set_training_mode(self, mode: bool) -> None:
        """Set the training mode for all submodules."""
        self.train(mode)
        self.features_extractor.train(mode)
        self.shared_mlp.train(mode)
        for head in self.q_heads:
            head.train(mode)

class MultiHeadQPolicy(DQNPolicy):
    """Policy class for Bootstrapped DQN with multiple heads."""
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        n_heads: int = 10,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.n_heads = n_heads
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self) -> nn.Module:
        """Create the Q-network with multiple heads."""
        # Create shared feature extractor
        features_extractor = self.features_extractor_class(
            self.observation_space,
            **self.features_extractor_kwargs
        )
        features_dim = features_extractor.features_dim
        
        # Create shared MLP if needed
        if not self.net_arch:
            shared_mlp = nn.Identity()
            shared_dim = features_dim
        else:
            layers = []
            last_layer_dim = features_dim
            for layer_size in self.net_arch:
                layers.append(nn.Linear(last_layer_dim, layer_size))
                layers.append(self.activation_fn())
                last_layer_dim = layer_size
            shared_mlp = nn.Sequential(*layers)
            shared_dim = self.net_arch[-1]
        
        # Create multiple heads
        q_heads = nn.ModuleList([
            nn.Linear(shared_dim, self.action_space.n)
            for _ in range(self.n_heads)
        ])
        
        return MultiHeadNetwork(features_extractor, shared_mlp, q_heads)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.q_net(obs)

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get the action from an observation (in training mode using the policy).

        :param obs: Observation
        :param deterministic: Whether to use deterministic actions
        :return: Taken action according to the policy
        """
        q_values = self.forward(obs)  # [batch, n_heads, n_actions]
        # Select the current active head per environment
        if hasattr(self, "_last_head"):
            batch_indices = torch.arange(obs.shape[0], device=obs.device)
            q_values = q_values[batch_indices, self._last_head]  # [batch, n_actions]
        else:
            q_values = q_values[:, 0]  # fallback to first head
        # Apply epsilon-greedy exploration
        exploration_rate = getattr(self, "exploration_rate", 0.0)
        if not deterministic and np.random.rand() < exploration_rate:
            actions = torch.randint(0, self.action_space.n, (obs.shape[0],), device=obs.device)
        else:
            actions = q_values.argmax(dim=-1)
        return actions

class BootstrappedDQN(DQN):
    """
    Bootstrapped DQN (Osband et al., 2016)
    Paper: https://arxiv.org/abs/1602.04621
    
    This implementation extends DQN with bootstrapped uncertainty estimation
    using multiple Q-heads and bootstrap masking. Each head is trained on a
    different subset of the replay buffer, allowing for better exploration
    and uncertainty estimation.
    """
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MultiHeadQPolicy,
        "CnnPolicy": MultiHeadQPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[MultiHeadQPolicy]],
        env: Union[GymEnv, str],
        n_heads: int = 10,
        bootstrap_prob: float = 0.5,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Set these before calling parent's __init__
        self.n_heads = n_heads
        self.bootstrap_prob = bootstrap_prob
        self._last_head = None  # Will be initialized in _setup_model

        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["n_heads"] = n_heads

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
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
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        """Initialize Q-networks (+ noisy target), bootstrap buffer, callbacks, and logger.
        
        This method:
        1. Initializes the Q-network and target network with different parameters
        2. Sets up the replay buffer with bootstrap masking
        3. Creates the head switching callback
        4. Configures the logger and exploration parameters
        
        The target network is initialized with small random noise to break symmetry
        and encourage diverse exploration across heads.
        """
        # Set replay buffer class and kwargs before parent initialization
        if self.replay_buffer_class is None:
            self.replay_buffer_class = BootstrapMaskBuffer

        if self.replay_buffer_kwargs is None:
            self.replay_buffer_kwargs = {}
        # Add bootstrap parameters to kwargs if using BootstrapMaskBuffer
        if issubclass(self.replay_buffer_class, BootstrapMaskBuffer):
            self.replay_buffer_kwargs.update({
                "n_heads": self.n_heads,
                "bootstrap_prob": self.bootstrap_prob,
            })

        super()._setup_model()

        # Create target network
        self.policy.q_net_target = self.policy.make_q_net().to(self.device)
        # Initialize target network with different parameters to break symmetry
        for p, tp in zip(self.policy.q_net.parameters(), self.policy.q_net_target.parameters()):
            tp.data.copy_(p.data + torch.randn_like(p.data) * 0.1)

        # Create callbacks
        self.callbacks = []
        self.callbacks.append(HeadSwitchCallback(self.n_heads, self))

        # Initialize exploration rate
        self.exploration_rate = self.exploration_initial_eps
        
        # Initialize logger
        self._logger = configure_logger(folder=None, format_strings=["stdout"])

        # Initialize last head for each environment
        self._last_head = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        # Sync with policy
        self.policy._last_head = self._last_head.clone()

    def _on_step(self) -> bool:
        """
        Update the exploration rate and target network if needed.
        """
        # Update exploration rate
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("train/exploration_rate", self.exploration_rate)
        
        # Update target network
        polyak_update(
            self.q_net.parameters(),
            self.q_net_target.parameters(),
            self.tau
        )
        return True

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Perform a training step with Double DQN."""
        self.policy.train()
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                # Double DQN: use online network to select actions
                next_q_values = self.policy.q_net(replay_data.next_observations)  # [batch, n_heads, n_actions]
                next_actions = next_q_values.argmax(dim=2)  # [batch, n_heads]
                
                # Use target network to evaluate actions
                next_q_values_target = self.policy.q_net_target(replay_data.next_observations)  # [batch, n_heads, n_actions]
                next_q_values = next_q_values_target.gather(2, next_actions.unsqueeze(2)).squeeze(2)  # [batch, n_heads]
                
                # Expand rewards and dones to match next_q_values shape
                rewards = replay_data.rewards.view(-1, 1).expand(-1, self.n_heads)  # [batch, n_heads]
                dones = replay_data.dones.view(-1, 1).expand(-1, self.n_heads)  # [batch, n_heads]
                target_q = rewards + (1 - dones) * self.gamma * next_q_values  # [batch, n_heads]
            
            current_q = self.policy.q_net(replay_data.observations)  # [batch, n_heads, n_actions]
            actions = replay_data.actions.view(-1, 1)  # [batch, 1]
            head_losses = []
            
            for head_idx in range(self.n_heads):
                active_mask = replay_data.mask[:, head_idx]  # [batch]
                if active_mask.any():
                    head_q = current_q[:, head_idx].gather(1, actions).squeeze(1)[active_mask]  # [active_batch]
                    head_target = target_q[:, head_idx][active_mask]  # [active_batch]
                    loss = F.smooth_l1_loss(head_q, head_target, reduction="mean")
                    head_losses.append(loss)
            
            if not head_losses:
                self._n_updates += 1
                self._logger.record("train/skipped_step", 1, exclude="tensorboard")
                continue
                
            loss = torch.stack(head_losses).mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            self._n_updates += 1
            if self._n_updates % self.target_update_interval == 0:
                with torch.no_grad():
                    for p, tp in zip(self.policy.q_net.parameters(), self.policy.q_net_target.parameters()):
                        tp.data.mul_(1 - self.tau)
                        tp.data.add_(p.data * self.tau)
            
            losses.append(loss.item())
        
        self._logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._logger.record("train/loss", np.mean(losses))

    def _excluded_save_params(self) -> List[str]:
        """Returns the names of the parameters that should be excluded from being saved."""
        excluded = super()._excluded_save_params()
        excluded.extend([
            "replay_buffer",
            "logger",
            "callbacks",
            "_vec_normalize_env",
            "_last_head",
            "policy._last_head",
            "policy.exploration_rate",
        ])
        return excluded

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """Returns the parameters that should be saved in the torch model file."""
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    def set_logger(self, logger):
        """Set the logger for the model."""
        self._logger = logger

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BootstrappedDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "BootstrappedDQN":
        """
        Return a trained model.

        :param total_timesteps: The total number of samples to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for tensorboard log
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich
        :return: the trained model
        """
        # Combine user callbacks with internal callbacks
        if callback is None:
            callback = []
        elif not isinstance(callback, list):
            callback = [callback]
        
        if not hasattr(self, 'callbacks'):
            self.callbacks = []
        elif not isinstance(self.callbacks, list):
            self.callbacks = [self.callbacks]
            
        all_callbacks = callback + self.callbacks
        
        return super().learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

# Register the algorithm
ALGOS = {
    "bootstrapdqn": BootstrappedDQN,
}

# Register the policy
POLICY_ALIASES = {
    "MlpPolicy": MultiHeadQPolicy,
    "CnnPolicy": MultiHeadQPolicy,
}

def get_policy_kwargs(
    policy: str,
    policy_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get the policy kwargs for the given policy."""
    if policy_kwargs is None:
        policy_kwargs = {}

    if policy == "CnnPolicy":
        if "features_extractor_class" not in policy_kwargs:
            policy_kwargs["features_extractor_class"] = NatureCNN
        if "features_extractor_kwargs" not in policy_kwargs:
            policy_kwargs["features_extractor_kwargs"] = {
                "features_dim": 512,
                "normalize_images": True
            }
        if "net_arch" not in policy_kwargs:
            policy_kwargs["net_arch"] = []
        if "normalize_images" not in policy_kwargs:
            policy_kwargs["normalize_images"] = True

    return policy_kwargs

class HeadSwitchCallback(BaseCallback):
    """Callback to switch active head during training.
    
    This callback randomly switches the active head for each environment at the
    start of each rollout. This ensures that different heads are used for
    exploration and helps maintain diversity in the Q-value estimates.
    """
    def __init__(self, n_heads: int, model: Optional["BootstrappedDQN"] = None, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self._model = model

    def _init_callback(self) -> None:
        """Initialize callback attributes."""
        if self._model is None:
            self._model = self.model

    def _on_rollout_start(self) -> None:
        """Switch to a random head at the start of each rollout."""
        for i in range(self._model.n_envs):
            current_head = self._model._last_head[i].item()
            available_heads = list(range(self.n_heads))
            available_heads.remove(current_head)
            new_head = available_heads[np.random.randint(len(available_heads))]
            self._model._last_head[i] = torch.tensor(new_head, device=self._model.device)
            if self._model.verbose:
                print(f'[Callback] HeadSwitchCallback triggered at rollout start. Env {i}: {current_head} -> {new_head}')
        self._model.policy._last_head = self._model._last_head.clone()
        # Sync exploration rate with policy
        self._model.policy.exploration_rate = self._model.exploration_rate

    def _on_step(self) -> bool:
        """Required method for BaseCallback."""
        return True