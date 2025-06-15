from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
)

class DropoutMLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, arch: List[int], p: float):
        layers: List[nn.Module] = []
        last = in_dim
        for h in arch:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))
            last = h
        layers.append(nn.Linear(last, out_dim))
        super().__init__(*layers)


class DropoutQNet(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_dim: int,
        feature_extractor_class: Type[BaseFeaturesExtractor],
        feature_extractor_kwargs: Dict[str, Any],
        net_arch: List[int],
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.features_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        feat_dim = self.features_extractor.features_dim
        self.mlp = DropoutMLP(feat_dim, action_dim, net_arch, dropout_p)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype != torch.float32:
            obs = obs.float()
        if obs.ndim in (1, 3):
            obs = obs.unsqueeze(0)
        if obs.ndim == 4:
            obs = obs / 255.0
        feats = self.features_extractor(obs)
        return self.mlp(feats)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(obs)
        return q_values.argmax(dim=1)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
        self.features_extractor.train(mode)
        self.mlp.train(mode)


Schedule = Callable[[float], float]


class MCDropoutQPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        dropout_p: float = 0.2,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        **policy_kwargs,
    ) -> None:
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.dropout_p = dropout_p
        self._net_arch = net_arch
        self._features_extractor_class = features_extractor_class
        self._features_extractor_kwargs = features_extractor_kwargs

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            **policy_kwargs,
        )

    def make_q_net(self) -> nn.Module:
        return DropoutQNet(
            self.observation_space,
            self.action_space.n,
            self._features_extractor_class,
            self._features_extractor_kwargs,
            self._net_arch,
            self.dropout_p,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

    def mc_forward(self, obs: torch.Tensor, n: int = 20) -> torch.Tensor:
        self.q_net.set_training_mode(True)
        with torch.no_grad():
            outs = [self.q_net(obs) for _ in range(n)]
        self.q_net.set_training_mode(False)
        return torch.stack(outs, dim=0)


class MCDropoutDQN(DQN):
    policy_aliases: Dict[str, Type[MCDropoutQPolicy]] = {
        "MlpPolicy": MCDropoutQPolicy,
        "CnnPolicy": MCDropoutQPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[MCDropoutQPolicy]],
        env: Union[GymEnv, str],
        dropout_p: float = 0.2,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        **kwargs,
    ) -> None:
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.update({"dropout_p": dropout_p})
        super().__init__(
            policy=policy,
            env=env,
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
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            **kwargs,
        )

    def predict_mc(self, obs: Union[torch.Tensor, Any], n: int = 20) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.policy.mc_forward(obs, n)


ALGOS = {
    "mcdropoutdqn": MCDropoutDQN,
}

POLICY_ALIASES = {
    "MlpPolicy": MCDropoutQPolicy,
    "CnnPolicy": MCDropoutQPolicy,
}