"""Actor-critic MLP for discrete-action PPO.

One instance per rollup in the independent-PPO Phase 2 setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_init(layer: nn.Linear, std: float = 1.4142135, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init; matches CleanRL PPO defaults."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Discrete-action actor-critic with a shared trunk.

    Args:
        obs_dim: Observation dimension.
        n_actions: Number of discrete actions.
        hidden_dim: MLP hidden size.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.policy_head = _layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value) for a batch of observations."""
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).
            deterministic: If True, take argmax instead of sampling.

        Returns:
            (action, log_prob, value). action has shape (batch,).
        """
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob, entropy, and value for given (obs, action) pairs."""
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
