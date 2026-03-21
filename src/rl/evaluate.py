"""Policy evaluation and threshold extraction for RL agents.

Provides:
- ``evaluate_policy``: Run a policy through the environment and collect metrics.
- ``extract_threshold_surface``: Extract the implied decision boundary from
  a DQN's Q-values for comparison with analytical thresholds.
- ``plot_threshold_comparison``: Side-by-side visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Predictable(Protocol):
    """Minimal interface for policies (SB3 models or analytical baselines)."""

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]: ...


def evaluate_policy(
    env: Any,
    policy: Predictable,
    n_episodes: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate a policy over multiple episodes and return aggregate metrics.

    Args:
        env: A Gymnasium-compatible environment (e.g., BlobMarketEnv).
        policy: A policy with a ``predict(obs, deterministic)`` method.
        n_episodes: Number of episodes to run.
        seed: Base seed for episode resets.

    Returns:
        Dict with keys: ``mean_reward``, ``std_reward``,
        ``mean_episode_length``, ``mean_post_frequency``.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_post_counts: list[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        posts = 0
        done = False

        while not done:
            action_arr, _ = policy.predict(obs, deterministic=True)
            action = int(action_arr[0]) if hasattr(action_arr, '__len__') else int(action_arr)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if action > 0:
                posts += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_post_counts.append(posts)

    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    post_counts = np.array(episode_post_counts)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_post_frequency": float(
            np.mean(post_counts / np.maximum(lengths, 1))
        ),
    }


def extract_threshold_surface(
    model: Any,
    queue_range: np.ndarray,
    price_range: np.ndarray,
    blob_fee: float = 1.0,
    action_mode: str = "binary",
) -> np.ndarray:
    """Extract the implied decision boundary from a DQN's Q-values.

    For each (queue, price) pair, compute Q(s, wait) vs Q(s, post) and
    record the price at which the agent switches from waiting to posting.

    Args:
        model: An SB3 DQN model with a ``q_net`` attribute.
        queue_range: 1D array of queue/waiting-time values to evaluate.
        price_range: 1D array of price values to evaluate.
        blob_fee: Fixed blob fee for the observation (held constant).
        action_mode: ``"binary"`` or ``"discrete_blobs"``.

    Returns:
        2D array of shape (len(queue_range), len(price_range)) where
        entry [i, j] = 1 if the agent posts at (queue_range[i],
        price_range[j]), 0 otherwise.
    """
    import torch

    surface = np.zeros((len(queue_range), len(price_range)), dtype=np.float32)

    for i, q in enumerate(queue_range):
        for j, p in enumerate(price_range):
            obs = np.array([q, blob_fee, p], dtype=np.float32)
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                q_values = model.q_net(obs_tensor)
            action = int(q_values.argmax(dim=1).item())
            surface[i, j] = 1.0 if action > 0 else 0.0

    return surface


def plot_threshold_comparison(
    learned_surface: np.ndarray,
    analytical_threshold: np.ndarray,
    queue_range: np.ndarray,
    price_range: np.ndarray,
    save_path: Path | str,
) -> None:
    """Plot learned vs analytical threshold surfaces side by side.

    Args:
        learned_surface: 2D array from ``extract_threshold_surface``.
        analytical_threshold: 1D array of threshold prices for each queue value.
        queue_range: Queue/waiting-time values (y-axis).
        price_range: Price values (x-axis).
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Learned threshold
    axes[0].pcolormesh(
        price_range, queue_range, learned_surface, cmap="RdYlGn_r", shading="auto"
    )
    axes[0].set_xlabel("Gas Price")
    axes[0].set_ylabel("Queue / Waiting Time")
    axes[0].set_title("Learned Policy (DQN)")

    # Analytical threshold
    axes[1].pcolormesh(
        price_range, queue_range, learned_surface, cmap="RdYlGn_r",
        shading="auto", alpha=0.3,
    )
    axes[1].plot(analytical_threshold, queue_range, "k--", linewidth=2,
                 label="Analytical Threshold")
    axes[1].set_xlabel("Gas Price")
    axes[1].set_title("Analytical vs Learned")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
