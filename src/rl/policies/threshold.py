"""Analytical threshold policies for benchmarking RL agents.

These implement closed-form optimal policies from the literature:

- ``BarOnMansourThreshold``: From Bar-On & Mansour (2312.06448).
  Post if price < 2*alpha*waiting_time / (1 - gamma).

- ``ShouqiaoThreshold``: From Wang, Crapis, and Moallemi (2505.19556).
  Post if queue > Q*(price).  Requires numerical computation (deferred).
"""

from __future__ import annotations

import numpy as np


class BarOnMansourThreshold:
    """Analytical threshold policy from Bar-On & Mansour (2312.06448).

    For a martingale price process with linear delay cost, the optimal
    strategy is to post if the current price falls below a threshold
    that is linear in waiting time:

        threshold(x) = 2 * alpha * x / (1 - gamma)

    This is the exact closed-form optimum for the infinite-horizon
    discounted problem with mu = -sigma^2/2 (martingale prices).

    Args:
        alpha: Delay cost coefficient (same as ``rollup.alpha_i``).
        gamma: Discount factor.
        gas_fixed: Fixed gas cost scalar (normalizes the price comparison).
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        gas_fixed: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.gas_fixed = gas_fixed

    def threshold(self, waiting_time: int) -> float:
        """Compute the price threshold for a given waiting time.

        Post if current price < threshold(waiting_time).

        Args:
            waiting_time: Number of blocks since last post.

        Returns:
            Price threshold value.
        """
        return 2.0 * self.alpha * waiting_time / (1.0 - self.gamma)

    def act(self, obs: np.ndarray) -> int:
        """Select action given observation.

        Args:
            obs: Array [waiting_time, blob_fee, gas_price].  If prices are
                log-transformed, the comparison uses exp(gas_price).

        Returns:
            Action: 1 (post) or 0 (wait).
        """
        waiting_time = obs[0]
        price = obs[2]  # gas_price (the price process in Stage A)
        thresh = self.threshold(int(waiting_time))
        if self.gas_fixed > 0:
            effective_price = price * self.gas_fixed
        else:
            effective_price = price
        return 1 if effective_price < thresh else 0

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """SB3-compatible predict interface.

        Args:
            obs: Observation array (may be batched).
            deterministic: Unused (policy is always deterministic).

        Returns:
            Tuple of (action_array, None).
        """
        if obs.ndim == 1:
            return np.array([self.act(obs)]), None
        return np.array([self.act(o) for o in obs]), None
