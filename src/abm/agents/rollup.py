"""RollupAgent: Mesa agent representing a single L2 rollup.

Manages queue state (Poisson arrivals or fixed-batch waiting) and delegates
action selection to a pluggable policy.  The Gymnasium wrapper (env.py) sets
the action externally each step.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import mesa
import numpy as np


class Policy(Protocol):
    """Interface for pluggable rollup policies."""

    def act(self, obs: np.ndarray) -> int: ...


class RollupAgent(mesa.Agent):
    """Mesa agent for a single rollup.

    Supports two queue modes controlled by ``queue_mode``:

    - ``"poisson_arrival"``: Queue grows by ``Poisson(lambda_i)`` each step.
      Posting ``k`` blobs drains ``k * tx_per_blob()`` transactions.
    - ``"fixed_batch"``: No stochastic arrivals; ``waiting_time`` increments
      by 1 each step.  Posting resets ``waiting_time`` to 0.  This matches
      the Bar-On & Mansour (2312.06448) formulation.

    Args:
        model: Parent Mesa model.
        lambda_i: Transaction arrival rate (tx/block, Poisson mode).
        alpha_i: Delay cost coefficient.
        d_i: Data size per transaction in bytes.
        queue_mode: ``"poisson_arrival"`` or ``"fixed_batch"``.
        seed: Random seed for the agent's RNG.
    """

    def __init__(
        self,
        model: mesa.Model,
        lambda_i: float = 180.0,
        alpha_i: float = 1.0,
        d_i: int = 500,
        queue_mode: str = "poisson_arrival",
        seed: int | None = None,
    ) -> None:
        super().__init__(model)
        self.lambda_i = lambda_i
        self.alpha_i = alpha_i
        self.d_i = d_i
        self.queue_mode = queue_mode

        # State
        self.queue: int = 0
        self.waiting_time: int = 0

        # Pluggable policy (set externally or by env wrapper)
        self.policy: Policy | None = None

        self._rng = np.random.default_rng(seed)

    def tx_per_blob(self) -> int:
        """Number of transactions that fit in one 128 KB blob.

        Returns:
            At least 1 transaction per blob.
        """
        return max(1, 131_072 // self.d_i)

    def arrive(self) -> None:
        """Generate new transaction arrivals for this block.

        In ``poisson_arrival`` mode, draws from Poisson(lambda_i).
        In ``fixed_batch`` mode, increments waiting_time by 1.
        """
        if self.queue_mode == "poisson_arrival":
            self.queue += int(self._rng.poisson(self.lambda_i))
        else:  # fixed_batch
            self.waiting_time += 1

    def post(self, n_blobs: int) -> int:
        """Post blobs, drain queue, return actual blobs used.

        Args:
            n_blobs: Requested number of blobs to post (1 to 6).

        Returns:
            Actual number of blobs posted (may be less if queue is small).
        """
        if self.queue_mode == "poisson_arrival":
            max_drain = n_blobs * self.tx_per_blob()
            actual_drain = min(max_drain, self.queue)
            self.queue -= actual_drain
            if actual_drain == 0:
                return 0
            actual_blobs = math.ceil(actual_drain / self.tx_per_blob())
            return min(actual_blobs, n_blobs)
        else:  # fixed_batch
            self.waiting_time = 0
            return 1  # always 1 "blob" in fixed-batch mode

    def post_calldata(self) -> int:
        """Post the entire queue via L1 calldata (no blob).

        Calldata has effectively unbounded capacity per transaction but is
        priced per-byte at the L1 gas rate, so it is usually more expensive
        than blobs at low blob fees and cheaper when the blob fee is high.

        Returns:
            Number of transactions drained from the queue.
        """
        if self.queue_mode == "poisson_arrival":
            drained = self.queue
            self.queue = 0
            return drained
        # fixed_batch mode: a calldata post resets the waiting time,
        # analogous to a normal post.
        self.waiting_time = 0
        return 1

    def compute_delay_cost(self, cost_mode: str = "linear_overhead") -> float:
        """Compute current-step delay cost.

        Args:
            cost_mode: Cost calculation mode.
                - ``"bar_on_mansour"``: ``alpha_i * waiting_time``
                - ``"linear_overhead"``: ``alpha_i * queue``
                - ``"quadratic_delay"``: ``alpha_i * queue^2``

        Returns:
            Delay cost for this step.
        """
        if cost_mode == "bar_on_mansour":
            return self.alpha_i * self.waiting_time

        if cost_mode == "linear_overhead":
            return self.alpha_i * self.queue

        if cost_mode == "quadratic_delay":
            return self.alpha_i * self.queue ** 2

        raise ValueError(f"Unknown cost_mode: {cost_mode!r}")

    def reset(self) -> None:
        """Reset agent state to initial conditions."""
        self.queue = 0
        self.waiting_time = 0

    def step(self) -> None:
        """Mesa step hook.  Arrivals happen here; action is set externally."""
        self.arrive()
