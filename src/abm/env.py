"""BlobMarketEnv: Gymnasium-compatible environment for EIP-4844 blob posting.

Wraps a Mesa ABM with a single rollup agent and a pass-through builder.
All behavior is controlled by config flags so the same class supports
both Stage A (Bar-On & Mansour replication) and Stage B (Shouqiao et al.
relaxations).

Config flags:
    env.queue_mode:   "fixed_batch" | "poisson_arrival"
    env.action_mode:  "binary" | "discrete_blobs"
    env.cost_mode:    "bar_on_mansour" | "linear_overhead" | "quadratic_delay"
    env.reward_mode:  "per_step" | "on_post_only"
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium
import mesa
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.abm.agents.builder import BuilderAgent
from src.abm.agents.rollup import RollupAgent
from src.abm.fee_market import compute_posting_cost
from src.abm.gas_process import create_price_process


class BlobMarketEnv(gymnasium.Env):
    """Gymnasium environment for the EIP-4844 blob fee market.

    Observation space (Box, shape=(3,)):
        [queue_or_waiting_time, blob_base_fee, gas_price]
        Prices are optionally log-transformed (``env.log_transform_prices``).

    Action space:
        - ``Discrete(2)`` in ``binary`` mode: 0 = wait, 1 = post
        - ``Discrete(7)`` in ``discrete_blobs`` mode: 0..6 blobs

    Reward:
        Negative cost.  Depends on ``reward_mode``:
        - ``per_step``: delay cost every step + posting cost on post steps
        - ``on_post_only``: 0 while waiting, full cost on post step

    Args:
        cfg: Full OmegaConf config (with ``env``, ``rollup``,
            ``price_process``, ``blob_fee_process`` sections).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces from config."""
        if self.cfg.env.action_mode == "binary":
            self.action_space = gymnasium.spaces.Discrete(2)
        else:
            self.action_space = gymnasium.spaces.Discrete(7)

        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def _build_model(self, seed: int) -> None:
        """Initialize Mesa model, agents, and price processes."""
        self.gas_process = create_price_process(self.cfg.price_process, seed=seed)
        self.blob_fee_process = create_price_process(
            self.cfg.blob_fee_process, seed=seed + 1
        )
        self.mesa_model = mesa.Model(seed=seed + 2)
        self.rollup = RollupAgent(
            model=self.mesa_model,
            lambda_i=self.cfg.rollup.lambda_i,
            alpha_i=self.cfg.rollup.alpha_i,
            d_i=self.cfg.rollup.d_i,
            queue_mode=self.cfg.env.queue_mode,
            seed=seed + 3,
        )
        self.builder = BuilderAgent(model=self.mesa_model)
        self.gas_price: float = 0.0
        self.blob_fee: float = 0.0
        self.step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional seed override.
            options: Unused, for Gymnasium API compatibility.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.cfg.env.seed
        self._build_model(effective_seed)
        self.gas_price = self.gas_process.reset()
        self.blob_fee = self.blob_fee_process.reset()
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one block step.

        Args:
            action: Number of blobs to post (0 = wait).  In binary mode,
                0 = wait, 1 = post one blob.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self.step_count += 1

        # 1. Arrivals
        self.rollup.arrive()

        # 2. Delay cost (computed AFTER arrivals, on current queue/waiting_time)
        delay_cost = self.rollup.compute_delay_cost(self.cfg.env.cost_mode)

        # 3. Resolve action
        if self.cfg.env.action_mode == "binary":
            n_blobs = 1 if action == 1 else 0
        else:
            n_blobs = int(action)

        # 4. Execute posting
        posting_cost = 0.0
        did_post = False
        if n_blobs > 0:
            actual_blobs = self.rollup.post(n_blobs)
            if actual_blobs > 0:
                did_post = True
                posting_cost = compute_posting_cost(
                    blob_base_fee=self.blob_fee,
                    gas_price=self.gas_price,
                    n_blobs=actual_blobs,
                    gas_fixed=self.cfg.env.gas_fixed,
                    cost_mode=self.cfg.env.cost_mode,
                )

        # 5. Cost normalization (divide both costs by C_ref so they are O(1))
        cost_norm = self.cfg.env.get("cost_normalization", 1.0)
        if cost_norm != 1.0:
            delay_cost = delay_cost / cost_norm
            posting_cost = posting_cost / cost_norm

        # 6. Reward
        if self.cfg.env.reward_mode == "on_post_only":
            if did_post:
                reward = -(delay_cost + posting_cost)
            else:
                reward = 0.0
        else:  # per_step
            reward = -(delay_cost + posting_cost)

        # 6. Update exogenous prices
        self.gas_price = self.gas_process.step()
        self.blob_fee = self.blob_fee_process.step()
        blob_fee_floor = self.cfg.env.get("blob_fee_floor", 1)
        self.blob_fee = max(blob_fee_floor, self.blob_fee)

        # 7. Termination
        truncated = self.step_count >= self.cfg.env.max_steps
        terminated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        if self.cfg.env.queue_mode == "fixed_batch":
            q = float(self.rollup.waiting_time)
        else:
            q = float(self.rollup.queue)

        blob_fee = self.blob_fee
        gas_price = self.gas_price

        if self.cfg.env.get("log_transform_prices", False):
            blob_fee = math.log(max(blob_fee, 1e-30))
            gas_price = math.log(max(gas_price, 1e-30))

        return np.array([q, blob_fee, gas_price], dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Build info dict with diagnostic data."""
        return {
            "step": self.step_count,
            "queue": self.rollup.queue,
            "waiting_time": self.rollup.waiting_time,
            "gas_price": self.gas_price,
            "blob_fee": self.blob_fee,
        }
