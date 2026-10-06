"""MultiAgentBlobEnv: multi-rollup ABM with endogenous EIP-4844 blob fee.

Phase 2 environment. N heterogeneous rollups share a common blob base fee
that evolves deterministically via the EIP-4844 rule applied to aggregated
blobs per block. The L1 gas price is still exogenous (AR(1)).

Interface mirrors PettingZoo's ParallelEnv conventions but has no external
dependency: methods take/return dicts keyed by rollup label.

    reset()                         -> (obs_dict, info)
    step(action_dict: {id: int})    -> (obs_dict, reward_dict,
                                        term_dict, trunc_dict, info)

Each per-rollup observation is:
    [own_queue_or_wait, blob_base_fee, gas_price]
(optionally log-transformed; same convention as src/abm/env.py).
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium
import mesa
import numpy as np
from omegaconf import DictConfig

from src.abm.agents.rollup import RollupAgent
from src.abm.fee_market import compute_posting_cost, update_blob_base_fee
from src.abm.gas_process import create_price_process


RollupId = str


def _build_rollup_spec(cfg: DictConfig) -> list[tuple[RollupId, float, float]]:
    """Resolve per-rollup (label, lambda_i, alpha_i) from config.

    Pulls labels from ``cfg.rollups.labels``, looks up arrival rates in
    ``cfg.rollup_arrival.rollup_rates`` (merged from calibration.yaml),
    and derives ``alpha_i`` via revealed-preference rule.

    Args:
        cfg: Full resolved config.

    Returns:
        List of (label, lambda_i, alpha_i) tuples in config order.
    """
    rates = cfg.rollup_arrival.rollup_rates
    c_ref = float(cfg.env.cost_normalization)
    alpha_scale = float(cfg.rollups.alpha_scale)
    lambda_scale = float(cfg.rollups.get("lambda_scale", 1.0))

    spec: list[tuple[RollupId, float, float]] = []
    for label in cfg.rollups.labels:
        if label not in rates:
            raise KeyError(
                f"Rollup label {label!r} not found in calibration rollup_rates. "
                f"Available: {list(rates.keys())}"
            )
        lam = lambda_scale * float(rates[label])
        # alpha_i uses the unscaled (empirical) lambda so that delay-cost
        # magnitudes stay calibrated while arrival intensity changes.
        # alpha_fixed_lambda replaces the per-rollup lambda with a common
        # reference value (fixed delay-coefficient ablation).
        alpha_lambda = cfg.rollups.get("alpha_fixed_lambda", None)
        alpha_lambda = float(rates[label]) if alpha_lambda is None else float(alpha_lambda)
        alpha = alpha_scale * 2.0 * c_ref * alpha_lambda
        spec.append((label, lam, alpha))
    return spec


class MultiAgentBlobEnv:
    """Multi-rollup blob-market environment with endogenous fee.

    The blob base fee ``B_t`` starts at ``cfg.blob_fee_init.initial_fee``
    and evolves via the EIP-4844 rule applied to the *aggregate* blobs
    posted across all rollups each block.

    Args:
        cfg: Full OmegaConf config (base + calibration + phase2 merged).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.rollup_spec = _build_rollup_spec(cfg)
        self.agent_ids: list[RollupId] = [label for label, _, _ in self.rollup_spec]
        self.n_agents = len(self.agent_ids)

        # Action / observation spaces are identical across agents.
        #   binary         : Discrete(2) = {wait, post-one}
        #   discrete_blobs : Discrete(7) = {wait, post 1..6 blobs}
        #   with_calldata  : Discrete(8) = {wait, post 1..6 blobs, calldata}
        if cfg.env.action_mode == "binary":
            self.single_action_space = gymnasium.spaces.Discrete(2)
        elif cfg.env.action_mode == "with_calldata":
            self.single_action_space = gymnasium.spaces.Discrete(8)
        else:
            self.single_action_space = gymnasium.spaces.Discrete(7)

        self.single_observation_space = gymnasium.spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self.rollups: dict[RollupId, RollupAgent] = {}
        self.gas_process = None
        self.blob_fee: float = 0.0
        self.gas_price: float = 0.0
        self.step_count: int = 0
        self._np_rng = np.random.default_rng(cfg.env.seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[RollupId, np.ndarray], dict[str, Any]]:
        """Reset all agents, fee, and gas process.

        Args:
            seed: Optional seed override.

        Returns:
            (obs_dict, info) where obs_dict maps rollup id to observation.
        """
        effective_seed = seed if seed is not None else int(self.cfg.env.seed)
        self._np_rng = np.random.default_rng(effective_seed)

        self.gas_process = create_price_process(
            self.cfg.price_process, seed=effective_seed
        )
        self.gas_price = self.gas_process.reset()

        self.blob_fee = float(self.cfg.blob_fee_init.initial_fee)

        self.mesa_model = mesa.Model(seed=effective_seed + 2)
        self.rollups = {}
        for i, (label, lam, alpha) in enumerate(self.rollup_spec):
            self.rollups[label] = RollupAgent(
                model=self.mesa_model,
                lambda_i=lam,
                alpha_i=alpha,
                d_i=int(self.cfg.rollups.d_i),
                queue_mode=self.cfg.env.queue_mode,
                seed=effective_seed + 1000 + i,
            )

        self.step_count = 0
        return self._get_obs_dict(), self._get_info()

    def step(
        self, action_dict: dict[RollupId, int]
    ) -> tuple[
        dict[RollupId, np.ndarray],
        dict[RollupId, float],
        dict[RollupId, bool],
        dict[RollupId, bool],
        dict[str, Any],
    ]:
        """Advance one block. All agents act simultaneously.

        Args:
            action_dict: {rollup_id: action}. Missing agents default to 0.

        Returns:
            (obs_dict, reward_dict, term_dict, trunc_dict, info).
        """
        self.step_count += 1
        cost_norm = float(self.cfg.env.get("cost_normalization", 1.0))
        cost_mode = self.cfg.env.cost_mode
        blob_fee_pre = self.blob_fee
        gas_price_pre = self.gas_price

        # 1) Arrivals for all rollups.
        for agent in self.rollups.values():
            agent.arrive()

        # 2) Delay cost (post-arrival), computed per agent.
        delay_costs = {
            label: agent.compute_delay_cost(cost_mode)
            for label, agent in self.rollups.items()
        }

        # 3) Resolve per-agent action -> number of blobs requested,
        #    then execute posting against the agent's queue.
        posting_costs: dict[RollupId, float] = {}
        blobs_by_agent: dict[RollupId, int] = {}
        calldata_tx_by_agent: dict[RollupId, int] = {}
        gas_fixed = int(self.cfg.env.gas_fixed)
        action_mode = self.cfg.env.action_mode
        calldata_byte_cost = int(self.cfg.env.get("calldata_byte_cost", 16))

        for label, agent in self.rollups.items():
            action = int(action_dict.get(label, 0))

            if action_mode == "with_calldata" and action == 7:
                # Calldata post: drain entire queue, no blob consumption.
                drained = agent.post_calldata()
                calldata_tx_by_agent[label] = drained
                blobs_by_agent[label] = 0
                if drained > 0:
                    posting_costs[label] = gas_price_pre * (
                        gas_fixed + calldata_byte_cost * int(self.cfg.rollups.d_i) * drained
                    )
                else:
                    posting_costs[label] = 0.0
                continue

            if action_mode == "binary":
                requested = 1 if action == 1 else 0
            else:
                requested = action

            if requested > 0:
                posted = agent.post(requested)
            else:
                posted = 0

            blobs_by_agent[label] = posted
            calldata_tx_by_agent[label] = 0
            if posted > 0:
                posting_costs[label] = compute_posting_cost(
                    blob_base_fee=blob_fee_pre,
                    gas_price=gas_price_pre,
                    n_blobs=posted,
                    gas_fixed=gas_fixed,
                    cost_mode=cost_mode,
                )
            else:
                posting_costs[label] = 0.0

        total_blobs = int(sum(blobs_by_agent.values()))
        # Enforce the per-block cap (6). If agents collectively exceed it,
        # we clip the effective blob count for the fee update. The extras
        # count as "not included this block" for the fee, while the queues
        # are already drained and the costs already paid. The model has no
        # builder-side rationing.
        effective_blobs = min(total_blobs, int(self.cfg.env.blob_max))

        # 4) Normalize costs.
        if cost_norm != 1.0:
            delay_costs = {k: v / cost_norm for k, v in delay_costs.items()}
            posting_costs = {k: v / cost_norm for k, v in posting_costs.items()}

        # 5) Rewards.
        rewards: dict[RollupId, float] = {}
        reward_mode = self.cfg.env.reward_mode
        reward_scale = float(self.cfg.env.get("reward_scale", 1.0))
        for label in self.agent_ids:
            delay = delay_costs[label]
            post = posting_costs[label]
            if reward_mode == "on_post_only":
                r = -(delay + post) if blobs_by_agent[label] > 0 else 0.0
            else:
                r = -(delay + post)
            rewards[label] = r * reward_scale

        # 6) Update blob base fee via the EIP-4844 rule.
        blob_fee_floor = float(self.cfg.env.get("blob_fee_floor", 1.0))
        blob_fee_ceiling = float(self.cfg.env.get("blob_fee_ceiling", float("inf")))
        self.blob_fee = update_blob_base_fee(
            self.blob_fee,
            blobs_used=effective_blobs,
            target=int(self.cfg.env.blob_target),
        )
        self.blob_fee = max(blob_fee_floor, self.blob_fee)
        self.blob_fee = min(blob_fee_ceiling, self.blob_fee)

        # 7) Advance exogenous gas price.
        self.gas_price = self.gas_process.step()

        # 8) Termination.
        truncated = self.step_count >= int(self.cfg.env.max_steps)
        term_dict = {label: False for label in self.agent_ids}
        trunc_dict = {label: truncated for label in self.agent_ids}

        info = self._get_info()
        info.update(
            {
                "blobs_by_agent": blobs_by_agent,
                "calldata_tx_by_agent": calldata_tx_by_agent,
                "total_blobs": total_blobs,
                "effective_blobs": effective_blobs,
                "blob_fee_pre": blob_fee_pre,
            }
        )
        return self._get_obs_dict(), rewards, term_dict, trunc_dict, info

    # ------------------------------------------------------------------
    # Observation / info helpers
    # ------------------------------------------------------------------
    def _get_obs_dict(self) -> dict[RollupId, np.ndarray]:
        blob_fee = self.blob_fee
        gas_price = self.gas_price
        if self.cfg.env.get("log_transform_prices", False):
            blob_fee = math.log(max(blob_fee, 1e-30))
            gas_price = math.log(max(gas_price, 1e-30))

        # Queue is expressed in blob-units (queue / tx_per_blob) so the
        # observation stays on an O(1)-O(10) scale regardless of the raw
        # arrival-rate scaling. The MLP trunk is sensitive to input scale
        # and diverges to NaN when queues reach tens of thousands raw.
        normalize_queue = self.cfg.env.get("normalize_queue_by_blob_capacity", True)

        out: dict[RollupId, np.ndarray] = {}
        for label, agent in self.rollups.items():
            if self.cfg.env.queue_mode == "fixed_batch":
                q = float(agent.waiting_time)
            else:
                raw_q = float(agent.queue)
                if normalize_queue:
                    q = raw_q / float(max(1, agent.tx_per_blob()))
                else:
                    q = raw_q
            out[label] = np.array([q, blob_fee, gas_price], dtype=np.float32)
        return out

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "blob_fee": self.blob_fee,
            "gas_price": self.gas_price,
            "queues": {lbl: a.queue for lbl, a in self.rollups.items()},
        }
