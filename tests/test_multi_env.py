"""Tests for MultiAgentBlobEnv."""

import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.abm.fee_market import update_blob_base_fee
from src.abm.multi_env import MultiAgentBlobEnv


def _base_cfg(labels=None, **overrides):
    """Minimal config for Phase 2 tests."""
    if labels is None:
        labels = ["taiko", "base", "scroll"]
    cfg = OmegaConf.create({
        "env": {
            "queue_mode": "poisson_arrival",
            "action_mode": "discrete_blobs",
            "cost_mode": "linear_overhead",
            "reward_mode": "per_step",
            "max_steps": 50,
            "blob_fee_floor": 1.0,
            "gas_fixed": 21000,
            "blob_target": 3,
            "blob_max": 6,
            "log_transform_prices": False,
            "cost_normalization": 1.0,
            "seed": 0,
        },
        "blob_fee_init": {
            "initial_fee": 1.0,
            "floor": 1.0,
        },
        "price_process": {
            "type": "ar1",
            "mu": 1.0e-8,
            "theta": 0.01,
            "sigma": 1.0e-9,
            "initial_price": 1.0e-8,
            "floor": 0.0,
        },
        "rollup_arrival": {
            "rollup_rates": {
                "taiko": 0.8,
                "base": 0.36,
                "scroll": 0.15,
                "arbitrum_one": 0.22,
            },
        },
        "rollups": {
            "labels": labels,
            "alpha_scale": 1.0,
            "d_i": 500,
            "gamma": 0.99,
        },
    })
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def test_reset_shapes():
    env = MultiAgentBlobEnv(_base_cfg())
    obs, info = env.reset()
    assert set(obs) == {"taiko", "base", "scroll"}
    for v in obs.values():
        assert v.shape == (3,)
    assert env.blob_fee == pytest.approx(1.0)


def test_step_returns_per_agent_dicts():
    env = MultiAgentBlobEnv(_base_cfg())
    env.reset()
    actions = {"taiko": 2, "base": 1, "scroll": 0}
    obs, rewards, term, trunc, info = env.step(actions)
    assert set(obs) == set(rewards) == set(term) == set(trunc)
    assert info["total_blobs"] >= 0
    for r in rewards.values():
        assert r <= 0.0  # costs are non-negative, reward is -cost


def test_endogenous_fee_matches_eip4844_rule():
    """Post exactly 6 blobs (above target 3). Fee should increase per the rule."""
    cfg = _base_cfg()
    env = MultiAgentBlobEnv(cfg)
    env.reset()
    # Build up queues so all 3 agents can post 2 blobs each -> 6 total.
    # Run a few arrival-only steps with action 0 to fill queues.
    for _ in range(5):
        env.step({"taiko": 0, "base": 0, "scroll": 0})

    fee_before = env.blob_fee
    _, _, _, _, info = env.step({"taiko": 2, "base": 2, "scroll": 2})
    # effective_blobs = min(6, 6) = 6
    expected = update_blob_base_fee(fee_before, blobs_used=info["effective_blobs"], target=3)
    expected = max(cfg.env.blob_fee_floor, expected)
    assert env.blob_fee == pytest.approx(expected, rel=1e-6)


def test_fee_stays_at_floor_when_no_one_posts():
    env = MultiAgentBlobEnv(_base_cfg())
    env.reset()
    # With 0 blobs vs target 3, the fee should decay each block but be
    # clamped to the floor.
    prev = env.blob_fee
    for _ in range(20):
        env.step({lbl: 0 for lbl in env.agent_ids})
        assert env.blob_fee >= float(env.cfg.env.blob_fee_floor)
        # Fee should be monotone non-increasing while at or above floor.
        assert env.blob_fee <= prev + 1e-9
        prev = env.blob_fee


def test_heterogeneous_alpha_i():
    """alpha_i should scale with lambda_i per the calibration rule."""
    cfg = _base_cfg()
    env = MultiAgentBlobEnv(cfg)
    env.reset()
    alpha_taiko = env.rollups["taiko"].alpha_i
    alpha_scroll = env.rollups["scroll"].alpha_i
    # lambda_taiko = 0.8, lambda_scroll = 0.15 -> ratio ~ 5.33
    assert alpha_taiko > alpha_scroll
    assert alpha_taiko / alpha_scroll == pytest.approx(0.8 / 0.15, rel=1e-6)


def test_queue_isolation():
    """Each rollup's queue should evolve independently."""
    env = MultiAgentBlobEnv(_base_cfg())
    env.reset()
    for _ in range(10):
        env.step({"taiko": 0, "base": 0, "scroll": 0})
    qs = [env.rollups[l].queue for l in env.agent_ids]
    # With different lambdas, expected queue sizes should differ.
    assert len(set(qs)) >= 2  # at least two distinct queue lengths


def test_truncation_at_max_steps():
    cfg = _base_cfg()
    cfg.env.max_steps = 3
    env = MultiAgentBlobEnv(cfg)
    env.reset()
    for t in range(3):
        obs, rew, term, trunc, info = env.step({lbl: 0 for lbl in env.agent_ids})
    assert all(trunc.values())


def test_missing_label_raises():
    cfg = _base_cfg(labels=["taiko", "nonexistent_rollup"])
    with pytest.raises(KeyError):
        MultiAgentBlobEnv(cfg)


def test_calldata_action_space_is_8():
    cfg = _base_cfg()
    cfg.env.action_mode = "with_calldata"
    env = MultiAgentBlobEnv(cfg)
    assert env.single_action_space.n == 8


def test_calldata_drains_queue_and_does_not_update_fee():
    cfg = _base_cfg()
    cfg.env.action_mode = "with_calldata"
    env = MultiAgentBlobEnv(cfg)
    env.reset()
    # Fill queues with 10 arrival-only steps.
    for _ in range(10):
        env.step({lbl: 0 for lbl in env.agent_ids})
    # Sanity: at least one rollup has queue > 0.
    assert any(env.rollups[lbl].queue > 0 for lbl in env.agent_ids)

    fee_before = env.blob_fee
    # All agents post via calldata (action 7).
    _, _, _, _, info = env.step({lbl: 7 for lbl in env.agent_ids})

    # All queues should be empty; no blobs used; fee decays as if 0 blobs posted.
    for lbl in env.agent_ids:
        assert env.rollups[lbl].queue == 0
        assert info["calldata_tx_by_agent"][lbl] >= 0
    assert info["total_blobs"] == 0
    # fee evolves via update rule with 0 blobs (which drives it DOWN, but it
    # was already at 1.0 floor -> stays at floor).
    assert env.blob_fee <= fee_before + 1e-9


def test_calldata_cost_scales_with_queue():
    """Calldata cost should scale linearly with tx drained."""
    cfg = _base_cfg()
    cfg.env.action_mode = "with_calldata"
    cfg.env.cost_normalization = 1.0  # raw costs for easy comparison
    env = MultiAgentBlobEnv(cfg)
    env.reset()
    # Accumulate different-sized queues across 10 idle steps.
    for _ in range(10):
        env.step({lbl: 0 for lbl in env.agent_ids})

    queues = {lbl: env.rollups[lbl].queue for lbl in env.agent_ids}
    _, rewards, _, _, info = env.step(
        {lbl: 7 for lbl in env.agent_ids}
    )
    # Agents with non-zero queue should have more negative reward than those with 0.
    rewards_by_queue = sorted(
        ((queues[l], rewards[l]) for l in env.agent_ids), key=lambda x: x[0]
    )
    # Non-decreasing queue should give non-increasing reward (more negative).
    for (q1, r1), (q2, r2) in zip(rewards_by_queue[:-1], rewards_by_queue[1:]):
        if q2 > q1:
            assert r2 <= r1 + 1e-6


def test_mixed_blob_and_calldata_actions():
    """Some agents post blobs, others use calldata, simultaneously."""
    cfg = _base_cfg()
    cfg.env.action_mode = "with_calldata"
    env = MultiAgentBlobEnv(cfg)
    env.reset()
    for _ in range(10):
        env.step({lbl: 0 for lbl in env.agent_ids})

    actions = {"taiko": 3, "base": 7, "scroll": 1}
    _, _, _, _, info = env.step(actions)
    # taiko posted blobs -> non-zero blobs_by_agent
    assert info["blobs_by_agent"]["taiko"] > 0
    # base used calldata -> zero blobs, calldata tx > 0
    assert info["blobs_by_agent"]["base"] == 0
    assert info["calldata_tx_by_agent"]["base"] > 0
    # scroll posted blobs
    assert info["blobs_by_agent"]["scroll"] >= 0
