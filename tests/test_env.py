"""Tests for BlobMarketEnv."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.abm.env import BlobMarketEnv


def _base_cfg(**overrides):
    """Create a minimal config for testing, with optional overrides."""
    cfg = OmegaConf.create({
        "env": {
            "queue_mode": "poisson_arrival",
            "action_mode": "discrete_blobs",
            "cost_mode": "linear_overhead",
            "reward_mode": "per_step",
            "max_steps": 100,
            "blob_fee_floor": 1,
            "gas_fixed": 21000,
            "blob_target": 3,
            "blob_max": 6,
            "log_transform_prices": False,
            "seed": 42,
        },
        "rollup": {
            "lambda_i": 10.0,
            "alpha_i": 1.0,
            "d_i": 500,
            "gamma": 0.99,
        },
        "price_process": {
            "type": "log_normal_rw",
            "ln_mu": 0.0,
            "ln_sigma": 0.01,
            "initial_price": 1.0,
        },
        "blob_fee_process": {
            "type": "exogenous_log_normal_rw",
            "sigma": 0.01,
            "initial_fee": 100.0,
        },
    })
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


class TestBlobMarketEnvBasic:
    """Basic environment lifecycle tests."""

    def test_reset_returns_valid_obs(self) -> None:
        env = BlobMarketEnv(_base_cfg())
        obs, info = env.reset()
        assert obs.shape == (3,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_wait_does_not_crash(self) -> None:
        env = BlobMarketEnv(_base_cfg())
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (3,)
        assert isinstance(reward, float)
        assert terminated is False
        assert truncated is False

    def test_step_post_does_not_crash(self) -> None:
        env = BlobMarketEnv(_base_cfg())
        env.reset()
        obs, reward, terminated, truncated, info = env.step(3)
        assert obs.shape == (3,)

    def test_truncation_at_max_steps(self) -> None:
        cfg = _base_cfg(**{"env": {"max_steps": 5}})
        env = BlobMarketEnv(cfg)
        env.reset()
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
        assert truncated is True
        assert terminated is False

    def test_never_terminates(self) -> None:
        env = BlobMarketEnv(_base_cfg())
        env.reset()
        for _ in range(50):
            _, _, terminated, _, _ = env.step(0)
            assert terminated is False


class TestBlobMarketEnvPoissonMode:
    """Tests specific to poisson_arrival mode."""

    def test_queue_grows_with_wait(self) -> None:
        cfg = _base_cfg(**{"rollup": {"lambda_i": 100.0}})
        env = BlobMarketEnv(cfg)
        env.reset()
        for _ in range(10):
            env.step(0)
        assert env.rollup.queue > 0, "Queue should grow when waiting"

    def test_queue_drains_on_post(self) -> None:
        cfg = _base_cfg(**{"rollup": {"lambda_i": 100.0}})
        env = BlobMarketEnv(cfg)
        env.reset()
        # Let queue build up
        for _ in range(10):
            env.step(0)
        q_before = env.rollup.queue
        env.step(6)  # post 6 blobs
        assert env.rollup.queue < q_before, "Queue should decrease after posting"

    def test_reward_negative_per_step(self) -> None:
        cfg = _base_cfg(**{"rollup": {"lambda_i": 50.0}})
        env = BlobMarketEnv(cfg)
        env.reset()
        # Let queue build, then check reward is negative
        for _ in range(5):
            env.step(0)
        _, reward, _, _, _ = env.step(0)
        assert reward < 0, "Reward should be negative (delay cost)"


class TestBlobMarketEnvFixedBatchMode:
    """Tests specific to fixed_batch mode."""

    def test_waiting_time_increments(self) -> None:
        cfg = _base_cfg(**{
            "env": {"queue_mode": "fixed_batch", "action_mode": "binary",
                    "cost_mode": "bar_on_mansour", "reward_mode": "on_post_only"},
        })
        env = BlobMarketEnv(cfg)
        env.reset()
        env.step(0)  # wait
        assert env.rollup.waiting_time == 1
        env.step(0)  # wait
        assert env.rollup.waiting_time == 2

    def test_waiting_time_resets_on_post(self) -> None:
        cfg = _base_cfg(**{
            "env": {"queue_mode": "fixed_batch", "action_mode": "binary",
                    "cost_mode": "bar_on_mansour", "reward_mode": "on_post_only"},
        })
        env = BlobMarketEnv(cfg)
        env.reset()
        for _ in range(5):
            env.step(0)
        assert env.rollup.waiting_time == 5
        env.step(1)  # post
        assert env.rollup.waiting_time == 0

    def test_on_post_only_reward_zero_while_waiting(self) -> None:
        cfg = _base_cfg(**{
            "env": {"queue_mode": "fixed_batch", "action_mode": "binary",
                    "cost_mode": "bar_on_mansour", "reward_mode": "on_post_only"},
        })
        env = BlobMarketEnv(cfg)
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0, "Reward should be 0 while waiting in on_post_only mode"

    def test_on_post_only_reward_negative_on_post(self) -> None:
        cfg = _base_cfg(**{
            "env": {"queue_mode": "fixed_batch", "action_mode": "binary",
                    "cost_mode": "bar_on_mansour", "reward_mode": "on_post_only"},
        })
        env = BlobMarketEnv(cfg)
        env.reset()
        # Wait a few steps to build delay cost
        for _ in range(5):
            env.step(0)
        _, reward, _, _, _ = env.step(1)  # post
        assert reward < 0, "Posting should incur negative reward"


class TestBlobMarketEnvReproducibility:
    """Test that same seed produces identical trajectories."""

    def test_same_seed_same_trajectory(self) -> None:
        cfg = _base_cfg()

        env_a = BlobMarketEnv(cfg)
        env_b = BlobMarketEnv(cfg)

        obs_a, _ = env_a.reset(seed=99)
        obs_b, _ = env_b.reset(seed=99)
        np.testing.assert_array_equal(obs_a, obs_b)

        actions = [0, 1, 3, 0, 6, 2, 0, 0, 4, 1]
        for action in actions:
            oa, ra, _, _, _ = env_a.step(action)
            ob, rb, _, _, _ = env_b.step(action)
            np.testing.assert_array_equal(oa, ob)
            assert ra == rb


class TestBlobMarketEnvLogTransform:
    """Test log-transform of prices in observations."""

    def test_log_transform_applied(self) -> None:
        cfg = _base_cfg(**{"env": {"log_transform_prices": True}})
        env = BlobMarketEnv(cfg)
        obs, _ = env.reset()
        # Initial gas price is 1.0 -> log(1.0) = 0.0
        # Initial blob fee is 100.0 -> log(100.0) ≈ 4.605
        assert obs[2] == pytest.approx(0.0, abs=0.01)  # log(gas_price=1.0)
        assert obs[1] == pytest.approx(np.log(100.0), abs=0.1)  # log(blob_fee=100)

    def test_no_log_transform(self) -> None:
        cfg = _base_cfg(**{"env": {"log_transform_prices": False}})
        env = BlobMarketEnv(cfg)
        obs, _ = env.reset()
        assert obs[2] == pytest.approx(1.0, abs=0.01)  # raw gas_price
        assert obs[1] == pytest.approx(100.0, abs=1.0)  # raw blob_fee
