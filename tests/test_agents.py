"""Tests for RollupAgent and BuilderAgent."""

import math

import mesa
import numpy as np
import pytest

from src.abm.agents.builder import BuilderAgent
from src.abm.agents.rollup import RollupAgent


def _make_model(seed: int = 42) -> mesa.Model:
    """Create a minimal Mesa model for testing."""
    return mesa.Model(seed=seed)


class TestRollupAgentPoissonMode:
    """Tests for RollupAgent in poisson_arrival mode."""

    def test_initial_state(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, lambda_i=10.0, queue_mode="poisson_arrival", seed=0)
        assert agent.queue == 0
        assert agent.waiting_time == 0

    def test_arrive_grows_queue(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, lambda_i=100.0, queue_mode="poisson_arrival", seed=42)
        agent.arrive()
        assert agent.queue > 0, "Queue should grow after arrive with lambda=100"

    def test_mean_arrival_rate(self) -> None:
        """Mean queue growth per step should approximate lambda_i."""
        lambda_i = 50.0
        model = _make_model()
        agent = RollupAgent(model, lambda_i=lambda_i, queue_mode="poisson_arrival", seed=123)
        n_steps = 10_000
        for _ in range(n_steps):
            agent.arrive()
        mean_per_step = agent.queue / n_steps
        assert abs(mean_per_step - lambda_i) < 2.0, (
            f"Mean arrivals {mean_per_step} too far from lambda={lambda_i}"
        )

    def test_tx_per_blob(self) -> None:
        model = _make_model()
        # 500 bytes/tx -> floor(131072/500) = 262 tx/blob
        agent = RollupAgent(model, d_i=500, queue_mode="poisson_arrival", seed=0)
        assert agent.tx_per_blob() == 262

        # 131072 bytes/tx -> exactly 1 tx/blob
        agent2 = RollupAgent(model, d_i=131_072, queue_mode="poisson_arrival", seed=0)
        assert agent2.tx_per_blob() == 1

        # Very large d_i -> still at least 1
        agent3 = RollupAgent(model, d_i=500_000, queue_mode="poisson_arrival", seed=0)
        assert agent3.tx_per_blob() == 1

    def test_post_drains_queue(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, d_i=500, queue_mode="poisson_arrival", seed=0)
        agent.queue = 1000
        # 1 blob drains 262 tx
        actual = agent.post(1)
        assert actual == 1
        assert agent.queue == 1000 - 262

    def test_post_caps_at_queue_size(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, d_i=500, queue_mode="poisson_arrival", seed=0)
        agent.queue = 100  # less than 1 blob's capacity (262)
        actual = agent.post(3)
        assert agent.queue == 0
        assert actual == 1  # only needed 1 blob for 100 tx

    def test_queue_never_negative(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, d_i=500, queue_mode="poisson_arrival", seed=0)
        agent.queue = 0
        actual = agent.post(6)
        assert agent.queue == 0
        assert actual == 0

    def test_post_multiple_blobs(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, d_i=500, queue_mode="poisson_arrival", seed=0)
        # 3 blobs can drain 3*262 = 786 tx
        agent.queue = 786
        actual = agent.post(3)
        assert actual == 3
        assert agent.queue == 0

    def test_reset(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, queue_mode="poisson_arrival", seed=0)
        agent.queue = 500
        agent.waiting_time = 10
        agent.reset()
        assert agent.queue == 0
        assert agent.waiting_time == 0


class TestRollupAgentFixedBatchMode:
    """Tests for RollupAgent in fixed_batch mode."""

    def test_arrive_increments_waiting_time(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, queue_mode="fixed_batch", seed=0)
        assert agent.waiting_time == 0
        agent.arrive()
        assert agent.waiting_time == 1
        agent.arrive()
        assert agent.waiting_time == 2

    def test_post_resets_waiting_time(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, queue_mode="fixed_batch", seed=0)
        agent.waiting_time = 10
        actual = agent.post(1)
        assert actual == 1
        assert agent.waiting_time == 0

    def test_queue_unchanged_in_fixed_batch(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, queue_mode="fixed_batch", seed=0)
        agent.arrive()
        assert agent.queue == 0  # queue not used in fixed_batch mode


class TestRollupAgentDelayCost:
    """Tests for delay cost computation."""

    def test_bar_on_mansour_cost(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, alpha_i=2.0, queue_mode="fixed_batch", seed=0)
        agent.waiting_time = 5
        assert agent.compute_delay_cost("bar_on_mansour") == 10.0

    def test_linear_overhead_cost(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, alpha_i=3.0, queue_mode="poisson_arrival", seed=0)
        agent.queue = 100
        assert agent.compute_delay_cost("linear_overhead") == 300.0

    def test_quadratic_delay_cost(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, alpha_i=0.01, queue_mode="poisson_arrival", seed=0)
        agent.queue = 50
        assert agent.compute_delay_cost("quadratic_delay") == pytest.approx(0.01 * 2500)

    def test_unknown_cost_mode_raises(self) -> None:
        model = _make_model()
        agent = RollupAgent(model, seed=0)
        with pytest.raises(ValueError, match="Unknown cost_mode"):
            agent.compute_delay_cost("invalid")


class TestBuilderAgent:
    """Tests for the Phase 1 pass-through builder."""

    def test_include_passes_through(self) -> None:
        model = _make_model()
        builder = BuilderAgent(model)
        tx_list = [{"tx": 1}, {"tx": 2}]
        result = builder.include(tx_list)
        assert result is tx_list

    def test_step_no_error(self) -> None:
        model = _make_model()
        builder = BuilderAgent(model)
        builder.step()  # should not raise
