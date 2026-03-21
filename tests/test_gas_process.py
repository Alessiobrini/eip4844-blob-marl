"""Tests for stochastic price processes."""

import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.abm.gas_process import (
    AR1Process,
    LogNormalRandomWalk,
    create_price_process,
)


class TestLogNormalRandomWalk:
    """Tests for the log-normal multiplicative random walk."""

    def test_initial_price(self) -> None:
        proc = LogNormalRandomWalk(mu=0.0, sigma=0.1, initial_price=5.0, seed=0)
        assert proc.current_price == 5.0

    def test_reset_restores_initial(self) -> None:
        proc = LogNormalRandomWalk(mu=0.0, sigma=0.1, initial_price=5.0, seed=0)
        for _ in range(100):
            proc.step()
        assert proc.current_price != 5.0
        proc.reset()
        assert proc.current_price == 5.0

    def test_prices_stay_positive(self) -> None:
        proc = LogNormalRandomWalk(mu=0.0, sigma=0.5, initial_price=1.0, seed=42)
        for _ in range(10_000):
            p = proc.step()
            assert p > 0, "Log-normal RW should always be positive"

    def test_martingale_property(self) -> None:
        """With mu = -sigma^2/2, E[P_{t+1}] = P_t (true martingale)."""
        sigma = 0.1
        mu = -sigma**2 / 2  # martingale correction
        n_samples = 50_000
        p0 = 10.0

        ratios = []
        for i in range(n_samples):
            proc = LogNormalRandomWalk(mu=mu, sigma=sigma, initial_price=p0, seed=i)
            p1 = proc.step()
            ratios.append(p1 / p0)

        mean_ratio = np.mean(ratios)
        # Should be close to 1.0 (E[P1/P0] = 1 for a martingale)
        assert abs(mean_ratio - 1.0) < 0.02, f"Mean ratio {mean_ratio} too far from 1.0"

    def test_reproducibility(self) -> None:
        proc_a = LogNormalRandomWalk(mu=0.0, sigma=0.1, initial_price=1.0, seed=123)
        proc_b = LogNormalRandomWalk(mu=0.0, sigma=0.1, initial_price=1.0, seed=123)
        for _ in range(100):
            assert proc_a.step() == proc_b.step()


class TestAR1Process:
    """Tests for the AR(1) mean-reverting process."""

    def test_initial_price(self) -> None:
        proc = AR1Process(mu=10.0, theta=0.1, sigma=1.0, initial_price=5.0, seed=0)
        assert proc.current_price == 5.0

    def test_reset_restores_initial(self) -> None:
        proc = AR1Process(mu=10.0, theta=0.1, sigma=1.0, initial_price=5.0, seed=0)
        for _ in range(100):
            proc.step()
        proc.reset()
        assert proc.current_price == 5.0

    def test_mean_reversion(self) -> None:
        """Starting far from mu, the process should converge toward mu."""
        mu = 10.0
        proc = AR1Process(
            mu=mu, theta=0.3, sigma=0.01, initial_price=100.0, seed=42
        )
        for _ in range(500):
            proc.step()
        # After many steps with strong mean reversion and low noise, should be near mu
        assert abs(proc.current_price - mu) < 1.0, (
            f"Price {proc.current_price} too far from mu={mu}"
        )

    def test_floor_prevents_negative(self) -> None:
        proc = AR1Process(
            mu=0.001, theta=0.1, sigma=0.1, initial_price=0.001,
            floor=0.0, seed=42,
        )
        for _ in range(10_000):
            p = proc.step()
            assert p >= 0.0, f"Price {p} went below floor"

    def test_reproducibility(self) -> None:
        proc_a = AR1Process(mu=1.0, theta=0.1, sigma=0.5, initial_price=1.0, seed=99)
        proc_b = AR1Process(mu=1.0, theta=0.1, sigma=0.5, initial_price=1.0, seed=99)
        for _ in range(100):
            assert proc_a.step() == proc_b.step()


class TestCreatePriceProcess:
    """Tests for the factory function."""

    def test_log_normal_rw(self) -> None:
        cfg = OmegaConf.create({
            "type": "log_normal_rw",
            "ln_mu": 0.0,
            "ln_sigma": 0.1,
            "initial_price": 1.0,
        })
        proc = create_price_process(cfg, seed=0)
        assert isinstance(proc, LogNormalRandomWalk)

    def test_ar1(self) -> None:
        cfg = OmegaConf.create({
            "type": "ar1",
            "mu": 10.0,
            "theta": 0.1,
            "sigma": 1.0,
            "initial_price": 10.0,
            "floor": 0.0,
        })
        proc = create_price_process(cfg, seed=0)
        assert isinstance(proc, AR1Process)

    def test_exogenous_ar1(self) -> None:
        cfg = OmegaConf.create({
            "type": "exogenous_ar1",
            "mu": 1.0,
            "theta": 0.05,
            "sigma": 0.2,
            "initial_fee": 1.0,
            "floor": 1.0,
        })
        proc = create_price_process(cfg, seed=0)
        assert isinstance(proc, AR1Process)

    def test_exogenous_log_normal_rw(self) -> None:
        cfg = OmegaConf.create({
            "type": "exogenous_log_normal_rw",
            "sigma": 0.2,
            "initial_fee": 1.0,
        })
        proc = create_price_process(cfg, seed=0)
        assert isinstance(proc, LogNormalRandomWalk)

    def test_unknown_type_raises(self) -> None:
        cfg = OmegaConf.create({"type": "unknown"})
        with pytest.raises(ValueError, match="Unknown price process type"):
            create_price_process(cfg)
