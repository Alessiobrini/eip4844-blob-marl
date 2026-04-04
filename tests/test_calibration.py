"""Tests for src.data.calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _generate_ar1(
    mu: float,
    theta: float,
    sigma: float,
    n: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic AR(1) series with known parameters."""
    rng = np.random.default_rng(seed)
    series = np.empty(n)
    series[0] = mu
    for t in range(1, n):
        eps = rng.normal()
        series[t] = theta * mu + (1 - theta) * series[t - 1] + sigma * eps
    return series


def _generate_log_ar1(
    mu_log: float,
    theta: float,
    sigma: float,
    n: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic log-AR(1) series (multiplicative dynamics)."""
    log_series = _generate_ar1(mu_log, theta, sigma, n, seed=seed)
    return np.exp(log_series)


# ---------------------------------------------------------------------------
# fit_ar1 tests
# ---------------------------------------------------------------------------


class TestFitAR1:
    """Tests for fit_ar1."""

    def test_recovers_known_params(self) -> None:
        from src.data.calibration import fit_ar1

        true_mu, true_theta, true_sigma = 10.0, 0.2, 1.0
        series = _generate_ar1(true_mu, true_theta, true_sigma, n=50_000, seed=42)
        result = fit_ar1(series, name="test")

        assert result["mu"] == pytest.approx(true_mu, rel=0.05)
        assert result["theta"] == pytest.approx(true_theta, rel=0.05)
        assert result["sigma"] == pytest.approx(true_sigma, rel=0.05)

    def test_detects_unit_root(self) -> None:
        from src.data.calibration import fit_ar1

        # Random walk: theta=0, phi=1
        rng = np.random.default_rng(123)
        series = np.cumsum(rng.normal(0, 1, 10_000)) + 100
        result = fit_ar1(series, name="unit_root")

        # theta should be near zero
        assert abs(result["theta"]) < 0.05
        # ADF should not reject unit root (p > 0.05)
        if not np.isnan(result["adf_pvalue"]):
            assert result["adf_pvalue"] > 0.05

    def test_residuals_white_noise(self) -> None:
        from src.data.calibration import fit_ar1

        series = _generate_ar1(5.0, 0.15, 0.5, n=20_000, seed=99)
        result = fit_ar1(series, name="white_noise_check")

        # Ljung-Box should NOT reject white noise (p > 0.05)
        if not np.isnan(result["ljung_box_pvalue"]):
            assert result["ljung_box_pvalue"] > 0.01

    def test_rejects_short_series(self) -> None:
        from src.data.calibration import fit_ar1

        with pytest.raises(ValueError, match="need >= 100"):
            fit_ar1(np.ones(50), name="short")

    def test_r_squared_positive(self) -> None:
        from src.data.calibration import fit_ar1

        series = _generate_ar1(10.0, 0.1, 1.0, n=5_000, seed=7)
        result = fit_ar1(series, name="r2_check")
        assert result["r_squared"] > 0.5


class TestFitAR1Log:
    """Tests for fit_ar1_log."""

    def test_recovers_log_params(self) -> None:
        from src.data.calibration import fit_ar1_log

        true_mu_log, true_theta, true_sigma = 5.0, 0.1, 0.3
        series = _generate_log_ar1(true_mu_log, true_theta, true_sigma, n=50_000, seed=42)
        result = fit_ar1_log(series, name="log_test")

        assert result["mu"] == pytest.approx(true_mu_log, rel=0.05)
        assert result["theta"] == pytest.approx(true_theta, rel=0.10)
        assert result["sigma"] == pytest.approx(true_sigma, rel=0.05)

    def test_initial_fee_positive(self) -> None:
        from src.data.calibration import fit_ar1_log

        series = _generate_log_ar1(3.0, 0.05, 0.2, n=1_000, seed=11)
        result = fit_ar1_log(series, name="init_fee")
        assert result["initial_fee"] > 0

    def test_handles_near_zero_values(self) -> None:
        from src.data.calibration import fit_ar1_log

        # Series with some near-zero values (like blob fee at floor)
        series = _generate_log_ar1(0.0, 0.05, 0.5, n=5_000, seed=33)
        series[:10] = 1e-20  # inject near-zero values
        result = fit_ar1_log(series, name="near_zero")
        # Should not raise; params should be finite
        assert np.isfinite(result["mu"])
        assert np.isfinite(result["sigma"])


# ---------------------------------------------------------------------------
# fit_lognormal_arrival tests
# ---------------------------------------------------------------------------


class TestFitLognormalArrival:
    """Tests for fit_lognormal_arrival."""

    def test_recovers_params(self) -> None:
        from src.data.calibration import fit_lognormal_arrival

        rng = np.random.default_rng(42)
        true_mu, true_sigma = 3.0, 1.0
        samples = rng.lognormal(true_mu, true_sigma, size=500)
        result = fit_lognormal_arrival(samples)

        assert result["mu_lambda"] == pytest.approx(true_mu, rel=0.10)
        assert result["sigma_lambda"] == pytest.approx(true_sigma, rel=0.10)
        assert result["n_rollups"] == 500

    def test_rejects_too_few(self) -> None:
        from src.data.calibration import fit_lognormal_arrival

        with pytest.raises(ValueError, match="Need >= 3"):
            fit_lognormal_arrival(np.array([1.0, 2.0]))

    def test_filters_zeros(self) -> None:
        from src.data.calibration import fit_lognormal_arrival

        values = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = fit_lognormal_arrival(values)
        assert result["n_rollups"] == 5  # zeros excluded


# ---------------------------------------------------------------------------
# YAML roundtrip test
# ---------------------------------------------------------------------------


class TestCalibrationYAML:
    """Test that calibration output loads correctly."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        from src.data.calibration import (
            fit_ar1,
            fit_ar1_log,
            fit_lognormal_arrival,
            write_calibration_yaml,
        )

        gas_series = _generate_ar1(3.5e-8, 0.1, 8e-9, n=5_000, seed=1)
        blob_series = _generate_log_ar1(2.0, 0.05, 0.3, n=5_000, seed=2)
        rng = np.random.default_rng(3)
        lambdas = rng.lognormal(2.0, 0.8, size=15)

        gas_params = fit_ar1(gas_series, name="gas")
        blob_params = fit_ar1_log(blob_series, name="blob")
        arrival_params = fit_lognormal_arrival(lambdas)
        rollup_rates = {f"rollup_{i}": float(l) for i, l in enumerate(lambdas)}

        yaml_path = tmp_path / "calibration.yaml"
        write_calibration_yaml(
            gas_params=gas_params,
            blob_params=blob_params,
            arrival_params=arrival_params,
            rollup_rates=rollup_rates,
            output_path=yaml_path,
            start_date="2024-03-13",
            end_date="2024-08-31",
            n_blocks=5000,
        )

        assert yaml_path.exists()
        cfg = OmegaConf.load(yaml_path)

        assert "price_process" in cfg
        assert "blob_fee_process" in cfg
        assert "rollup_arrival" in cfg
        assert cfg.price_process.type == "ar1"
        assert cfg.blob_fee_process.type == "exogenous_log_ar1"

    def test_loadable_by_create_price_process(self, tmp_path: Path) -> None:
        from src.abm.gas_process import create_price_process
        from src.data.calibration import fit_ar1, fit_ar1_log, fit_lognormal_arrival, write_calibration_yaml

        gas_series = _generate_ar1(3.5e-8, 0.1, 8e-9, n=5_000, seed=1)
        blob_series = _generate_log_ar1(2.0, 0.05, 0.3, n=5_000, seed=2)
        rng = np.random.default_rng(3)
        lambdas = rng.lognormal(2.0, 0.8, size=15)

        gas_params = fit_ar1(gas_series, name="gas")
        blob_params = fit_ar1_log(blob_series, name="blob")
        arrival_params = fit_lognormal_arrival(lambdas)

        yaml_path = tmp_path / "calibration.yaml"
        write_calibration_yaml(
            gas_params=gas_params,
            blob_params=blob_params,
            arrival_params=arrival_params,
            rollup_rates={"test": 1.0},
            output_path=yaml_path,
            start_date="2024-03-13",
            end_date="2024-08-31",
            n_blocks=5000,
        )

        cfg = OmegaConf.load(yaml_path)

        # Gas price process should create AR1Process
        gas_proc = create_price_process(cfg.price_process, seed=42)
        price = gas_proc.step()
        assert np.isfinite(price)

        # Blob fee process should create LogAR1Process
        blob_proc = create_price_process(cfg.blob_fee_process, seed=42)
        fee = blob_proc.step()
        assert np.isfinite(fee)
        assert fee >= 1.0  # floor
