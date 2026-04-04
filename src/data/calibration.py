"""Calibrate AR(1) price processes and rollup arrival rate distributions.

Reads cleaned Ethereum block data via :mod:`src.data.loaders`, fits
statistical models, and writes ``configs/calibration.yaml``.

Usage::

    python src/data/calibration.py --raw-dir data/raw/ --output configs/calibration.yaml
    python src/data/calibration.py --raw-dir data/raw/ --output configs/calibration.yaml \\
        --start 2024-03-13 --end 2024-08-31

"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AR(1) fitting
# ---------------------------------------------------------------------------


def fit_ar1(series: np.ndarray, *, name: str = "series") -> dict:
    """Fit AR(1) parameters via OLS in level space.

    Model: ``P_t = c + phi * P_{t-1} + e_t``

    Mapping to simulation parameters
    (:class:`src.abm.gas_process.AR1Process`):

    - ``theta = 1 - phi``
    - ``mu = c / theta``
    - ``sigma = std(residuals)``

    Args:
        series: 1-D array of price observations (one per block).
        name: Label for log messages.

    Returns:
        Dict with keys ``mu``, ``theta``, ``sigma``, ``phi``,
        ``r_squared``, ``half_life_blocks``, ``adf_pvalue``,
        ``ljung_box_pvalue``.
    """
    series = np.asarray(series, dtype=np.float64)
    if len(series) < 100:
        raise ValueError(f"{name}: need >= 100 observations, got {len(series)}.")

    y = series[1:]
    x = series[:-1]

    # OLS: y = c + phi * x + e
    X = np.column_stack([np.ones_like(x), x])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    c, phi = coeffs

    e = y - (c + phi * x)
    sigma = float(np.std(e, ddof=2))

    theta = 1.0 - phi
    if abs(theta) < 1e-12:
        mu = float(np.mean(series))
        logger.warning("%s: theta ≈ 0 (unit root). Setting mu to sample mean.", name)
    else:
        mu = c / theta

    # R-squared
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Half-life of mean reversion
    if 0 < abs(phi) < 1:
        half_life = math.log(0.5) / math.log(abs(phi))
    else:
        half_life = float("inf")

    # Augmented Dickey-Fuller test
    adf_pvalue = _adf_pvalue(series)

    # Ljung-Box test on residuals (lag 10)
    lb_pvalue = _ljung_box_pvalue(e, nlags=10)

    result = {
        "mu": float(mu),
        "theta": float(theta),
        "sigma": float(sigma),
        "phi": float(phi),
        "r_squared": float(r_squared),
        "half_life_blocks": float(half_life),
        "adf_pvalue": float(adf_pvalue),
        "ljung_box_pvalue": float(lb_pvalue),
    }

    _log_ar1_diagnostics(result, name)
    return result


def fit_ar1_log(series: np.ndarray, *, name: str = "series") -> dict:
    """Fit AR(1) in log-space for multiplicative processes.

    Applies ``log(series)`` and delegates to :func:`fit_ar1`. The resulting
    parameters describe the process::

        log(P_{t+1}) = theta * mu + (1 - theta) * log(P_t) + sigma * eps

    which, when exponentiated, gives multiplicative dynamics matching the
    EIP-4844 blob fee update rule.

    Args:
        series: 1-D array of *positive* price observations.
        name: Label for log messages.

    Returns:
        Same keys as :func:`fit_ar1`, with values in log-space.
        Additional key ``initial_fee`` = exp(mean of first 100 log-prices).
    """
    series = np.asarray(series, dtype=np.float64)
    if np.any(series <= 0):
        n_neg = int(np.sum(series <= 0))
        logger.warning(
            "%s: %d non-positive values found. Clipping to 1e-18 before log.",
            name,
            n_neg,
        )
        series = np.clip(series, 1e-18, None)

    log_series = np.log(series)
    result = fit_ar1(log_series, name=f"{name} (log)")

    # Initial fee for the simulation (geometric mean of first 100 blocks)
    n_init = min(100, len(log_series))
    result["initial_fee"] = float(np.exp(np.mean(log_series[:n_init])))

    return result


# ---------------------------------------------------------------------------
# LogNormal arrival rate fitting
# ---------------------------------------------------------------------------


def fit_lognormal_arrival(lambda_values: np.ndarray) -> dict:
    """Fit a LogNormal distribution to per-rollup arrival rates.

    Args:
        lambda_values: 1-D array of lambda_i values (one per rollup,
            must be positive).

    Returns:
        Dict with keys ``mu_lambda``, ``sigma_lambda``, ``n_rollups``,
        ``ks_pvalue``.
    """
    lv = np.asarray(lambda_values, dtype=np.float64)
    lv = lv[lv > 0]  # drop zero-rate rollups

    if len(lv) < 3:
        raise ValueError(
            f"Need >= 3 positive arrival rates for LogNormal fit, got {len(lv)}."
        )

    if len(lv) < 10:
        logger.warning(
            "Only %d rollups — LogNormal fit has wide confidence intervals.",
            len(lv),
        )

    log_lv = np.log(lv)
    mu_lambda = float(np.mean(log_lv))
    sigma_lambda = float(np.std(log_lv, ddof=1))

    # KS goodness-of-fit test
    ks_stat, ks_pvalue = sp_stats.kstest(
        lv, "lognorm", args=(sigma_lambda, 0, np.exp(mu_lambda))
    )

    return {
        "mu_lambda": mu_lambda,
        "sigma_lambda": sigma_lambda,
        "n_rollups": int(len(lv)),
        "ks_pvalue": float(ks_pvalue),
    }


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def write_calibration_yaml(
    gas_params: dict,
    blob_params: dict,
    arrival_params: dict,
    rollup_rates: dict[str, float],
    output_path: Path,
    *,
    start_date: str,
    end_date: str,
    n_blocks: int,
) -> None:
    """Write calibration results to a YAML config file.

    Args:
        gas_params: Output of :func:`fit_ar1` for L1 gas base fee.
        blob_params: Output of :func:`fit_ar1_log` for blob base fee.
        arrival_params: Output of :func:`fit_lognormal_arrival`.
        rollup_rates: Dict of rollup_name -> lambda_i.
        output_path: Path to write (e.g. ``configs/calibration.yaml``).
        start_date: Calibration window start (ISO date string).
        end_date: Calibration window end (ISO date string).
        n_blocks: Number of blocks in the calibration window.
    """
    # Initial gas price: mean of the series (already in mu)
    n_init = 100  # used for initial_price derivation
    cal = {
        "calibration_window": {
            "start_date": start_date,
            "end_date": end_date,
            "n_blocks": n_blocks,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "price_process": {
            "type": "ar1",
            "mu": gas_params["mu"],
            "theta": gas_params["theta"],
            "sigma": gas_params["sigma"],
            "initial_price": gas_params["mu"],
            "floor": 0.0,
            "r_squared": gas_params["r_squared"],
            "adf_pvalue": gas_params["adf_pvalue"],
            "half_life_blocks": gas_params["half_life_blocks"],
            "ljung_box_pvalue": gas_params["ljung_box_pvalue"],
        },
        "blob_fee_process": {
            "type": "exogenous_log_ar1",
            "mu": blob_params["mu"],
            "theta": blob_params["theta"],
            "sigma": blob_params["sigma"],
            "initial_fee": blob_params["initial_fee"],
            "floor": 1.0,
            "r_squared": blob_params["r_squared"],
            "adf_pvalue": blob_params["adf_pvalue"],
            "half_life_blocks": blob_params["half_life_blocks"],
            "ljung_box_pvalue": blob_params["ljung_box_pvalue"],
        },
        "rollup_arrival": {
            "mu_lambda": arrival_params["mu_lambda"],
            "sigma_lambda": arrival_params["sigma_lambda"],
            "n_rollups": arrival_params["n_rollups"],
            "ks_pvalue": arrival_params["ks_pvalue"],
            "rollup_rates": rollup_rates,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conf = OmegaConf.create(cal)
    OmegaConf.save(conf, output_path)
    logger.info("Wrote calibration config to %s", output_path)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def _adf_pvalue(series: np.ndarray) -> float:
    """Run ADF test and return p-value. Falls back to NaN if statsmodels missing."""
    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series, maxlag=20, autolag="AIC")
        return float(result[1])
    except ImportError:
        logger.warning("statsmodels not installed — skipping ADF test.")
        return float("nan")


def _ljung_box_pvalue(residuals: np.ndarray, nlags: int = 10) -> float:
    """Run Ljung-Box test and return minimum p-value across lags."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(residuals, lags=nlags, return_df=True)
        return float(lb["lb_pvalue"].min())
    except ImportError:
        logger.warning("statsmodels not installed — skipping Ljung-Box test.")
        return float("nan")


def _log_ar1_diagnostics(params: dict, name: str) -> None:
    """Log key diagnostics for an AR(1) fit."""
    logger.info(
        "%s AR(1): mu=%.4e, theta=%.4f, sigma=%.4e, R²=%.4f, "
        "half-life=%.1f blocks, ADF p=%.4f",
        name,
        params["mu"],
        params["theta"],
        params["sigma"],
        params["r_squared"],
        params["half_life_blocks"],
        params["adf_pvalue"],
    )

    if abs(params["phi"]) >= 1.0:
        logger.warning("%s: |phi|=%.4f >= 1 — process is non-stationary!", name, abs(params["phi"]))
    if params["adf_pvalue"] > 0.05:
        logger.warning("%s: ADF p-value=%.4f > 0.05 — cannot reject unit root.", name, params["adf_pvalue"])
    if params["ljung_box_pvalue"] < 0.05:
        logger.warning(
            "%s: Ljung-Box p=%.4f < 0.05 — residuals show autocorrelation; "
            "AR(1) may be under-specified.",
            name,
            params["ljung_box_pvalue"],
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full calibration pipeline."""
    parser = argparse.ArgumentParser(
        description="Calibrate AR(1) price processes and rollup arrival rates."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
        help="Directory containing raw CSV files (default: data/raw/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "calibration.yaml",
        help="Output YAML path (default: configs/calibration.yaml).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-03-13",
        help="Calibration window start date (default: 2024-03-13).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-08-31",
        help="Calibration window end date (default: 2024-08-31).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.data.loaders import (
        compute_rollup_arrival_rates,
        load_block_fees,
        load_rollup_blobs,
    )

    processed_dir = args.raw_dir.parent / "processed"

    # --- Load data ---
    logger.info("Loading block fees from %s ...", args.raw_dir)
    block_fees = load_block_fees(
        args.raw_dir,
        start_date=args.start,
        end_date=args.end,
        processed_dir=processed_dir,
    )

    logger.info("Loading rollup blobs from %s ...", args.raw_dir)
    blobs = load_rollup_blobs(
        args.raw_dir,
        start_date=args.start,
        end_date=args.end,
        processed_dir=processed_dir,
    )

    # --- Fit AR(1) for gas base fee ---
    gas_series = block_fees["base_fee_per_gas"].values
    logger.info("Fitting AR(1) for L1 gas base fee (%d observations)...", len(gas_series))
    gas_params = fit_ar1(gas_series, name="gas_base_fee")

    # --- Fit AR(1) in log-space for blob base fee ---
    blob_series = block_fees["blob_base_fee"].values
    logger.info("Fitting log-AR(1) for blob base fee (%d observations)...", len(blob_series))
    blob_params = fit_ar1_log(blob_series, name="blob_base_fee")

    # --- Fit rollup arrival rates ---
    rates_df = compute_rollup_arrival_rates(blobs, total_blocks=len(block_fees))
    lambda_values = rates_df["lambda_i"].values
    logger.info("Fitting LogNormal to %d rollup arrival rates...", len(lambda_values))
    arrival_params = fit_lognormal_arrival(lambda_values)

    rollup_rates = dict(zip(rates_df["rollup_name"], rates_df["lambda_i"]))

    # --- Write YAML ---
    write_calibration_yaml(
        gas_params=gas_params,
        blob_params=blob_params,
        arrival_params=arrival_params,
        rollup_rates=rollup_rates,
        output_path=args.output,
        start_date=args.start,
        end_date=args.end,
        n_blocks=len(block_fees),
    )

    # --- Summary ---
    print("\n=== Calibration Summary ===")
    print(f"Window: {args.start} to {args.end} ({len(block_fees)} blocks)")
    print(f"\nGas base fee AR(1) [ETH/gas]:")
    print(f"  mu    = {gas_params['mu']:.4e}")
    print(f"  theta = {gas_params['theta']:.6f}")
    print(f"  sigma = {gas_params['sigma']:.4e}")
    print(f"  half-life = {gas_params['half_life_blocks']:.0f} blocks")
    print(f"\nBlob base fee log-AR(1) [log-wei]:")
    print(f"  mu    = {blob_params['mu']:.4f}")
    print(f"  theta = {blob_params['theta']:.6f}")
    print(f"  sigma = {blob_params['sigma']:.4f}")
    print(f"  initial_fee = {blob_params['initial_fee']:.2f} wei")
    print(f"\nRollup arrival rates (top 5):")
    for _, row in rates_df.head(5).iterrows():
        print(f"  {row['rollup_name']:20s}  lambda={row['lambda_i']:.6f} tx/block")
    print(f"\nLogNormal fit: mu_lambda={arrival_params['mu_lambda']:.4f}, "
          f"sigma_lambda={arrival_params['sigma_lambda']:.4f}, "
          f"KS p={arrival_params['ks_pvalue']:.4f}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
