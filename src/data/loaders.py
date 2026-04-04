"""Data loaders for raw Ethereum CSV files from BigQuery exports.

Loads block-level fee data and per-rollup blob transaction data,
cleans them, and optionally caches as Parquet in data/processed/.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEI_PER_ETH: float = 1e18
"""Conversion factor from wei to ETH."""

DENCUN_BLOCK: int = 19_426_589
"""First block with EIP-4844 blob support (March 13, 2024)."""

# EIP-4844 blob base fee parameters (for computing from excess_blob_gas)
MIN_BASE_FEE_PER_BLOB_GAS: int = 1  # 1 wei minimum
BLOB_BASE_FEE_UPDATE_FRACTION: int = 3_338_477  # per EIP-4844


# ---------------------------------------------------------------------------
# Block-level fee data
# ---------------------------------------------------------------------------


def load_block_fees(
    raw_dir: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Load and clean the block-level fee time series.

    Reads ``block_fees.csv`` from *raw_dir*, drops nulls and duplicates,
    converts L1 base fee from wei to ETH (to match the simulation convention
    in ``configs/base.yaml``), and optionally caches the result as Parquet.

    Args:
        raw_dir: Directory containing ``block_fees.csv``.
        start_date: Optional start date filter (ISO format, inclusive).
        end_date: Optional end date filter (ISO format, inclusive).
        processed_dir: If given, save cleaned data to
            ``processed_dir/block_fees.parquet``.

    Returns:
        DataFrame with columns:
            - ``block_number`` (int64)
            - ``timestamp`` (datetime64[ns, UTC])
            - ``base_fee_per_gas`` (float64, in ETH)
            - ``blob_base_fee`` (float64, in wei)
            - ``excess_blob_gas`` (int64)
            - ``blob_gas_used`` (int64)
        Sorted by ``block_number``, no duplicates, no nulls on fee columns.
    """
    csv_path = raw_dir / "block_fees.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run src/data/fetch.py first to download "
            "block-level fee data from BigQuery."
        )

    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        dtype={
            "block_number": "int64",
            "excess_blob_gas": "Int64",   # nullable int (NA-safe)
            "blob_gas_used": "Int64",     # nullable int (NA-safe)
        },
    )

    # --- Clean ---
    n_before = len(df)
    df = df.dropna(subset=["base_fee_per_gas", "excess_blob_gas"])
    n_nulls = n_before - len(df)
    if n_nulls:
        logger.info("Dropped %d rows with null fee values.", n_nulls)

    df = df.drop_duplicates(subset=["block_number"], keep="first")
    df = df.sort_values("block_number").reset_index(drop=True)

    # --- Date filter ---
    if start_date is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

    # --- Compute blob base fee from excess_blob_gas (EIP-4844 formula) ---
    # blob_base_fee = MIN_BASE_FEE * exp(excess_blob_gas / UPDATE_FRACTION)
    # If blob_base_fee column already exists in the CSV (e.g. from test data),
    # keep it; otherwise compute it.
    if "blob_base_fee" not in df.columns:
        df["blob_base_fee"] = df["excess_blob_gas"].apply(
            lambda x: float(
                MIN_BASE_FEE_PER_BLOB_GAS
                * math.exp(int(x) / BLOB_BASE_FEE_UPDATE_FRACTION)
            )
        )

    # --- Unit conversion ---
    # L1 gas base fee: wei -> ETH (config convention: mu ~ 3.86e-8 ETH)
    df["base_fee_per_gas"] = df["base_fee_per_gas"].astype(float) / WEI_PER_ETH

    # Blob base fee: keep in wei (config convention: mu ~ 1.0 wei at floor)
    df["blob_base_fee"] = df["blob_base_fee"].astype(float)

    # --- Sanity checks ---
    _check_block_gaps(df["block_number"].values)

    gas_median = df["base_fee_per_gas"].median()
    if gas_median < 1e-12 or gas_median > 1e-4:
        logger.warning(
            "Median gas base fee (%.2e ETH) looks unusual. Check units.",
            gas_median,
        )

    # --- Cache ---
    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / "block_fees.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df), out_path)

    if len(df) > 0:
        logger.info(
            "Loaded %d blocks (%s to %s).",
            len(df),
            df["timestamp"].iloc[0].date(),
            df["timestamp"].iloc[-1].date(),
        )
    else:
        logger.warning("No blocks remaining after filtering.")
    return df


# ---------------------------------------------------------------------------
# Rollup blob transaction data
# ---------------------------------------------------------------------------


def load_rollup_blobs(
    raw_dir: Path,
    labels: dict[str, str] | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Load per-transaction blob data and apply rollup labels.

    Args:
        raw_dir: Directory containing ``rollup_blobs.csv``.
        labels: Optional mapping of lowercase address -> rollup name.
            If *None*, uses :data:`src.data.rollup_labels.ROLLUP_LABELS`.
        start_date: Optional start date filter (ISO format).
        end_date: Optional end date filter (ISO format).
        processed_dir: If given, cache as Parquet.

    Returns:
        DataFrame with columns:
            - ``block_number`` (int64)
            - ``from_address`` (str, lowercase)
            - ``blob_count`` (int64)
            - ``max_fee_per_blob_gas`` (float64)
            - ``rollup_name`` (str, or ``"unknown"``)
    """
    csv_path = raw_dir / "rollup_blobs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run src/data/fetch.py first to download "
            "rollup blob transaction data from BigQuery."
        )

    df = pd.read_csv(csv_path, dtype={"block_number": "int64", "blob_count": "int64"})

    df["from_address"] = df["from_address"].str.lower()

    # --- Labels ---
    if labels is None:
        from src.data.rollup_labels import ROLLUP_LABELS

        labels = ROLLUP_LABELS

    df["rollup_name"] = df["from_address"].map(labels).fillna("unknown")

    n_unknown = (df["rollup_name"] == "unknown").sum()
    if n_unknown:
        logger.info(
            "%d blob txs (%.1f%%) from unlabeled addresses.",
            n_unknown,
            100 * n_unknown / len(df),
        )

    # --- Cache ---
    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / "rollup_blobs.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df), out_path)

    logger.info("Loaded %d blob transactions.", len(df))
    return df


# ---------------------------------------------------------------------------
# Arrival rate computation
# ---------------------------------------------------------------------------


def compute_rollup_arrival_rates(
    blobs_df: pd.DataFrame,
    total_blocks: int | None = None,
) -> pd.DataFrame:
    """Compute per-rollup blob posting frequency (lambda_i).

    For each labeled rollup, counts total blob-carrying transactions and
    divides by the number of blocks in the window to get a rate
    (blob transactions per block).

    Args:
        blobs_df: DataFrame from :func:`load_rollup_blobs` (must have
            ``rollup_name``, ``block_number``, ``blob_count`` columns).
        total_blocks: Total blocks in the calibration window. If *None*,
            inferred from min/max ``block_number`` in the data.

    Returns:
        DataFrame with columns:
            - ``rollup_name`` (str)
            - ``lambda_i`` (float64, blob txs per block)
            - ``mean_blobs_per_tx`` (float64)
            - ``n_transactions`` (int64)
        Sorted by ``lambda_i`` descending, excluding ``"unknown"``.
    """
    labeled = blobs_df[blobs_df["rollup_name"] != "unknown"].copy()

    if total_blocks is None:
        block_min = blobs_df["block_number"].min()
        block_max = blobs_df["block_number"].max()
        total_blocks = int(block_max - block_min + 1)

    rates = (
        labeled.groupby("rollup_name")
        .agg(
            n_transactions=("block_number", "count"),
            mean_blobs_per_tx=("blob_count", "mean"),
        )
        .reset_index()
    )
    rates["lambda_i"] = rates["n_transactions"] / total_blocks
    rates = rates.sort_values("lambda_i", ascending=False).reset_index(drop=True)

    logger.info(
        "Computed arrival rates for %d rollups over %d blocks.",
        len(rates),
        total_blocks,
    )
    return rates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_block_gaps(block_numbers: np.ndarray, max_gap_pct: float = 1.0) -> None:
    """Warn if more than *max_gap_pct* % of expected blocks are missing."""
    if len(block_numbers) < 2:
        return
    expected = int(block_numbers[-1] - block_numbers[0] + 1)
    actual = len(block_numbers)
    missing_pct = 100 * (1 - actual / expected)
    if missing_pct > max_gap_pct:
        logger.warning(
            "%.1f%% of expected blocks are missing (%d of %d). "
            "Missed slots are normal on Ethereum but check for data gaps.",
            missing_pct,
            expected - actual,
            expected,
        )
