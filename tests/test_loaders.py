"""Tests for src.data.loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """Create a minimal data/raw/ directory with test CSVs."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # --- block_fees.csv ---
    blocks = pd.DataFrame(
        {
            "block_number": [100, 101, 102, 103, 104],
            "timestamp": pd.date_range("2024-04-01", periods=5, freq="12s", tz="UTC"),
            "base_fee_per_gas": [
                30e9,  # 30 Gwei in wei
                35e9,
                32e9,
                np.nan,  # null row to test dropping
                28e9,
            ],
            "blob_base_fee": [1.0, 2.0, 1.5, np.nan, 3.0],
            "excess_blob_gas": [0, 131072, 0, 0, 262144],
            "blob_gas_used": [0, 393216, 131072, 0, 524288],
        }
    )
    blocks.to_csv(raw / "block_fees.csv", index=False)

    # --- rollup_blobs.csv ---
    blobs = pd.DataFrame(
        {
            "block_number": [100, 101, 101, 102, 104, 104],
            "from_address": [
                "0xaaa",
                "0xaaa",
                "0xbbb",
                "0xbbb",
                "0xaaa",
                "0xccc",
            ],
            "blob_count": [3, 2, 1, 4, 1, 2],
            "max_fee_per_blob_gas": [100, 200, 150, 300, 120, 250],
        }
    )
    blobs.to_csv(raw / "rollup_blobs.csv", index=False)

    return raw


@pytest.fixture()
def labels() -> dict[str, str]:
    """Rollup address labels for testing."""
    return {
        "0xaaa": "rollup_alpha",
        "0xbbb": "rollup_beta",
    }


class TestLoadBlockFees:
    """Tests for load_block_fees."""

    def test_happy_path(self, raw_dir: Path) -> None:
        from src.data.loaders import load_block_fees

        df = load_block_fees(raw_dir)

        # Null row dropped → 4 rows
        assert len(df) == 4
        assert list(df.columns) == [
            "block_number",
            "timestamp",
            "base_fee_per_gas",
            "blob_base_fee",
            "excess_blob_gas",
            "blob_gas_used",
        ]
        # Sorted by block_number
        assert df["block_number"].is_monotonic_increasing

    def test_unit_conversion(self, raw_dir: Path) -> None:
        from src.data.loaders import load_block_fees

        df = load_block_fees(raw_dir)

        # 30 Gwei = 30e9 wei = 30e9 / 1e18 = 3e-8 ETH
        first_gas = df.iloc[0]["base_fee_per_gas"]
        assert pytest.approx(first_gas, rel=1e-6) == 3e-8

        # Blob base fee stays in wei
        first_blob = df.iloc[0]["blob_base_fee"]
        assert pytest.approx(first_blob) == 1.0

    def test_drops_nulls_and_dupes(self, raw_dir: Path) -> None:
        from src.data.loaders import load_block_fees

        # Add a duplicate block_number
        csv_path = raw_dir / "block_fees.csv"
        df_raw = pd.read_csv(csv_path)
        dup_row = df_raw.iloc[[0]].copy()
        df_raw = pd.concat([df_raw, dup_row], ignore_index=True)
        df_raw.to_csv(csv_path, index=False)

        df = load_block_fees(raw_dir)

        # Still 4 unique non-null rows
        assert len(df) == 4
        assert df["block_number"].is_unique

    def test_date_filter(self, raw_dir: Path) -> None:
        from src.data.loaders import load_block_fees

        # All valid rows have timestamps within ~1 minute of 2024-04-01 00:00
        df = load_block_fees(raw_dir, start_date="2024-03-31", end_date="2024-04-02")
        assert len(df) == 4

        # Restrictive filter: no rows after 2024-04-02
        df2 = load_block_fees(raw_dir, start_date="2024-04-02")
        assert len(df2) == 0

    def test_parquet_cache(self, raw_dir: Path, tmp_path: Path) -> None:
        from src.data.loaders import load_block_fees

        processed = tmp_path / "processed"
        df = load_block_fees(raw_dir, processed_dir=processed)

        parquet_path = processed / "block_fees.parquet"
        assert parquet_path.exists()

        df_pq = pd.read_parquet(parquet_path)
        assert len(df_pq) == len(df)


class TestLoadRollupBlobs:
    """Tests for load_rollup_blobs."""

    def test_labels_applied(self, raw_dir: Path, labels: dict) -> None:
        from src.data.loaders import load_rollup_blobs

        df = load_rollup_blobs(raw_dir, labels=labels)

        assert "rollup_name" in df.columns
        labeled = df[df["rollup_name"] != "unknown"]
        assert len(labeled) == 5  # 3 from 0xaaa + 2 from 0xbbb
        assert set(labeled["rollup_name"]) == {"rollup_alpha", "rollup_beta"}

    def test_unknown_addresses(self, raw_dir: Path, labels: dict) -> None:
        from src.data.loaders import load_rollup_blobs

        df = load_rollup_blobs(raw_dir, labels=labels)

        unknown = df[df["rollup_name"] == "unknown"]
        assert len(unknown) == 1  # 0xccc is not in labels


class TestComputeArrivalRates:
    """Tests for compute_rollup_arrival_rates."""

    def test_rates_correct(self, raw_dir: Path, labels: dict) -> None:
        from src.data.loaders import compute_rollup_arrival_rates, load_rollup_blobs

        blobs_df = load_rollup_blobs(raw_dir, labels=labels)

        # 5 blocks total (100–104)
        rates = compute_rollup_arrival_rates(blobs_df, total_blocks=5)

        assert "lambda_i" in rates.columns
        assert "unknown" not in rates["rollup_name"].values

        alpha_rate = rates[rates["rollup_name"] == "rollup_alpha"]["lambda_i"].iloc[0]
        beta_rate = rates[rates["rollup_name"] == "rollup_beta"]["lambda_i"].iloc[0]

        # rollup_alpha: 3 txs in 5 blocks
        assert pytest.approx(alpha_rate) == 3 / 5
        # rollup_beta: 2 txs in 5 blocks
        assert pytest.approx(beta_rate) == 2 / 5

    def test_sorted_descending(self, raw_dir: Path, labels: dict) -> None:
        from src.data.loaders import compute_rollup_arrival_rates, load_rollup_blobs

        blobs_df = load_rollup_blobs(raw_dir, labels=labels)
        rates = compute_rollup_arrival_rates(blobs_df, total_blocks=5)

        assert rates["lambda_i"].is_monotonic_decreasing
