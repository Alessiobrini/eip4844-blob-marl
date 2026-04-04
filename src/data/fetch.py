"""Download Ethereum block and blob transaction data from Google BigQuery.

Saves raw CSV files to ``data/raw/`` for use by the calibration pipeline.

Prerequisites:
    1. A Google Cloud account (free tier is sufficient).
    2. ``gcloud auth application-default login`` for credentials.
    3. ``pip install google-cloud-bigquery`` (or conda).

Usage::

    # Dry run (show estimated bytes, no cost)
    python src/data/fetch.py --output data/raw/ --dry-run

    # Full download (March 2024 – March 2025)
    python src/data/fetch.py --output data/raw/

    # Custom date range
    python src/data/fetch.py --output data/raw/ --start 2024-03-13 --end 2024-08-31

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Block number constants (approximate, for reference)
# ---------------------------------------------------------------------------

DENCUN_BLOCK = 19_426_589  # March 13, 2024 — first block with blob support


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

BLOCK_FEES_QUERY = """\
SELECT
  number AS block_number,
  timestamp,
  base_fee_per_gas,
  excess_blob_gas,
  blob_gas_used
FROM `bigquery-public-data.crypto_ethereum.blocks`
WHERE timestamp BETWEEN '{start}' AND '{end}'
ORDER BY number
"""
# Note: blob_base_fee is not a direct column in BigQuery's crypto_ethereum.blocks.
# It is computed in loaders.py from excess_blob_gas using the EIP-4844 formula:
#   blob_base_fee = MIN_BASE_FEE_PER_BLOB_GAS * exp(excess_blob_gas / BLOB_BASE_FEE_UPDATE_FRACTION)
# Alternatively, receipt_blob_gas_price on transactions gives the per-tx blob base fee.

ROLLUP_BLOBS_QUERY = """\
SELECT
  block_number,
  from_address,
  ARRAY_LENGTH(blob_versioned_hashes) AS blob_count,
  max_fee_per_blob_gas,
  receipt_blob_gas_price
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE transaction_type = 3
  AND block_timestamp BETWEEN '{start}' AND '{end}'
ORDER BY block_number
"""


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------


def _run_query(
    client: "google.cloud.bigquery.Client",
    sql: str,
    *,
    dry_run: bool = False,
    label: str = "query",
) -> "pandas.DataFrame | None":
    """Execute a BigQuery query, optionally as a dry run.

    Args:
        client: Authenticated BigQuery client.
        sql: SQL query string.
        dry_run: If True, estimate bytes only (no cost).
        label: Human-readable label for logging.

    Returns:
        DataFrame with query results, or None if dry_run.
    """
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(dry_run=dry_run, use_query_cache=True)
    query_job = client.query(sql, job_config=job_config)

    if dry_run:
        bytes_est = query_job.total_bytes_processed
        gb_est = bytes_est / (1024**3)
        cost_est = max(0, (gb_est - 1024) * 5 / 1024)  # $5/TB after 1TB free
        logger.info(
            "[%s] Estimated scan: %.2f GB (est. cost: $%.4f, free if < 1 TB/month total)",
            label,
            gb_est,
            cost_est,
        )
        return None

    logger.info("[%s] Running query...", label)
    df = query_job.to_dataframe()
    bytes_proc = query_job.total_bytes_processed
    logger.info(
        "[%s] Done. %d rows, %.2f GB scanned.",
        label,
        len(df),
        bytes_proc / (1024**3),
    )
    return df


def fetch_bigquery(
    output_dir: Path,
    *,
    start: str = "2024-03-13",
    end: str = "2025-03-31",
    dry_run: bool = False,
    project: str | None = None,
) -> None:
    """Download block fees and rollup blob data from BigQuery.

    Args:
        output_dir: Directory to write CSV files to (created if needed).
        start: Start date (ISO format, inclusive).
        end: End date (ISO format, inclusive).
        dry_run: If True, only estimate query costs.
        project: GCP project ID. If None, auto-detected from environment.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Query 1: Block-level fees ---
    block_sql = BLOCK_FEES_QUERY.format(start=start, end=end)
    block_df = _run_query(client, block_sql, dry_run=dry_run, label="block_fees")

    if block_df is not None:
        out_path = output_dir / "block_fees.csv"
        block_df.to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / (1024**2)
        logger.info("Saved %d rows to %s (%.1f MB)", len(block_df), out_path, size_mb)

    # --- Query 2: Rollup blob transactions ---
    blob_sql = ROLLUP_BLOBS_QUERY.format(start=start, end=end)
    blob_df = _run_query(client, blob_sql, dry_run=dry_run, label="rollup_blobs")

    if blob_df is not None:
        out_path = output_dir / "rollup_blobs.csv"
        blob_df.to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / (1024**2)
        logger.info("Saved %d rows to %s (%.1f MB)", len(blob_df), out_path, size_mb)

    if dry_run:
        print("\nDry run complete. No data was downloaded or charged.")
        print("Run without --dry-run to download.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for BigQuery data fetch."""
    parser = argparse.ArgumentParser(
        description="Download Ethereum block and blob data from BigQuery."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
        help="Output directory for CSV files (default: data/raw/).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-03-13",
        help="Start date, ISO format (default: 2024-03-13, Dencun activation).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-03-31",
        help="End date, ISO format (default: 2025-03-31).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate query cost without downloading data.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="GCP project ID (auto-detected if not set).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        fetch_bigquery(
            output_dir=args.output,
            start=args.start,
            end=args.end,
            dry_run=args.dry_run,
            project=args.project,
        )
    except ImportError:
        logger.error(
            "google-cloud-bigquery is not installed. Install it with:\n"
            "  pip install google-cloud-bigquery\n"
            "Then authenticate with:\n"
            "  gcloud auth application-default login"
        )
        raise SystemExit(1)
    except Exception as exc:
        logger.error("BigQuery fetch failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
