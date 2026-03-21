"""EIP-4844 blob base fee update rule and posting cost helpers.

The ``update_blob_base_fee`` function implements the canonical EIP-4844 rule.
DO NOT MODIFY that function.
"""

from __future__ import annotations

import math


def update_blob_base_fee(
    B_t: float, blobs_used: int, target: int = 3
) -> float:
    """Apply the EIP-4844 blob base fee update rule.

    Args:
        B_t: Current blob base fee in wei.
        blobs_used: Number of blobs included in the current block.
        target: Target blobs per block (default 3 per EIP-4844).

    Returns:
        Updated blob base fee for the next block.
    """
    return B_t * math.exp((blobs_used - target) / (8 * target))


def compute_posting_cost(
    blob_base_fee: float,
    gas_price: float,
    n_blobs: int,
    gas_fixed: int = 21_000,
    cost_mode: str = "linear_overhead",
) -> float:
    """Compute total cost of posting n_blobs in a single type-3 transaction.

    Args:
        blob_base_fee: Current blob base fee (wei/blob-gas or normalized).
        gas_price: Current L1 gas price (wei/gas or normalized).
        n_blobs: Number of blobs posted (1 to 6).
        gas_fixed: Fixed gas cost of a type-3 transaction (default 21,000).
        cost_mode: Cost calculation mode.
            - ``"bar_on_mansour"``: Simple ``gas_price * gas_fixed`` (price is
              the single scalar cost in Bar-On & Mansour's formulation).
            - ``"linear_overhead"``: ``blob_base_fee * n_blobs * 131072 +
              gas_price * gas_fixed`` (blob gas cost plus execution gas).

    Returns:
        Total posting cost (in wei or normalized units).

    Raises:
        ValueError: If ``cost_mode`` is not recognized.
    """
    if cost_mode == "bar_on_mansour":
        return gas_price * gas_fixed

    if cost_mode in ("linear_overhead", "quadratic_delay"):
        # 131072 = 2^17 gas units per blob in the blob gas dimension
        blob_gas_cost = blob_base_fee * n_blobs * 131_072
        execution_gas_cost = gas_price * gas_fixed
        return blob_gas_cost + execution_gas_cost

    raise ValueError(f"Unknown cost_mode: {cost_mode!r}")
