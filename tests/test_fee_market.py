"""Tests for EIP-4844 fee market functions."""

import math

import pytest

from src.abm.fee_market import compute_posting_cost, update_blob_base_fee


class TestUpdateBlobBaseFee:
    """Tests for the canonical EIP-4844 blob base fee update rule."""

    def test_target_blobs_no_change(self) -> None:
        """At target (3 blobs), fee should stay the same."""
        B = 1000.0
        assert update_blob_base_fee(B, blobs_used=3, target=3) == pytest.approx(B)

    def test_zero_blobs_decreases_fee(self) -> None:
        """0 blobs should decrease fee by exp(-3/24) = exp(-1/8)."""
        B = 1000.0
        expected = B * math.exp(-1 / 8)
        assert update_blob_base_fee(B, blobs_used=0, target=3) == pytest.approx(expected)

    def test_max_blobs_increases_fee(self) -> None:
        """6 blobs should increase fee by exp(3/24) = exp(1/8)."""
        B = 1000.0
        expected = B * math.exp(1 / 8)
        assert update_blob_base_fee(B, blobs_used=6, target=3) == pytest.approx(expected)

    def test_increase_decrease_are_inverses(self) -> None:
        """exp(1/8) and exp(-1/8) are multiplicative inverses."""
        B = 1000.0
        B_up = update_blob_base_fee(B, blobs_used=6, target=3)
        B_restored = update_blob_base_fee(B_up, blobs_used=0, target=3)
        assert B_restored == pytest.approx(B)

    def test_single_blob(self) -> None:
        """1 blob: exp((1-3)/(8*3)) = exp(-2/24) = exp(-1/12)."""
        B = 500.0
        expected = B * math.exp(-1 / 12)
        assert update_blob_base_fee(B, blobs_used=1, target=3) == pytest.approx(expected)

    def test_preserves_positive(self) -> None:
        """Fee should stay positive for any blob count."""
        B = 1.0  # 1-wei minimum
        for blobs in range(7):
            result = update_blob_base_fee(B, blobs_used=blobs, target=3)
            assert result > 0


class TestComputePostingCost:
    """Tests for posting cost computation under different cost modes."""

    def test_bar_on_mansour_mode(self) -> None:
        """Bar-On & Mansour: cost = gas_price * gas_fixed (blob fee irrelevant)."""
        cost = compute_posting_cost(
            blob_base_fee=999.0,  # should be ignored
            gas_price=2.0,
            n_blobs=3,  # should be ignored
            gas_fixed=21_000,
            cost_mode="bar_on_mansour",
        )
        assert cost == pytest.approx(2.0 * 21_000)

    def test_linear_overhead_mode(self) -> None:
        """Linear overhead: blob_fee * n_blobs * 131072 + gas_price * gas_fixed."""
        cost = compute_posting_cost(
            blob_base_fee=10.0,
            gas_price=5.0,
            n_blobs=2,
            gas_fixed=21_000,
            cost_mode="linear_overhead",
        )
        expected = 10.0 * 2 * 131_072 + 5.0 * 21_000
        assert cost == pytest.approx(expected)

    def test_linear_overhead_single_blob(self) -> None:
        cost = compute_posting_cost(
            blob_base_fee=1.0,
            gas_price=1.0,
            n_blobs=1,
            gas_fixed=21_000,
            cost_mode="linear_overhead",
        )
        expected = 1.0 * 1 * 131_072 + 1.0 * 21_000
        assert cost == pytest.approx(expected)

    def test_quadratic_delay_same_as_linear_overhead(self) -> None:
        """quadratic_delay uses the same posting cost as linear_overhead."""
        args = dict(
            blob_base_fee=10.0, gas_price=5.0, n_blobs=3, gas_fixed=21_000,
        )
        cost_lo = compute_posting_cost(**args, cost_mode="linear_overhead")
        cost_qd = compute_posting_cost(**args, cost_mode="quadratic_delay")
        assert cost_lo == pytest.approx(cost_qd)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown cost_mode"):
            compute_posting_cost(1.0, 1.0, 1, 21_000, cost_mode="invalid")
