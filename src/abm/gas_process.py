"""Stochastic price processes for L1 gas price and exogenous blob base fee.

Two implementations:
- LogNormalRandomWalk: P_{t+1} = P_t * exp(N(mu, sigma^2))  [Bar-On & Mansour]
- AR1Process: P_{t+1} = theta*mu + (1-theta)*P_t + sigma*eps  [Wang et al.]
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig


class PriceProcess(ABC):
    """Base class for stochastic price processes."""

    def __init__(self, initial_price: float, seed: int | None = None) -> None:
        self._initial_price = initial_price
        self._price = initial_price
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def step(self) -> float:
        """Advance one block and return the new price."""

    def reset(self) -> float:
        """Reset to initial price and return it."""
        self._price = self._initial_price
        return self._price

    @property
    def current_price(self) -> float:
        """Return the current price."""
        return self._price


class LogNormalRandomWalk(PriceProcess):
    """Log-normal multiplicative random walk: P_{t+1} = P_t * exp(X_t).

    X_t ~ N(mu, sigma^2).  With mu = -sigma^2/2 the process is a martingale
    (E[P_{t+1}] = P_t).  This matches Bar-On & Mansour (2312.06448).

    Args:
        mu: Drift parameter for the log-increment.
        sigma: Volatility (std dev of log-increment).
        initial_price: Starting price (must be positive).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        initial_price: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(initial_price, seed)
        self.mu = mu
        self.sigma = sigma

    def step(self) -> float:
        """Advance one block via multiplicative log-normal shock."""
        log_increment = self._rng.normal(self.mu, self.sigma)
        self._price *= math.exp(log_increment)
        return self._price


class AR1Process(PriceProcess):
    """AR(1) mean-reverting process: P_{t+1} = theta*mu + (1-theta)*P_t + sigma*eps.

    eps ~ N(0, 1).  Calibrated parameters from Wang et al. (2505.19556):
    mu = 3.86e-8, theta = 0.1, sigma = 8.41e-9.

    Args:
        mu: Long-run mean price.
        theta: Mean-reversion speed (0 < theta < 1).
        sigma: Innovation volatility.
        initial_price: Starting price.
        floor: Minimum price (clamp to non-negative). Default 0.0.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        mu: float,
        theta: float,
        sigma: float,
        initial_price: float,
        floor: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(initial_price, seed)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.floor = floor

    def step(self) -> float:
        """Advance one block via AR(1) update with optional floor."""
        eps = self._rng.normal()
        self._price = (
            self.theta * self.mu
            + (1 - self.theta) * self._price
            + self.sigma * eps
        )
        if self.floor is not None:
            self._price = max(self.floor, self._price)
        return self._price


class LogAR1Process(PriceProcess):
    """AR(1) in log-space: log(P_{t+1}) = theta*mu + (1-theta)*log(P_t) + sigma*eps.

    Equivalent to multiplicative dynamics.  Exponentiating gives:
    P_{t+1} = exp(theta*mu + (1-theta)*log(P_t) + sigma*eps)

    This matches the structure of the EIP-4844 blob base fee update rule,
    which is multiplicative (B_{t+1} = B_t * exp(...)).

    Args:
        mu: Long-run mean of log-price.
        theta: Mean-reversion speed in log-space (0 < theta < 1).
        sigma: Innovation volatility in log-space.
        initial_price: Starting price (in level space, must be positive).
        floor: Minimum price (level space). Default 1.0.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        mu: float,
        theta: float,
        sigma: float,
        initial_price: float,
        floor: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(initial_price, seed)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.floor = floor
        self._log_price = math.log(max(initial_price, 1e-18))

    def step(self) -> float:
        """Advance one block via AR(1) update in log-space."""
        eps = self._rng.normal()
        self._log_price = (
            self.theta * self.mu
            + (1 - self.theta) * self._log_price
            + self.sigma * eps
        )
        self._price = math.exp(self._log_price)
        if self.floor is not None:
            self._price = max(self.floor, self._price)
        return self._price

    def reset(self) -> float:
        """Reset to initial price and return it."""
        self._price = self._initial_price
        self._log_price = math.log(max(self._initial_price, 1e-18))
        return self._price


def create_price_process(cfg: DictConfig, seed: int | None = None) -> PriceProcess:
    """Factory: create a PriceProcess from a config block.

    Args:
        cfg: Config with a ``type`` key and process-specific parameters.
        seed: Random seed (overrides any seed in cfg).

    Returns:
        An initialized PriceProcess instance.

    Raises:
        ValueError: If ``cfg.type`` is not recognized.
    """
    ptype = cfg.type

    if ptype == "log_normal_rw":
        return LogNormalRandomWalk(
            mu=cfg.ln_mu,
            sigma=cfg.ln_sigma,
            initial_price=cfg.initial_price,
            seed=seed,
        )

    if ptype in ("ar1", "exogenous_ar1"):
        return AR1Process(
            mu=cfg.mu,
            theta=cfg.theta,
            sigma=cfg.sigma,
            initial_price=cfg.get("initial_price", cfg.get("initial_fee", cfg.mu)),
            floor=cfg.get("floor", 0.0),
            seed=seed,
        )

    if ptype == "exogenous_log_ar1":
        return LogAR1Process(
            mu=cfg.mu,
            theta=cfg.theta,
            sigma=cfg.sigma,
            initial_price=cfg.get("initial_price", cfg.get("initial_fee", math.exp(cfg.mu))),
            floor=cfg.get("floor", 1.0),
            seed=seed,
        )

    if ptype == "exogenous_log_normal_rw":
        return LogNormalRandomWalk(
            mu=cfg.get("ln_mu", 0.0),
            sigma=cfg.get("ln_sigma", cfg.get("sigma", 0.1)),
            initial_price=cfg.get("initial_price", cfg.get("initial_fee", 1.0)),
            seed=seed,
        )

    raise ValueError(f"Unknown price process type: {ptype!r}")
