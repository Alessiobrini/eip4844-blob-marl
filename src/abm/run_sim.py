"""Standalone ABM runner: run with rule-based or analytical policies.

Useful for validating environment dynamics before training RL agents.

Usage:
    python src/abm/run_sim.py --config configs/phase1_stage_a.yaml --steps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.abm.env import BlobMarketEnv
from src.rl.policies.threshold import BarOnMansourThreshold


def run_simulation(
    cfg: OmegaConf,
    n_steps: int | None = None,
) -> pd.DataFrame:
    """Run the ABM with an analytical policy and return metrics.

    Args:
        cfg: Full resolved config.
        n_steps: Override for ``env.max_steps`` (optional).

    Returns:
        DataFrame with per-step metrics.
    """
    if n_steps is not None:
        cfg = OmegaConf.merge(cfg, {"env": {"max_steps": n_steps}})

    env = BlobMarketEnv(cfg)

    # Select policy based on config
    if cfg.env.queue_mode == "fixed_batch":
        policy = BarOnMansourThreshold(
            alpha=cfg.rollup.alpha_i,
            gamma=cfg.rollup.gamma,
            gas_fixed=cfg.env.gas_fixed,
        )
    else:
        # Simple heuristic for poisson mode: post when queue > threshold
        policy = _SimpleQueueThreshold(threshold=500)

    obs, info = env.reset()
    records = []

    done = False
    while not done:
        action_arr, _ = policy.predict(obs, deterministic=True)
        action = int(action_arr[0])
        obs, reward, terminated, truncated, info = env.step(action)
        records.append({
            **info,
            "reward": reward,
            "action": action,
        })
        done = terminated or truncated

    return pd.DataFrame(records)


class _SimpleQueueThreshold:
    """Simple queue-based threshold for poisson mode (placeholder)."""

    def __init__(self, threshold: int = 500) -> None:
        self.threshold = threshold

    def predict(self, obs, deterministic=True):
        queue = obs[0]
        if queue > self.threshold:
            return [6], None  # post max blobs
        return [0], None


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run ABM simulation")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save metrics CSV")
    args = parser.parse_args()

    base_cfg = OmegaConf.load("configs/base.yaml")
    stage_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, stage_cfg)

    df = run_simulation(cfg, n_steps=args.steps)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} steps to {out_path}")
    else:
        print(f"Simulation complete: {len(df)} steps")
        print(f"  Mean reward: {df['reward'].mean():.4f}")
        print(f"  Final queue: {df['queue'].iloc[-1]}")
        print(f"  Post frequency: {(df['action'] > 0).mean():.3f}")


if __name__ == "__main__":
    main()
