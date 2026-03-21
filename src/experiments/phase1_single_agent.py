"""Phase 1 experiment runner: single-agent RL on blob posting.

Usage:
    python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml
    python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml rollup.alpha_i=2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    """Parse config and launch training."""
    parser = argparse.ArgumentParser(
        description="Phase 1: single-agent RL for blob posting"
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config file",
    )
    args, overrides = parser.parse_known_args()

    # Load base config, then merge stage-specific config, then CLI overrides
    base_cfg = OmegaConf.load("configs/base.yaml")
    stage_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, stage_cfg)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Import here to avoid slow imports when just checking --help
    from src.rl.train import train

    results_dir = train(cfg)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
