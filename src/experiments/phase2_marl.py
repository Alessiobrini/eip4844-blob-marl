"""Phase 2 experiment runner: multi-rollup MARL with endogenous blob fee.

Usage:
    python src/experiments/phase2_marl.py --config configs/phase2.yaml
    python src/experiments/phase2_marl.py --config configs/phase2.yaml rl.total_timesteps=100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    """Parse config and launch MARL training."""
    parser = argparse.ArgumentParser(
        description="Phase 2: multi-rollup MARL for blob posting"
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config file",
    )
    args, overrides = parser.parse_known_args()

    base_cfg = OmegaConf.load("configs/base.yaml")
    cal_path = Path("configs/calibration.yaml")
    if cal_path.exists():
        cal_cfg = OmegaConf.load(cal_path)
        base_cfg = OmegaConf.merge(base_cfg, cal_cfg)
    phase_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, phase_cfg)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Import here so --help doesn't pull torch.
    from src.rl.marl_train import train

    results_dir = train(cfg)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
