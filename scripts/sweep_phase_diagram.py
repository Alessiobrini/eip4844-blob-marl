"""Sweep lambda_scale x seed for the Phase 2 phase diagram.

Runs a grid of training jobs in the current process (not Slurm). For each
combination, calls src.rl.marl_train.train and records the final steady
state (blobs/block, blob fee). Results are written to
results/sweeps/phase_diagram_<timestamp>.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from omegaconf import OmegaConf

from src.rl.marl_train import train

REPO = Path(__file__).resolve().parents[1]


def run_one(lambda_scale: float, seed: int, steps: int) -> dict[str, float]:
    """Launch one training run; return tail-averaged steady-state stats."""
    base_cfg = OmegaConf.load(REPO / "configs/base.yaml")
    cal_cfg = OmegaConf.load(REPO / "configs/calibration.yaml")
    phase_cfg = OmegaConf.load(REPO / "configs/phase2.yaml")
    cfg = OmegaConf.merge(base_cfg, cal_cfg, phase_cfg)
    cfg.env.seed = int(seed)
    cfg.env.reward_scale = 1.0e-2
    cfg.rollups.lambda_scale = float(lambda_scale)
    cfg.rl.total_timesteps = int(steps)

    results_dir = train(cfg)

    # Tail-averaged stats over last 20% of rollouts.
    import pandas as pd
    df = pd.read_csv(results_dir / "history.csv")
    tail = df.tail(max(1, len(df) // 5))
    return {
        "lambda_scale": float(lambda_scale),
        "seed": int(seed),
        "blobs_per_block_mean": float(tail["blobs_per_block_mean"].mean()),
        "blob_fee_mean": float(tail["blob_fee_mean"].mean()),
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[10, 50, 100, 200, 500])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()

    out_dir = REPO / "results" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"phase_diagram_{stamp}.csv"

    rows: list[dict[str, float]] = []
    total = len(args.scales) * len(args.seeds)
    done = 0
    for ls in args.scales:
        for seed in args.seeds:
            done += 1
            print(f"\n==== run {done}/{total}: lambda_scale={ls} seed={seed} ====")
            row = run_one(ls, seed, args.steps)
            print(f"  -> blobs/block={row['blobs_per_block_mean']:.3f}")
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
