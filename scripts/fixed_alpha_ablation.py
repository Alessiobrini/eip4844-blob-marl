"""Fixed delay-coefficient ablation.

The calibration alpha_i = 2 * C_ref * lambda_i ties the delay penalty
to rollup size, so the monotone posting-frequency ordering could be induced by
the reward parameterization rather than learned from the arrival process. This
script re-trains the headline N=5 congested configuration (phase2.yaml,
lambda_scale=100) with a common delay coefficient for every rollup: alpha_i is
computed from the roster-mean empirical arrival rate instead of each rollup's
own (rollups.alpha_fixed_lambda). Arrival heterogeneity is untouched.

Usage:
    PYTHONPATH=. python scripts/fixed_alpha_ablation.py

Writes results/multiseed/manifest_fixed_alpha_<stamp>.csv with one row per
seed, recording final-20%-window aggregate supply and per-rollup posting
frequencies for the ordering check.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.rl.marl_train import train

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 7, 99, 2024]

LABELS = ["taiko", "base", "arbitrum_one", "scroll", "world_chain"]


def main() -> None:
    base = OmegaConf.merge(
        OmegaConf.load(REPO / "configs/base.yaml"),
        OmegaConf.load(REPO / "configs/calibration.yaml"),
        OmegaConf.load(REPO / "configs/phase2.yaml"),
    )
    rates = base.rollup_arrival.rollup_rates
    mean_lambda = float(np.mean([float(rates[lbl]) for lbl in LABELS]))
    print(f"Common alpha reference lambda (roster mean): {mean_lambda:.4f}")

    rows = []
    for seed in SEEDS:
        cfg = OmegaConf.merge(
            OmegaConf.load(REPO / "configs/base.yaml"),
            OmegaConf.load(REPO / "configs/calibration.yaml"),
            OmegaConf.load(REPO / "configs/phase2.yaml"),
        )
        overrides = {
            "rollups.lambda_scale": 100.0,
            "rollups.alpha_fixed_lambda": mean_lambda,
            "env.reward_scale": 1.0e-2,
            "rl.total_timesteps": 100000,
        }
        for k, v in overrides.items():
            OmegaConf.update(cfg, k, v, force_add=True)
        cfg.env.seed = int(seed)

        results_dir = train(cfg)
        df = pd.read_csv(results_dir / "history.csv")
        tail = df.tail(max(1, len(df) // 5))
        row = {
            "seed": int(seed),
            "results_dir": str(results_dir),
            "blobs_per_block_mean": float(tail["blobs_per_block_mean"].mean()),
            "blob_fee_mean": float(tail["blob_fee_mean"].mean()),
        }
        for lbl in LABELS:
            row[f"postfreq_{lbl}"] = float(tail[f"post_freq_{lbl}"].mean())
        rows.append(row)
        print(row)

    out = REPO / "results" / "multiseed" / (
        f"manifest_fixed_alpha_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")

    df = pd.DataFrame(rows)
    print("\nPosting frequency, mean +/- s.d. across seeds (ordering check):")
    for lbl in LABELS:
        col = df[f"postfreq_{lbl}"]
        print(f"  {lbl:<14s} {col.mean():.3f} +/- {col.std():.3f}")
    print(f"Aggregate blobs/block: {df['blobs_per_block_mean'].mean():.3f} "
          f"+/- {df['blobs_per_block_mean'].std():.3f}")


if __name__ == "__main__":
    main()
