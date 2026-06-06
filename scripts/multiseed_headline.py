"""Multi-seed re-runs of the three single-seed headline Phase 2 configs, so the
paper can report mean +/- s.d. across seeds.

Configs (each at 100k steps, reward_scale 0.01, to match the original headline runs):
  - n5_congested : phase2.yaml, lambda_scale=100   (Fig 1 + Table 2)
  - calldata     : phase2.yaml, lambda_scale=100, action_mode=with_calldata (Fig 3)
  - n18          : phase2_n18.yaml, lambda_scale=300 (generality claim)

Writes results/multiseed/manifest_<stamp>.csv with one row per (config, seed),
recording the results_dir and the final-20%-window steady-state aggregate so
downstream analysis (notebooks/paper_figures.py) can compute across-seed bands.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.rl.marl_train import train

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 7, 99, 2024]

CONFIGS = {
    "n5_congested": {"phase": "configs/phase2.yaml",
                     "over": {"rollups.lambda_scale": 100.0,
                              "env.reward_scale": 1.0e-2,
                              "rl.total_timesteps": 100000}},
    "calldata": {"phase": "configs/phase2.yaml",
                 "over": {"rollups.lambda_scale": 100.0,
                          "env.reward_scale": 1.0e-2,
                          "env.action_mode": "with_calldata",
                          "rl.total_timesteps": 100000}},
    "n18": {"phase": "configs/phase2_n18.yaml",
            "over": {"rl.total_timesteps": 100000}},
}


def run_one(phase_path: str, over: dict, seed: int) -> dict:
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO / "configs/base.yaml"),
        OmegaConf.load(REPO / "configs/calibration.yaml"),
        OmegaConf.load(REPO / phase_path),
    )
    cfg.env.seed = int(seed)
    for k, v in over.items():
        OmegaConf.update(cfg, k, v, force_add=True)

    results_dir = train(cfg)
    df = pd.read_csv(results_dir / "history.csv")
    tail = df.tail(max(1, len(df) // 5))  # final 20% window
    return {
        "seed": int(seed),
        "results_dir": str(results_dir),
        "blobs_per_block_mean": float(tail["blobs_per_block_mean"].mean()),
        "blob_fee_mean": float(tail["blob_fee_mean"].mean()),
    }


def main() -> None:
    out_dir = REPO / "results" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"manifest_{stamp}.csv"

    rows = []
    total = len(CONFIGS) * len(SEEDS)
    done = 0
    for name, spec in CONFIGS.items():
        for seed in SEEDS:
            done += 1
            print(f"\n==== {done}/{total}: config={name} seed={seed} ====",
                  flush=True)
            row = run_one(spec["phase"], spec["over"], seed)
            row["config"] = name
            print(f"  -> blobs/block={row['blobs_per_block_mean']:.3f} "
                  f"dir={Path(row['results_dir']).name}", flush=True)
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
