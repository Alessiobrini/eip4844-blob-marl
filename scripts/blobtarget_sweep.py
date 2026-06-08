"""b* sweep: vary the EIP-4844 blob target at fixed demand, to test whether the
equilibrium operating point moves with b* (paper Discussion, design corollary).

Holds the arrival-rate multiplier fixed (in the controllable regime from the
phase diagram) and varies env.blob_target over {2,3,4}, 5 seeds each. Writes
results/sweeps/blobtarget_<stamp>.csv with the final-20%-window steady-state
aggregate per (b*, seed).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.rl.marl_train import train

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 7, 99, 2024]
TARGETS = [2, 3, 4]
LAMBDA_SCALE = 200.0   # top of the fee-controllable regime (phase diagram)
STEPS = 50000


def run_one(target: int, seed: int) -> dict:
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO / "configs/base.yaml"),
        OmegaConf.load(REPO / "configs/calibration.yaml"),
        OmegaConf.load(REPO / "configs/phase2.yaml"),
    )
    cfg.env.seed = int(seed)
    cfg.env.blob_target = int(target)
    cfg.env.reward_scale = 1.0e-2
    cfg.rollups.lambda_scale = LAMBDA_SCALE
    cfg.rl.total_timesteps = STEPS

    results_dir = train(cfg)
    df = pd.read_csv(results_dir / "history.csv")
    tail = df.tail(max(1, len(df) // 5))
    return {
        "blob_target": int(target),
        "seed": int(seed),
        "blobs_per_block_mean": float(tail["blobs_per_block_mean"].mean()),
        "blob_fee_mean": float(tail["blob_fee_mean"].mean()),
        "results_dir": str(results_dir),
    }


def main() -> None:
    out_dir = REPO / "results" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"blobtarget_{stamp}.csv"

    rows = []
    total = len(TARGETS) * len(SEEDS)
    done = 0
    for t in TARGETS:
        for seed in SEEDS:
            done += 1
            print(f"\n==== {done}/{total}: b*={t} seed={seed} "
                  f"(lambda x{LAMBDA_SCALE:.0f}) ====", flush=True)
            row = run_one(t, seed)
            print(f"  -> blobs/block={row['blobs_per_block_mean']:.3f}", flush=True)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)
    print(df.groupby("blob_target")["blobs_per_block_mean"]
          .agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
