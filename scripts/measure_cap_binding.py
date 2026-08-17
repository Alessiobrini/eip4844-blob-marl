"""Measure how often the per-block blob cap binds under trained policies.

The environment clips the
aggregate blob count to blob_max=6 only for the fee update, while queues are
drained and posting costs paid on the full request. This script quantifies how
often that clip is active in the trained steady state: it reloads the saved
policy checkpoints of the headline N=5 congested runs and of every
phase-diagram sweep run, rolls each out for a fixed number of blocks, and
reports the fraction of blocks with aggregate requested blobs above the cap,
together with the mean excess on those blocks.

Usage:
    python scripts/measure_cap_binding.py [--steps 20000]

Writes results/cap_binding_<stamp>.csv with one row per evaluated run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.abm.multi_env import MultiAgentBlobEnv
from src.rl.policies.actor_critic import ActorCritic

REPO = Path(__file__).resolve().parents[1]

HEADLINE_MANIFEST = REPO / "results/multiseed/manifest_20260606_151700.csv"
SWEEP_MANIFEST = REPO / "results/sweeps/phase_diagram_20260606_150026.csv"


def evaluate_run(results_dir: Path, n_steps: int, eval_seed: int) -> dict:
    """Roll out a trained run's checkpoints and tally cap exceedances.

    Args:
        results_dir: Run directory holding config.yaml and policy_checkpoint/.
        n_steps: Number of blocks to simulate.
        eval_seed: Seed for the evaluation environment (distinct from training).

    Returns:
        Row dict with exceedance statistics for the run.
    """
    cfg = OmegaConf.load(results_dir / "config.yaml")
    env = MultiAgentBlobEnv(cfg)
    obs_dim = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n

    policies = {}
    for label in env.agent_ids:
        policy = ActorCritic(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=int(cfg.rl.hidden_dim),
        )
        state = torch.load(
            results_dir / "policy_checkpoint" / f"{label}.pt",
            weights_only=True,
        )
        policy.load_state_dict(state)
        policy.eval()
        policies[label] = policy

    obs_dict, _ = env.reset(seed=eval_seed)
    totals = np.zeros(n_steps, dtype=np.int32)
    costs = np.zeros(n_steps, dtype=np.float64)
    reward_scale = float(cfg.env.get("reward_scale", 1.0))
    for t in range(n_steps):
        action_dict = {}
        for label in env.agent_ids:
            obs_t = torch.as_tensor(obs_dict[label]).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = policies[label].act(obs_t)
            action_dict[label] = int(action.item())
        obs_dict, rewards, _, trunc_dict, info = env.step(action_dict)
        totals[t] = info["total_blobs"]
        # Undo reward_scale so costs are in cost_normalization units,
        # comparable across configs and with the rule-based baselines.
        costs[t] = -sum(rewards.values()) / reward_scale
        if any(trunc_dict.values()):
            obs_dict, _ = env.reset(seed=eval_seed + 1 + t)

    cap = int(cfg.env.blob_max)
    over = totals > cap
    return {
        "results_dir": str(results_dir.relative_to(REPO)),
        "lambda_scale": float(cfg.rollups.get("lambda_scale", 1.0)),
        "n_steps": n_steps,
        "blobs_per_block_mean": float(totals.mean()),
        "cap": cap,
        "frac_over_cap": float(over.mean()),
        "mean_excess_when_over": float((totals[over] - cap).mean()) if over.any() else 0.0,
        "max_total": int(totals.max()),
        "total_cost_per_block": float(costs[int(0.8 * n_steps):].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-seed", type=int, default=777)
    args = parser.parse_args()

    rows = []
    headline = pd.read_csv(HEADLINE_MANIFEST)
    headline = headline[headline["config"] == "n5_congested"]
    for _, r in headline.iterrows():
        row = evaluate_run(REPO / r["results_dir"], args.steps, args.eval_seed)
        row["config"] = "n5_congested_100x"
        rows.append(row)
        print(row)

    sweep = pd.read_csv(SWEEP_MANIFEST)
    for _, r in sweep.iterrows():
        row = evaluate_run(REPO / r["results_dir"], args.steps, args.eval_seed)
        row["config"] = f"phase_diagram_{r['lambda_scale']:g}x"
        rows.append(row)
        print(row)

    out = REPO / "results" / f"cap_binding_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")

    df = pd.DataFrame(rows)
    print("\nSummary by config (mean across seeds):")
    print(
        df.groupby("config")[["blobs_per_block_mean", "frac_over_cap",
                              "mean_excess_when_over"]].mean().round(4)
    )


if __name__ == "__main__":
    main()
