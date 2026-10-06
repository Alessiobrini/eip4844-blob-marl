"""Non-learning baselines for the N=5 congested regime.

The baselines separate the contribution of learning to target tracking from
the negative feedback of the fee rule itself. This script runs rule-based policies through the same MultiAgentBlobEnv used for the
headline PPO experiment (phase2.yaml, lambda_scale=100) and reports the
steady-state aggregate blob supply, fee level, and per-rollup cost, so the
paper can compare them with the PPO equilibrium.

Baselines (all decide on the expected post-arrival queue Q + lambda_i, since
arrivals land before posting within a block):
  every_block  : drain the whole queue every block (post ceil(q/tau) blobs,
                 capped at 6), ignoring prices.
  full_blob    : post only fully packed blobs (floor(q/tau), capped at 6),
                 ignoring prices.
  myopic       : drain the whole queue iff the current delay cost of holding
                 it exceeds the posting cost at today's prices (a greedy,
                 price-responsive threshold with no learning).

Usage:
    PYTHONPATH=. python scripts/rule_based_baselines.py [--steps 20000]

Writes results/rule_baselines_<stamp>.csv with one row per (policy, seed).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.abm.multi_env import MultiAgentBlobEnv

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 7, 99, 2024]
BLOB_GAS = 131_072


def choose_action(
    policy: str,
    queue: int,
    lam: float,
    tau: int,
    alpha: float,
    blob_fee: float,
    gas_price: float,
    gas_fixed: int,
    blob_max: int,
) -> int:
    """Map a rule-based policy and the current state to a blob count.

    The decision uses the expected post-arrival queue q = queue + lam,
    because within env.step() arrivals land before posting executes.
    """
    q = queue + lam
    if policy == "every_block":
        return min(blob_max, math.ceil(q / tau)) if q > 0 else 0
    if policy == "full_blob":
        return min(blob_max, int(q // tau))
    if policy == "myopic":
        if q <= 0:
            return 0
        k = min(blob_max, math.ceil(q / tau))
        posting_cost = blob_fee * k * BLOB_GAS + gas_price * gas_fixed
        delay_cost = alpha * q
        return k if delay_cost >= posting_cost else 0
    raise ValueError(f"Unknown policy {policy!r}")


def run_one(policy: str, seed: int, n_steps: int, lambda_scale: float) -> dict:
    """Simulate one baseline policy for n_steps blocks.

    Args:
        policy: Baseline name (every_block, full_blob, myopic).
        seed: Environment seed.
        n_steps: Number of blocks to simulate.
        lambda_scale: Demand multiplier (100 = headline congested regime).

    Returns:
        Row dict with steady-state statistics over the final 20% of blocks.
    """
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO / "configs/base.yaml"),
        OmegaConf.load(REPO / "configs/calibration.yaml"),
        OmegaConf.load(REPO / "configs/phase2.yaml"),
    )
    OmegaConf.update(cfg, "rollups.lambda_scale", lambda_scale, force_add=True)
    cfg.env.seed = int(seed)
    env = MultiAgentBlobEnv(cfg)
    obs_dict, _ = env.reset(seed=seed)
    gas_fixed = int(cfg.env.gas_fixed)
    blob_max = int(cfg.env.blob_max)

    totals = np.zeros(n_steps, dtype=np.int32)
    fees = np.zeros(n_steps, dtype=np.float64)
    post_any = {lbl: np.zeros(n_steps, dtype=bool) for lbl in env.agent_ids}
    costs = {lbl: np.zeros(n_steps, dtype=np.float64) for lbl in env.agent_ids}

    for t in range(n_steps):
        action_dict = {}
        for lbl in env.agent_ids:
            agent = env.rollups[lbl]
            action_dict[lbl] = choose_action(
                policy=policy,
                queue=int(agent.queue),
                lam=float(agent.lambda_i),
                tau=agent.tx_per_blob(),
                alpha=float(agent.alpha_i),
                blob_fee=float(env.blob_fee),
                gas_price=float(env.gas_price),
                gas_fixed=gas_fixed,
                blob_max=blob_max,
            )
        obs_dict, rewards, _, trunc_dict, info = env.step(action_dict)
        totals[t] = info["total_blobs"]
        fees[t] = info["blob_fee"]
        for lbl in env.agent_ids:
            post_any[lbl][t] = info["blobs_by_agent"][lbl] > 0
            costs[lbl][t] = -rewards[lbl]
        if any(trunc_dict.values()):
            obs_dict, _ = env.reset(seed=seed + 1 + t)

    tail = slice(int(0.8 * n_steps), None)
    row = {
        "policy": policy,
        "seed": seed,
        "lambda_scale": lambda_scale,
        "blobs_per_block": float(totals[tail].mean()),
        "fee_geomean": float(np.exp(np.log(np.maximum(fees[tail], 1e-300)).mean())),
        "total_cost_per_block": float(
            sum(costs[lbl][tail].mean() for lbl in env.agent_ids)
        ),
    }
    for lbl in env.agent_ids:
        row[f"postfreq_{lbl}"] = float(post_any[lbl][tail].mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lambda-scale", type=float, default=100.0)
    args = parser.parse_args()

    rows = []
    for policy in ["every_block", "full_blob", "myopic"]:
        for seed in SEEDS:
            row = run_one(policy, seed, args.steps, args.lambda_scale)
            rows.append(row)
            print(row)

    df = pd.DataFrame(rows)
    out = REPO / "results" / f"rule_baselines_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\nMean +/- s.d. across seeds:")
    g = df.groupby("policy")[["blobs_per_block", "fee_geomean", "total_cost_per_block"]]
    print(g.agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
