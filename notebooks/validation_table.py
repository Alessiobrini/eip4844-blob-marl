"""Regenerate the single-agent validation table (Tab. II, `tab:validation`)
directly from the committed Phase-1 run checkpoints and the analytical policy.

This is the source of truth for the validation numbers that ship in the paper
(the paper repository -> paper/main.tex, tab:validation), the
analog of `paper_figures.py` for the single-agent sanity check. Run end-to-end:

    python notebooks/validation_table.py

Each row maps to a specific Phase-1 run directory under results/ (model
outputs, gitignored). For each regime we rebuild `BlobMarketEnv` from that
run's resolved config, load its DQN checkpoint, and evaluate with the SAME
protocol used at train time (`evaluate_policy`, n_episodes=20, seed=1000), so
the printed numbers reproduce the table cell-for-cell:

  Regime              DQN post freq   Reference
  Martingale price    0.46            0.48 (BarOnMansourThreshold optimum)
  Calibrated floor    1.00            1.00 (always-post)
  Extreme congestion  0.64            8.8x cheaper than always-post

The "Reference" column is computed, not hand-typed: the analytical optimum is
the Bar-On & Mansour threshold policy evaluated in the same env; the always-post
baseline posts every block; the 8.8x is the cost ratio of that baseline to the
DQN. A trailing block asserts every value rounds to what the paper prints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import DQN

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.abm.env import BlobMarketEnv  # noqa: E402
from src.rl.evaluate import evaluate_policy  # noqa: E402
from src.rl.policies.threshold import BarOnMansourThreshold  # noqa: E402

# Phase-1 run directories backing each validation row. Identified by config:
#   martingale -> action_mode=binary, queue_mode=fixed_batch, exogenous_log_normal_rw
#                 fee (a random walk = martingale), alpha=1, gamma=0.995, 1M steps.
#   fee_floor  -> calibrated AR(1) blob fee at base (lambda=0.36) demand, 1M steps.
#   extreme    -> volatile fee (log-AR(1) mu=30, sigma=0.8), lambda=0.36.
RUNS = {
    "martingale": ROOT / "results/20260404_141509_c9940cd1",
    "fee_floor": ROOT / "results/20260408_160851_37a2b569",
    "extreme": ROOT / "results/20260408_231710_a4a92f91",
}

N_EPISODES = 20  # matches src/rl/train.py eval protocol
EVAL_SEED = 1000


class AlwaysPost:
    """Post every block (action 1). The cost-baseline heuristic the paper's
    8.8x and the fee-floor 1.00 reference are measured against. In the
    high-fee regime the reward is dominated by per-block fee exposure, so the
    baseline cost is insensitive to how many blobs are posted (>=1)."""

    def predict(self, obs, deterministic: bool = True):
        return np.array(1), None


def _dqn_metrics(run: Path) -> tuple[OmegaConf, dict]:
    cfg = OmegaConf.load(run / "config.yaml")
    model = DQN.load(str(run / "policy_checkpoint" / "dqn_model.zip"))
    metrics = evaluate_policy(
        BlobMarketEnv(cfg), model, n_episodes=N_EPISODES, seed=EVAL_SEED
    )
    return cfg, metrics


def main() -> None:
    rows = {}

    # Row 1: martingale price -- DQN vs analytical Bar-On & Mansour threshold.
    cfg, dqn = _dqn_metrics(RUNS["martingale"])
    analytical = evaluate_policy(
        BlobMarketEnv(cfg),
        BarOnMansourThreshold(
            alpha=cfg.rollup.alpha_i, gamma=cfg.rollup.gamma,
            gas_fixed=cfg.env.gas_fixed,
        ),
        n_episodes=N_EPISODES, seed=EVAL_SEED,
    )
    rows["martingale"] = (dqn["mean_post_frequency"],
                          analytical["mean_post_frequency"])

    # Row 2: calibrated fee floor -- DQN posts every block; always-post is optimal.
    cfg, dqn = _dqn_metrics(RUNS["fee_floor"])
    floor_base = evaluate_policy(
        BlobMarketEnv(cfg), AlwaysPost(), n_episodes=N_EPISODES, seed=EVAL_SEED
    )
    rows["fee_floor"] = (dqn["mean_post_frequency"],
                         floor_base["mean_post_frequency"])

    # Row 3: extreme congestion -- DQN times the market; cost ratio vs always-post.
    cfg, dqn = _dqn_metrics(RUNS["extreme"])
    extreme_base = evaluate_policy(
        BlobMarketEnv(cfg), AlwaysPost(), n_episodes=N_EPISODES, seed=EVAL_SEED
    )
    cost_ratio = extreme_base["mean_reward"] / dqn["mean_reward"]  # both negative
    rows["extreme"] = (dqn["mean_post_frequency"], cost_ratio)

    print("\n===== Tab. II (tab:validation) regenerated from checkpoints =====")
    print(f"{'Regime':22s} {'DQN post freq':>14s}   Reference")
    print(f"{'Martingale price':22s} {rows['martingale'][0]:>14.3f}   "
          f"{rows['martingale'][1]:.3f} (analytical opt.)")
    print(f"{'Calibrated fee floor':22s} {rows['fee_floor'][0]:>14.3f}   "
          f"{rows['fee_floor'][1]:.3f} (always-post opt.)")
    print(f"{'Extreme congestion':22s} {rows['extreme'][0]:>14.3f}   "
          f"{rows['extreme'][1]:.2f}x cheaper than always-post")

    # Assert each value rounds to what the paper prints.
    checks = [
        ("martingale DQN", round(rows["martingale"][0], 2), 0.46),
        ("martingale analytical", round(rows["martingale"][1], 2), 0.48),
        ("fee-floor DQN", round(rows["fee_floor"][0], 2), 1.00),
        ("fee-floor always-post", round(rows["fee_floor"][1], 2), 1.00),
        ("extreme DQN", round(rows["extreme"][0], 2), 0.64),
        ("extreme cost ratio", round(rows["extreme"][1], 1), 8.8),
    ]
    print("\nVerification vs paper:")
    ok = True
    for name, got, want in checks:
        match = abs(got - want) < 1e-9
        ok = ok and match
        print(f"  {'OK ' if match else 'XX '}{name:24s} got {got}  paper {want}")
    if not ok:
        raise SystemExit("MISMATCH: regenerated validation numbers differ from the paper.")
    print("\nAll validation-table cells reproduce the paper.")


if __name__ == "__main__":
    main()
