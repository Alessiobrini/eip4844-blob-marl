"""Training entry point for RL agents on BlobMarketEnv.

Uses Stable-Baselines3 DQN for Phase 1.  Logs metrics to MLflow and saves
checkpoints + resolved config to the results directory.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.abm.env import BlobMarketEnv
from src.rl.evaluate import evaluate_policy
from src.rl.policies.threshold import BarOnMansourThreshold


class MetricsCallback(BaseCallback):
    """SB3 callback that logs episode metrics to MLflow.

    Args:
        log_interval: Log metrics every N timesteps.
    """

    def __init__(self, log_interval: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            infos = self.locals.get("infos", [])
            if infos:
                for info in infos:
                    if "episode" in info:
                        mlflow.log_metric(
                            "episode_reward", info["episode"]["r"],
                            step=self.num_timesteps,
                        )
                        mlflow.log_metric(
                            "episode_length", info["episode"]["l"],
                            step=self.num_timesteps,
                        )
        return True


def create_results_dir(cfg: DictConfig) -> Path:
    """Create a timestamped results directory with config hash.

    Args:
        cfg: Full resolved config.

    Returns:
        Path to the created results directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_str = OmegaConf.to_yaml(cfg)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    results_dir = Path("results") / f"{timestamp}_{cfg_hash}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def train(cfg: DictConfig) -> Path:
    """Train a DQN agent on BlobMarketEnv.

    Args:
        cfg: Full OmegaConf config with ``env``, ``rollup``,
            ``price_process``, ``blob_fee_process``, and ``rl`` sections.

    Returns:
        Path to the results directory.
    """
    # 1. Create environment
    env = BlobMarketEnv(cfg)
    env = Monitor(env)

    # 2. Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg.rl.learning_rate,
        buffer_size=cfg.rl.buffer_size,
        batch_size=cfg.rl.batch_size,
        gamma=cfg.rollup.gamma,
        exploration_fraction=cfg.rl.exploration_fraction,
        target_update_interval=cfg.rl.target_update_interval,
        verbose=1,
        seed=cfg.env.seed,
    )

    # 3. MLflow tracking
    mlflow.set_experiment("phase1_single_agent")
    with mlflow.start_run():
        mlflow.log_params(
            {k: str(v) for k, v in
             OmegaConf.to_container(cfg, resolve=True).items()}
        )

        # 4. Train
        callback = MetricsCallback(log_interval=1000)
        model.learn(
            total_timesteps=cfg.rl.total_timesteps,
            callback=callback,
        )

        # 5. Save results
        results_dir = create_results_dir(cfg)
        checkpoint_dir = results_dir / "policy_checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)
        model.save(str(checkpoint_dir / "dqn_model"))
        OmegaConf.save(cfg, str(results_dir / "config.yaml"))

        # 6. Evaluate trained agent
        eval_env = BlobMarketEnv(cfg)
        metrics = evaluate_policy(eval_env, model, n_episodes=20, seed=1000)
        for k, v in metrics.items():
            mlflow.log_metric(f"eval_{k}", v)

        # 7. Evaluate analytical baseline (Stage A only)
        if cfg.env.action_mode == "binary" and cfg.env.queue_mode == "fixed_batch":
            baseline = BarOnMansourThreshold(
                alpha=cfg.rollup.alpha_i,
                gamma=cfg.rollup.gamma,
                gas_fixed=cfg.env.gas_fixed,
            )
            baseline_metrics = evaluate_policy(
                eval_env, baseline, n_episodes=20, seed=1000
            )
            for k, v in baseline_metrics.items():
                mlflow.log_metric(f"baseline_{k}", v)

    return results_dir
