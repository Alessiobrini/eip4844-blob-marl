"""Independent PPO trainer for Phase 2.

N rollup agents learn simultaneously on a shared MultiAgentBlobEnv. Each
agent has its own ActorCritic, optimizer, and rollout buffer. Rollouts are
collected synchronously (one env step per timestep, all agents act together),
then each agent runs its own PPO update on its own trajectory.

This is the "Independent Learners" baseline specified in SPEC.md Phase 2.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.abm.multi_env import MultiAgentBlobEnv, RollupId
from src.rl.policies.actor_critic import ActorCritic


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    """Per-agent PPO rollout storage (one env, T timesteps)."""

    obs: np.ndarray          # (T, obs_dim)
    actions: np.ndarray      # (T,)
    log_probs: np.ndarray    # (T,)
    values: np.ndarray       # (T,)
    rewards: np.ndarray      # (T,)
    dones: np.ndarray        # (T,)

    @classmethod
    def empty(cls, T: int, obs_dim: int) -> "RolloutBuffer":
        return cls(
            obs=np.zeros((T, obs_dim), dtype=np.float32),
            actions=np.zeros(T, dtype=np.int64),
            log_probs=np.zeros(T, dtype=np.float32),
            values=np.zeros(T, dtype=np.float32),
            rewards=np.zeros(T, dtype=np.float32),
            dones=np.zeros(T, dtype=np.float32),
        )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation.

    Args:
        rewards: (T,) rewards.
        values: (T,) value estimates.
        dones: (T,) episode-end indicators.
        last_value: Value estimate at the step after the rollout.
        gamma: Discount.
        lam: GAE lambda.

    Returns:
        (advantages, returns) each of shape (T,).
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: DictConfig,
) -> dict[str, float]:
    """One PPO epoch sweep (mini-batched) over a rollout.

    Returns a dict of loss statistics averaged over mini-batches.
    """
    B = obs.shape[0]
    batch_size = int(cfg.rl.batch_size)
    idx = np.arange(B)

    # Normalize advantages (per epoch).
    adv_np = advantages.cpu().numpy()
    adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
    advantages = torch.as_tensor(adv_np, dtype=torch.float32)

    stats: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
    }

    for _epoch in range(int(cfg.rl.n_epochs)):
        np.random.shuffle(idx)
        for start in range(0, B, batch_size):
            mb = idx[start : start + batch_size]
            mb_obs = obs[mb]
            mb_actions = actions[mb]
            mb_old_lp = old_log_probs[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            new_lp, ent, value = policy.evaluate(mb_obs, mb_actions)
            ratio = torch.exp(new_lp - mb_old_lp)

            unclipped = ratio * mb_adv
            clipped = torch.clamp(
                ratio, 1.0 - float(cfg.rl.clip_coef), 1.0 + float(cfg.rl.clip_coef)
            ) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = 0.5 * (value - mb_ret).pow(2).mean()
            entropy_loss = -ent.mean()
            loss = (
                policy_loss
                + float(cfg.rl.vf_coef) * value_loss
                + float(cfg.rl.ent_coef) * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.rl.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_lp - new_lp).mean().item()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(-entropy_loss.item())
            stats["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Trainer entry point
# ---------------------------------------------------------------------------
def create_results_dir(cfg: DictConfig) -> Path:
    """Timestamped results dir with a short config hash."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_str = OmegaConf.to_yaml(cfg)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    results_dir = Path("results") / f"phase2_{timestamp}_{cfg_hash}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def train(cfg: DictConfig) -> Path:
    """Train independent PPO agents on MultiAgentBlobEnv.

    Args:
        cfg: Full resolved config.

    Returns:
        Path to the results directory.
    """
    env = MultiAgentBlobEnv(cfg)
    obs_dim = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n
    gamma = float(cfg.rollups.gamma)
    lam = float(cfg.rl.gae_lambda)

    # Build one policy + optimizer per agent.
    policies: dict[RollupId, ActorCritic] = {}
    optimizers: dict[RollupId, torch.optim.Optimizer] = {}
    for label in env.agent_ids:
        policy = ActorCritic(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=int(cfg.rl.hidden_dim),
        )
        policies[label] = policy
        optimizers[label] = torch.optim.Adam(
            policy.parameters(), lr=float(cfg.rl.learning_rate), eps=1e-5
        )

    T = int(cfg.rl.rollout_steps)
    total_steps = int(cfg.rl.total_timesteps)
    n_updates = max(1, total_steps // T)

    results_dir = create_results_dir(cfg)
    OmegaConf.save(cfg, str(results_dir / "config.yaml"))

    mlflow.set_experiment("phase2_marl")
    with mlflow.start_run():
        mlflow.log_params(
            {k: str(v) for k, v in OmegaConf.to_container(cfg, resolve=True).items()}
        )

        obs_dict, _ = env.reset()
        global_step = 0

        # Per-rollout aggregate metrics across training (for CSV dump).
        history: list[dict[str, float]] = []

        for update in range(1, n_updates + 1):
            # ---- Collect rollout ---------------------------------------------------
            buffers = {
                lbl: RolloutBuffer.empty(T, obs_dim) for lbl in env.agent_ids
            }
            fee_trace = np.zeros(T, dtype=np.float32)
            blobs_trace = np.zeros(T, dtype=np.int32)

            for t in range(T):
                action_dict: dict[RollupId, int] = {}
                step_cache: dict[RollupId, tuple[float, float, np.ndarray]] = {}

                for lbl in env.agent_ids:
                    obs_t = torch.as_tensor(obs_dict[lbl]).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob, value = policies[lbl].act(obs_t)
                    action_dict[lbl] = int(action.item())
                    step_cache[lbl] = (
                        float(log_prob.item()),
                        float(value.item()),
                        obs_dict[lbl].copy(),
                    )

                next_obs_dict, reward_dict, term_dict, trunc_dict, info = env.step(
                    action_dict
                )

                fee_trace[t] = info["blob_fee"]
                blobs_trace[t] = info["total_blobs"]

                for lbl in env.agent_ids:
                    lp, v, o = step_cache[lbl]
                    buf = buffers[lbl]
                    buf.obs[t] = o
                    buf.actions[t] = action_dict[lbl]
                    buf.log_probs[t] = lp
                    buf.values[t] = v
                    buf.rewards[t] = reward_dict[lbl]
                    buf.dones[t] = float(term_dict[lbl] or trunc_dict[lbl])

                done_any = any(term_dict[l] or trunc_dict[l] for l in env.agent_ids)
                if done_any:
                    obs_dict, _ = env.reset()
                else:
                    obs_dict = next_obs_dict

                global_step += 1

            # ---- Compute advantages + PPO update per agent ------------------------
            update_stats: dict[str, dict[str, float]] = {}
            for lbl in env.agent_ids:
                buf = buffers[lbl]
                with torch.no_grad():
                    last_obs = torch.as_tensor(obs_dict[lbl]).unsqueeze(0)
                    _, _, last_val = policies[lbl].act(last_obs)
                adv, ret = compute_gae(
                    rewards=buf.rewards,
                    values=buf.values,
                    dones=buf.dones,
                    last_value=float(last_val.item()),
                    gamma=gamma,
                    lam=lam,
                )
                stats = ppo_update(
                    policy=policies[lbl],
                    optimizer=optimizers[lbl],
                    obs=torch.as_tensor(buf.obs),
                    actions=torch.as_tensor(buf.actions),
                    old_log_probs=torch.as_tensor(buf.log_probs),
                    advantages=torch.as_tensor(adv),
                    returns=torch.as_tensor(ret),
                    cfg=cfg,
                )
                update_stats[lbl] = stats

            # ---- Logging -----------------------------------------------------------
            if update % int(cfg.rl.log_interval) == 0:
                row: dict[str, float] = {
                    "update": float(update),
                    "global_step": float(global_step),
                    "blob_fee_mean": float(fee_trace.mean()),
                    "blob_fee_final": float(fee_trace[-1]),
                    "blobs_per_block_mean": float(blobs_trace.mean()),
                }
                for lbl in env.agent_ids:
                    buf = buffers[lbl]
                    row[f"reward_{lbl}"] = float(buf.rewards.mean())
                    row[f"post_freq_{lbl}"] = float((buf.actions > 0).mean())
                    row[f"blob_post_freq_{lbl}"] = float(
                        ((buf.actions >= 1) & (buf.actions <= 6)).mean()
                    )
                    row[f"calldata_freq_{lbl}"] = float((buf.actions == 7).mean())
                    row[f"entropy_{lbl}"] = update_stats[lbl]["entropy"]
                for k, v in row.items():
                    if k == "update":
                        continue
                    mlflow.log_metric(k, v, step=global_step)
                history.append(row)
                print(
                    f"[update {update:>4d}/{n_updates}] "
                    f"step={global_step} "
                    f"fee={row['blob_fee_mean']:.4f} "
                    f"blobs/blk={row['blobs_per_block_mean']:.2f} "
                )

        # ---- Save checkpoints + history --------------------------------------------
        ckpt_dir = results_dir / "policy_checkpoint"
        ckpt_dir.mkdir(exist_ok=True)
        for lbl, policy in policies.items():
            torch.save(policy.state_dict(), ckpt_dir / f"{lbl}.pt")

        hist_path = results_dir / "history.csv"
        if history:
            keys = list(history[0].keys())
            with hist_path.open("w") as f:
                f.write(",".join(keys) + "\n")
                for row in history:
                    f.write(",".join(f"{row.get(k, ''):.6f}" for k in keys) + "\n")

    return results_dir
