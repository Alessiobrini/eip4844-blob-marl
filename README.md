# blob-abm-rl

Agent-based model and multi-agent reinforcement learning (RL) framework for
the EIP-4844 Ethereum blob fee market. The environment couples independent
learning rollups through an endogenous blob base fee and is calibrated to
post-Dencun on-chain data.

## Paper

This repository is the companion code for the paper

> Alessio Brini, "Do Rollups Self-Organize? Multi-Agent Reinforcement
> Learning in the EIP-4844 Blob Fee Market," accepted at the IEEE 4th
> International Conference on Artificial Intelligence, Blockchain, and
> Internet of Things (AIBThings 2026), Mount Pleasant, MI, USA.

It holds the environment, the calibrated parameters, the training and
evaluation scripts, and the BigQuery queries that regenerate the raw
on-chain data behind every number in the paper. See
[Reproducing the paper](#reproducing-the-paper) for the exhibit-by-exhibit
map.

```bibtex
@inproceedings{brini2026rollups,
  title     = {Do Rollups Self-Organize? Multi-Agent Reinforcement Learning
               in the {EIP-4844} Blob Fee Market},
  author    = {Brini, Alessio},
  booktitle = {IEEE 4th International Conference on Artificial Intelligence,
               Blockchain, and Internet of Things (AIBThings)},
  year      = {2026}
}
```

## Research question

EIP-4844 (Dencun, March 2024) gave rollups a dedicated data lane priced by a
multiplicative update rule targeting 3 blobs per block. The rule is a
negative-feedback controller, but whether the target is reached depends on
how independent, self-interested rollups respond to the price they jointly
create. Existing theory covers a single rollup under exogenous prices. This
project asks whether the protocol's intended operating point emerges when
heterogeneous rollups learn how to post against an endogenous fee.

## Repository layout

```
src/
├── abm/
│   ├── env.py               # BlobMarketEnv, single-agent Gymnasium env (validation)
│   ├── multi_env.py         # MultiAgentBlobEnv, N rollups + endogenous EIP-4844 fee
│   ├── fee_market.py        # EIP-4844 blob base fee update rule (canonical)
│   ├── gas_process.py       # Price processes: LogNormalRandomWalk, AR1Process
│   ├── run_sim.py           # Standalone ABM runner (rule-based policies, no RL)
│   └── agents/
│       ├── rollup.py        # RollupAgent, queue dynamics (Poisson / fixed batch)
│       └── builder.py       # BuilderAgent, pass-through inclusion
├── rl/
│   ├── train.py             # Single-agent DQN training (validation experiments)
│   ├── marl_train.py        # Independent PPO trainer, one policy per rollup
│   ├── evaluate.py          # Policy evaluation and threshold surface extraction
│   └── policies/            # Analytical threshold baseline, actor-critic net
├── data/
│   ├── fetch.py             # BigQuery queries: blocks + type-3 txs -> data/raw/
│   ├── calibration.py       # AR(1) fits + lambda_i estimation -> configs/calibration.yaml
│   ├── loaders.py           # Load raw CSV exports
│   └── rollup_labels.py     # Batcher-address -> rollup labeling
└── experiments/
    ├── phase1_single_agent.py   # CLI: single-agent DQN runs
    └── phase2_marl.py           # CLI: multi-agent PPO runs

configs/       # base.yaml + calibration.yaml (generated) + experiment configs
scripts/       # Paper experiments: multiseed runs, sweeps, baselines, ablations
notebooks/     # Figure and table regenerators (paper source of truth)
tests/         # 93 pytest tests: fee rule, queues, envs, price processes
```

## Setup

Python 3.11 with conda:

```bash
conda create -n blob-abm python=3.11
conda activate blob-abm
pip install mesa>=3.0 gymnasium stable-baselines3 torch numpy pandas \
    omegaconf mlflow matplotlib seaborn scipy pytest
```

Verify the installation (all 93 tests should pass):

```bash
pytest tests/ -v
```

## Data pipeline

Raw data come from Google BigQuery's public `crypto_ethereum` dataset over
the post-Dencun window (March 13 to August 31, 2024, 1,218,246 blocks).
`data/` is gitignored; regenerate it with your own Google Cloud credentials
(free tier suffices):

```bash
# Download blocks and type-3 (blob) transactions to data/raw/
python src/data/fetch.py

# Fit AR(1) price processes and per-rollup arrival rates
python src/data/calibration.py --output configs/calibration.yaml
```

The committed `configs/calibration.yaml` already contains the calibrated
parameters used in the paper, so training runs work without downloading
anything.

## Running experiments

```bash
# Single-agent validation runs (DQN vs the analytical threshold policy)
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml

# Multi-agent headline run (N=5 rollups, independent PPO, endogenous fee)
python src/experiments/phase2_marl.py --config configs/phase2.yaml \
    rollups.lambda_scale=100 env.reward_scale=0.01 rl.total_timesteps=100000

# 18-rollup roster
python src/experiments/phase2_marl.py --config configs/phase2_n18.yaml
```

Every run writes `results/{timestamp}_{config_hash}/` with the resolved
config, per-update metrics (`history.csv`), and policy checkpoints, and logs
to MLflow. All randomness is seeded through the configs.

## Reproducing the paper

Each exhibit in the paper maps to a committed script. The multi-seed runs
use seeds {42, 123, 7, 99, 2024} and report the final-20% steady-state
window.

| Paper exhibit | Script |
|---|---|
| Tab. I, AR(1) calibration | `src/data/calibration.py` |
| Tab. II, single-agent validation | `notebooks/validation_table.py` |
| Fig. 1 and Tab. III, headline N=5 runs | `scripts/multiseed_headline.py` |
| Fig. 2, phase diagram (10x to 500x) | `scripts/sweep_phase_diagram.py` |
| Fig. 3, 18-rollup generality | `scripts/multiseed_headline.py` (n18 config) |
| Fig. 4, calldata outside option | `scripts/multiseed_headline.py` (calldata config) |
| Non-learning baselines (Sec. V) | `scripts/rule_based_baselines.py` |
| Fixed delay-coefficient ablation (Sec. V) | `scripts/fixed_alpha_ablation.py` |
| Cap-binding footnote (Sec. III) | `scripts/measure_cap_binding.py` |
| Target sweep b* in {2,3,4} (Sec. VI) | `scripts/blobtarget_sweep.py` |
| All figure PDFs | `notebooks/paper_figures.py` |

`notebooks/paper_figures.py` and `notebooks/validation_table.py` are the
source of truth for the shipped figures and validation numbers. They read
the local `results/` tree produced by the scripts above and recompute every
headline statistic.

## Configuration

All parameters live in YAML under `configs/`, never hardcoded.
`configs/base.yaml` holds defaults, experiment configs override, and
`configs/calibration.yaml` is generated by the calibration pipeline. Key
switches: `env.queue_mode` (Poisson arrivals or fixed batch),
`env.action_mode` (binary, 0 to 6 blobs, or with calldata),
`env.cost_mode`, `price_process.type` (log-normal random walk or AR(1)),
`rollups.lambda_scale` (the demand multiplier), and
`rollups.alpha_fixed_lambda` (the fixed delay-coefficient ablation).

## Status

The project is concluded through the paper's scope: single-agent validation
against the analytical threshold policy and multi-agent experiments with an
endogenous fee (headline convergence, phase diagram, 18-rollup roster,
calldata outside option, baselines, and ablations). A learning builder and
cooperative blob sharing, Phases 3 and 4 of the original design in
`SPEC.md`, remain future work.

## Key references

- Bar-On and Mansour (2024). Optimal Publishing Strategies on a Base Layer. FC 2024.
- Wang, Crapis, and Moallemi (2025). A Framework for Combined Transaction Posting and Pricing for Layer 2 Blockchains. FC 2025.
- Crapis, Felten, and Mamageishvili (2024). EIP-4844 Economics and Rollup Strategies. FC 2024 Workshops.
- Huang et al. (2024). A First Look at Ethereum Blob Revolution. arXiv:2411.03892.
- Lee (2025). 180 Days After EIP-4844. IEEE ICDCSW 2025.

## License

See [LICENSE](LICENSE).
