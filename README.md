# blob-abm-rl

Agent-based model and multi-agent reinforcement learning framework for the
EIP-4844 Ethereum blob fee market. This project bridges the gap between
closed-form equilibrium theory and empirically observed anomalies in
post-Dencun rollup behavior.

## Research Motivation

EIP-4844 (Proto-Danksharding, March 2024) introduced a dedicated data lane
for L2 rollups with a target of 3 blobs per block and a multiplicative fee
update rule. The theoretical literature derives threshold-based optimal
posting strategies under simplifying assumptions, but empirical data reveals
persistent deviations:

- **29.48%** of blob-containing blocks are built suboptimally by builders
  (Huang et al., 2411.03892)
- Scroll and Starknet consistently use **one blob per transaction** (a
  dominated strategy) instead of batching
- Blob sharing could reduce costs by **85%+** for small rollups, but hasn't
  emerged spontaneously (Lee, 2410.04111)

This project uses RL to explore where and why analytical solutions break
down, producing novel insights for fee market design.

## Architecture

The environment is a **single codebase** that supports multiple experimental
configurations through YAML config flags. No separate implementations are
needed for different stages.

### Two-Stage Design (Phase 1)

| Axis | Stage A (Bar-On & Mansour) | Stage B (Shouqiao et al.) | Config Key |
|------|---------------------------|--------------------------|------------|
| Price process | Log-normal multiplicative RW | AR(1) mean-reverting | `price_process.type` |
| Queue mode | Fixed batch, deterministic wait | Poisson arrivals | `env.queue_mode` |
| Action space | Binary {wait, post} | Discrete {0..6} blobs | `env.action_mode` |
| Cost function | `alpha * wait_time + P * gas_fixed` | `alpha * Q + blob_cost + gas_cost` | `env.cost_mode` |
| Reward timing | Cost only on post step | Delay cost every step | `env.reward_mode` |

**Stage A** exactly replicates Bar-On & Mansour (2312.06448) so a DQN agent
can be benchmarked against their closed-form threshold policy
`lambda(x) = 2*alpha*x / (1 - gamma)`.

**Stage B** systematically relaxes assumptions one at a time to show where
RL adds value beyond analytical methods.

### Core Modules

```
src/
├── abm/
│   ├── env.py              # BlobMarketEnv — Gymnasium wrapper (central integration point)
│   ├── fee_market.py        # EIP-4844 blob base fee update rule (canonical, do not modify)
│   ├── gas_process.py       # Price processes: LogNormalRandomWalk, AR1Process
│   ├── run_sim.py           # Standalone ABM runner (no RL, analytical policies)
│   └── agents/
│       ├── rollup.py        # RollupAgent — Mesa agent with dual queue modes
│       └── builder.py       # BuilderAgent — pass-through in Phase 1
├── rl/
│   ├── train.py             # SB3 DQN training with MLflow logging
│   ├── evaluate.py          # Policy evaluation + threshold surface extraction
│   └── policies/
│       └── threshold.py     # Bar-On & Mansour analytical threshold baseline
├── data/
│   ├── calibration.py       # AR(1) fitting from Ethereum data (planned)
│   └── loaders.py           # Data loading from CSV/Dune exports (planned)
└── experiments/
    └── phase1_single_agent.py  # CLI entry point for Phase 1 experiments
```

### Environment Details

**BlobMarketEnv** (`src/abm/env.py`) is a Gymnasium-compatible environment:

- **Observation** `(3,)`: `[queue_or_waiting_time, blob_base_fee, gas_price]`
  (optionally log-transformed)
- **Action**: `Discrete(2)` in binary mode, `Discrete(7)` in discrete mode
- **Reward**: Negative cost = `-(delay_cost + posting_cost)`
- **Termination**: Fixed-horizon truncation (configurable, default 10,000 steps)
- **One step = one L1 block** (~12 seconds)

## Setup

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Installation

```bash
# Create and activate the conda environment
conda create -n blob-abm python=3.11
conda activate blob-abm

# Install dependencies
pip install mesa>=3.0 gymnasium stable-baselines3 numpy pandas \
    omegaconf mlflow matplotlib seaborn scipy pytest
```

### Verify Installation

```bash
pytest tests/ -v
```

All 59 tests should pass, covering price processes, fee market rules, agent
queue dynamics, and the Gymnasium environment lifecycle.

## Usage

### Run the ABM standalone (no RL)

Uses the analytical threshold policy from Bar-On & Mansour to validate
environment dynamics:

```bash
# Stage A: fixed batch, binary action, log-normal prices
python src/abm/run_sim.py --config configs/phase1_stage_a.yaml --steps 1000

# Stage B: Poisson arrivals, discrete blobs, AR(1) prices
python src/abm/run_sim.py --config configs/phase1_stage_b.yaml --steps 5000

# Save output to CSV
python src/abm/run_sim.py --config configs/phase1_stage_a.yaml --steps 10000 \
    --output results/abm_stage_a.csv
```

### Train a DQN agent

```bash
# Stage A: replicate Bar-On & Mansour threshold
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml

# Stage B: Shouqiao relaxation
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_b.yaml

# Override parameters from CLI
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml \
    rollup.alpha_i=2.0 rl.total_timesteps=1000000

# Short smoke test
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml \
    rl.total_timesteps=10000 env.max_steps=500
```

Training outputs are saved to `results/{timestamp}_{config_hash}/` and include:
- `config.yaml` — full resolved configuration
- `policy_checkpoint/dqn_model.zip` — saved model weights
- MLflow metrics (episode rewards, evaluation vs analytical baseline)

### Stage B Assumption Relaxations

Each relaxation is a config change. Run them one at a time to isolate effects:

```bash
# B1: stochastic arrivals (fixed_batch -> poisson_arrival)
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml \
    env.queue_mode=poisson_arrival rollup.lambda_i=180

# B2: fixed overhead cost
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml \
    env.cost_mode=linear_overhead

# B3: blob discretization
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml \
    env.action_mode=discrete_blobs

# B4: non-stationary (mean-reverting) prices
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml \
    price_process.type=ar1

# B5: quadratic delay cost
python src/experiments/phase1_single_agent.py --config configs/phase1_stage_a.yaml \
    env.cost_mode=quadratic_delay
```

### View experiment results

MLflow logs all metrics. To browse:

```bash
mlflow ui
# Then open http://localhost:5000
```

### Run tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_env.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Configuration

All parameters are in YAML configs under `configs/`. The base config
(`configs/base.yaml`) defines all defaults. Stage-specific configs override
only what changes.

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `env.queue_mode` | `"fixed_batch"` or `"poisson_arrival"` | `"poisson_arrival"` |
| `env.action_mode` | `"binary"` or `"discrete_blobs"` | `"discrete_blobs"` |
| `env.cost_mode` | `"bar_on_mansour"`, `"linear_overhead"`, or `"quadratic_delay"` | `"linear_overhead"` |
| `env.reward_mode` | `"per_step"` or `"on_post_only"` | `"per_step"` |
| `env.max_steps` | Episode length (blocks) | `10000` |
| `env.gas_fixed` | Type-3 tx intrinsic gas cost | `21000` |
| `rollup.lambda_i` | Transaction arrival rate (tx/block) | `180.0` |
| `rollup.alpha_i` | Delay cost coefficient | `1.0` |
| `rollup.d_i` | Data bytes per transaction | `500` |
| `rollup.gamma` | Discount factor | `0.99` |
| `price_process.type` | `"log_normal_rw"` or `"ar1"` | `"ar1"` |
| `rl.total_timesteps` | Training duration | `500000` |

## Project Phases

The full project proceeds in four phases (see `SPEC.md` for details):

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Single-agent RL vs analytical threshold | **In progress** |
| Phase 2 | Multi-rollup MARL with endogenous blob fee | Planned |
| Phase 3 | Strategic builder agent | Planned |
| Phase 4 | Blob sharing as cooperative MARL | Planned |

## Key References

- Bar-On & Mansour (2023). Optimal Publishing Strategies on a Base Layer. arXiv:2312.06448.
- Shouqiao et al. (2025). A Framework for Combined Transaction Posting and Pricing for L2 Blockchains. arXiv:2505.19556.
- Crapis, Felten & Mamageishvili (2023). EIP-4844 Economics and Rollup Strategies. arXiv:2310.01155.
- Huang et al. (2024). A First Look at Ethereum Blob Revolution. arXiv:2411.03892.
- Lee (2024). 180 Days After EIP-4844. arXiv:2410.04111.

## License

See [LICENSE](LICENSE).
