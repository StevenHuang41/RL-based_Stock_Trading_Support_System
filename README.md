# RL-based Stock Trading Support System

[![CI](https://github.com/StevenHuang41/RL-based_Stock_Trading_Support_System/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenHuang41/RL-based_Stock_Trading_Support_System/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)

An experimental reinforcement learning project for studying stock-trading decision policies with historical market data from Yahoo Finance.

The project compares classic tabular reinforcement learning agents and deep reinforcement learning agents in a simplified buy/sell/hold environment. It is designed as a learning and experimentation system, not as production trading software or financial advice.

<p align="center">
  <img src="best_performance/DQN_epsilon_greedy.png" alt="DQN epsilon-greedy portfolio value compared with buy-and-hold baseline" width="820">
</p>

<p align="center"><em>Example DQN + epsilon-greedy evaluation result against a buy-and-hold baseline.</em></p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Example Results](#example-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Research Limitations](#research-limitations)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

The goal of this project is to explore how reinforcement learning agents can learn trading behaviors from historical price and volume signals.

The environment is intentionally simple:

- **State**: price trend categories, volume state, and portfolio state.
- **Actions**: buy, sell, or hold.
- **Reward**: price movement and portfolio-aware trading outcome.
- **Baseline**: buy-and-hold strategy.
- **Output**: portfolio-value plots, text reports, learned Q-tables, and model weights.

This makes the repo useful for practicing RL concepts such as state design, reward shaping, exploration strategies, policy evaluation, and backtesting caveats.

---

## Features

- Fetches historical OHLCV data with [yfinance](https://finance.yahoo.com/).
- Converts raw stock data into discrete RL states.
- Supports classic tabular RL agents:
  - Q-learning
  - SARSA
- Supports deep RL agents:
  - Deep Q-Network, DQN
  - Deep SARSA
- Supports two action-selection strategies:
  - epsilon-greedy
  - softmax exploration
- Compares agent portfolio value against a buy-and-hold baseline.
- Saves generated outputs under local runtime directories:
  - `images/`
  - `documents/`
  - `model_weights/`
- Includes stored example artifacts under `best_performance/`.
- Includes pytest coverage for preprocessing, Q-table updates, action selection, and evaluation behavior.
- Includes GitHub Actions CI for branch pushes and pull requests.

---

## Architecture

```text
Yahoo Finance data
        |
        v
Data preprocessing
        |
        |-- trend states from moving averages
        |-- volume state from quantiles
        |-- portfolio state: empty / holding
        v
RL environment
        |
        |-- actions: buy / sell / hold
        |-- reward: price movement + portfolio outcome
        v
Agents
        |
        |-- Q-learning / SARSA
        |-- DQN / Deep SARSA
        v
Evaluation
        |
        |-- compare with buy-and-hold
        |-- save plots, reports, Q-tables, weights
```

---

## How It Works

### 1. Preprocess market data

The preprocessing pipeline keeps the core price columns and creates categorical state features:

```text
Open, Close, Trend_0, Trend_1, volume_state
```

Trend features are derived from moving averages, while `volume_state` is derived from volume quantiles.

### 2. Train agents

Classic agents use a tabular Q-table over combinations of:

```text
trend states × volume states × portfolio states
```

Deep agents one-hot encode the state representation and learn Q-values with TensorFlow/Keras models.

### 3. Evaluate against a baseline

The evaluation compares the learned policy with a buy-and-hold baseline using the same historical period.

Generated outputs are written locally so runtime artifacts do not have to be committed:

```text
images/
documents/
model_weights/
```

---

## Example Results

The repository includes stored example outputs in `best_performance/` for demonstration.

Example historical run on a tracked dataset:

| Agent | Final agent value | Buy-and-hold value | Notes |
|---|---:|---:|---|
| Q-learning + epsilon-greedy | 92,987.29 | 46,781.40 | Stored example artifact |
| DQN + epsilon-greedy | 95,501.89 | 46,827.80 | Stored example artifact |
| SARSA + softmax | 48,521.56 | 46,827.80 | Stored example artifact |

These results are useful for comparing experimental behavior, but they should not be interpreted as evidence of a profitable live-trading strategy. See [Research Limitations](#research-limitations).

Example screenshots:

![runtime example](./readme_images/img1.png)
![finished training](./readme_images/img2.png)
![portfolio plot](./readme_images/img3.png)
![text report](./readme_images/img4.png)

---

## Project Structure

```text
.
├── .github/workflows/ci.yml       # GitHub Actions CI
├── best_performance/              # Stored example outputs and weights
├── packages/
│   ├── agent.py                   # Tabular Q-learning and SARSA agent
│   ├── deep_learning_agent.py     # DQN and Deep SARSA agents
│   └── preprocess.py              # Data preprocessing and state construction
├── readme_images/                 # README screenshots
├── tests/                         # Unit tests
├── main.py                        # CLI entry point
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Locked uv dependency graph
├── requirements.txt               # pip-compatible dependency export
└── save_best.sh                   # Helper script for saving selected outputs
```

Runtime directories are intentionally git-ignored:

```text
images/
documents/
model_weights/
```

---

## Installation

### Prerequisites

- Python `>=3.12,<3.13`
- [uv](https://docs.astral.sh/uv/) recommended

### Clone

```bash
git clone https://github.com/StevenHuang41/RL-based_Stock_Trading_Support_System.git
cd RL-based_Stock_Trading_Support_System
```

### Install with uv

```bash
uv sync --locked
```

### Alternative: install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

Train and evaluate the agents for a Yahoo Finance ticker:

```bash
uv run python main.py AAPL
```

Taiwan market tickers can also be used, for example:

```bash
uv run python main.py 0050.TW
```

After training, generated artifacts are written to:

```text
images/
documents/
model_weights/
```

### Save selected best results

If a run produces a result you want to preserve, use:

```bash
bash save_best.sh
```

---

## Testing

Run the unit tests locally:

```bash
uv sync --dev --locked
uv run pytest
```

Current tests cover:

- preprocessing without making network calls;
- deep-agent one-hot preprocessing;
- Q-learning update behavior;
- SARSA next-action validation;
- deterministic evaluation-time action selection;
- portfolio evaluation behavior for a simple hold-only policy.

CI runs the same pytest suite on branch pushes and pull requests to `main`.

---

## Research Limitations

This project is an RL/backtesting learning project, not a trading recommendation system.

Important limitations:

- No transaction costs or brokerage fees are modeled.
- No slippage or market-impact assumptions are modeled.
- Historical performance may overfit the selected period.
- A high backtest return does not imply future profitability.
- The environment uses simplified state and action spaces.
- Real markets include liquidity, volatility regimes, survivorship bias, and execution constraints.
- Yahoo Finance data may contain missing data, adjustment issues, or delayed split information.

A production-grade trading research system would need stricter train/test splits, walk-forward validation, risk-adjusted metrics, transaction-cost modeling, and robust experiment tracking.

---

## Future Work

- Add transaction-cost and slippage modeling.
- Add result tables generated from evaluation artifacts.
- Add hyperparameter-search scripts with fixed seeds.
- Add Docker support for reproducible local execution.

---

## License

This project is licensed under the [MIT License](./LICENSE).
