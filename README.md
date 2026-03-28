# Portfolio Management Neural Network

A modular research repository for **mean-variance portfolio allocation with neural networks and reinforcement learning**.

This project is designed to support **fast iteration**, **clean experimentation**, and **reliable extension**. The first goal is to build a strong and testable core for portfolio allocation under a mean-variance objective, while keeping the codebase flexible enough to incorporate richer conditional information and additional learning modes over time.

---

## Overview

This repository focuses on **portfolio allocation**, not pricing or hedging, with the initial setting centered on:

- dynamic portfolio allocation under a **mean-variance objective**
- evaluation using broader risk and performance metrics
- support for **optional auxiliary conditioning information**
- both **offline** and **online** learning workflows
- modular components with **pytest-based verification**
- efficient training and evaluation for quick development cycles

The intent is to create a codebase that is practical for research and structured for repeated refinement using coding agents and human review.

---

## Project Goals

### Primary goals

- Build a clean baseline for **mean-variance portfolio optimization with neural networks**.
- Support **fast experiment cycles** with short feedback loops.
- Keep the codebase **modular, testable, and maintainable**.
- Allow the model to use **additional conditional inputs when available**, while remaining functional when some inputs are missing.
- Support both **offline learning** from historical or precomputed data and **online learning** from streaming or sequential updates.
- Maintain reasonable **memory and runtime efficiency** so that experiments remain practical on standard hardware.

### Secondary goals

- Compare neural and RL-based methods against simple classical baselines.
- Add richer financial context over time, such as macro features, factor signals, or option-implied information.
- Prepare the repository for systematic extension, benchmarking, and future research.

---

## Initial Scope

The initial version of the repository will focus on:

- **portfolio allocation only**
- a **mean-variance style reward/objective**
- neural policies that map observed state information to portfolio weights
- a compact training pipeline for quick validation
- strong synthetic and small-scale tests before expanding to larger datasets

The first version will **not** attempt to solve all related finance problems at once. In particular, pricing, hedging, full distributional modeling, and highly complex sequence architectures are outside the first implementation target.

---

## Design Principles

### 1. Modularity first
Each major concern should be isolated behind clear interfaces:

- data ingestion
- feature generation
- environments / simulators
- model definitions
- algorithms / trainers
- evaluation / backtesting
- configuration
- tests

### 2. Fast verification
Every important subsystem should be easy to validate with short tests, smoke runs, and deterministic toy examples.

### 3. Graceful handling of missing conditional inputs
The model should work with:

- only base state inputs
- base state plus additional context
- partially missing context
- fully missing optional context

The architecture should degrade gracefully rather than requiring every feature to be present.

### 4. Offline and online learning under one framework
Offline and online workflows should share the same core abstractions wherever possible. The code should not split into separate disconnected implementations.

### 5. Efficient by default
The baseline implementation should be compact, understandable, and lightweight enough to iterate quickly.

---

## Learning Setting

The repository is intended to support two related training modes.

### Offline learning
Train from historical or precomputed transition data.

Example use cases:

- replay from historical market observations
- training on stored trajectories
- quick reproducible experiments
- controlled ablation studies

### Online learning
Update the model sequentially as new information arrives.

Example use cases:

- rolling retraining
- periodic model updates
- sequential decision-making in a simulator
- adaptive policies under changing regimes

The long-term goal is for offline pretraining and online adaptation to work together cleanly.

---

## Model Philosophy

The default model family is intended to be:

- compact
- memory-conscious
- easy to benchmark
- easy to modify
- robust to partial input availability

A likely initial architecture is a small actor-critic or related policy/value setup with:

- a **base-state encoder**
- an **optional-context encoder**
- a **mask-aware input path**
- portfolio-weight output subject to allocation constraints

The first implementation should favor **clarity and stability** over architectural novelty.

---

## Inputs and Conditional Information

The repository will distinguish between:

### Base inputs
Information that is always expected to be available, such as:

- recent returns
- realized volatility features
- current wealth
- current portfolio state
- simple factor or market descriptors

### Optional conditional inputs
Additional context that may or may not be available, such as:

- macro variables
- cross-sectional signals
- sentiment-like indicators
- fundamental features
- option-implied quantities

Optional inputs should be accompanied by masking or equivalent mechanisms so the model can detect what is missing.

---

## Evaluation Metrics

Although the training objective is centered on mean-variance optimization, evaluation should be broader.

Planned metrics include:

- mean return
- volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- turnover
- realized transaction cost impact
- tail-risk metrics such as drawdown- or loss-based summaries
- stability across seeds and time splits

This separation between **training objective** and **evaluation metrics** is important: an improving training loss does not automatically imply a better portfolio strategy.

---

## Baselines

The repository should include simple reference points before more complex models are added.

Examples:

- equal-weight benchmark
- static mean-variance optimizer
- simple rule-based allocator
- lightweight supervised allocator
- compact RL baseline

The aim is to make improvements interpretable rather than relying on a single complex model.

---

## Testing Philosophy

Testing is a core feature of this project.

The repository should include:

### Unit tests
For:

- tensor shapes
- masking logic
- reward computation
- weight constraints
- replay buffer behavior
- loss computation

### Smoke tests
For:

- short training runs on toy data
- end-to-end pipeline checks
- deterministic seeded runs

### Regression tests
For:

- stable benchmark metrics on synthetic datasets
- basic runtime and memory checks
- prevention of silent training failures

The objective is not only correctness, but also fast confidence during iteration.

---

## Suggested Repository Structure

```text
portfolio-management-nn/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── base/
│   ├── experiments/
│   └── tests/
├── src/
│   ├── data/
│   ├── features/
│   ├── envs/
│   ├── models/
│   ├── algos/
│   ├── train/
│   ├── eval/
│   ├── backtest/
│   └── utils/
├── scripts/
│   ├── train_offline.py
│   ├── train_online.py
│   ├── evaluate.py
│   └── run_smoke_test.py
├── tests/
│   ├── unit/
│   ├── smoke/
│   └── regression/
├── notebooks/
└── outputs/
```

This layout may evolve, but the guiding idea is to keep training, evaluation, and data logic clearly separated.

---

## Development Priorities

A reasonable first build order is:

1. repository skeleton, packaging, and utility foundation
2. typed config and shared data contracts
3. masking support for optional conditional inputs
4. synthetic market environment and reward / constraint logic
5. compact neural allocation model and paper-aligned actor-critic baselines
6. offline training pipeline
7. online update pipeline
8. evaluation and backtesting utilities
9. historical data integration
10. richer conditional features and further extensions

---

## What This Repository Is Optimized For

This project is optimized for:

- repeated research iteration
- short implementation-test cycles
- clean comparison across experiments
- safe modification by coding agents and reviewers
- progressive expansion from simple baselines to richer models

It is **not** initially optimized for:

- large production trading infrastructure
- very large model architectures
- exhaustive asset universes on day one
- highly customized backtesting engines before the core learning loop is stable

---

## Long-Term Extension Directions

Possible future directions include:

- richer online learning schemes
- transaction cost modeling and execution constraints
- regime-aware or multi-frequency allocation
- option-implied state variables
- distribution-aware or robust portfolio optimization
- operator or functional representations of conditional state information
- extensions of continuous-time reinforcement learning frameworks

These are deliberately out of the first implementation scope, but the repository should be structured so that they can be added without major rewrites.

---

## Development Workflow

This project is intended to support iterative development with strong review loops.

General workflow goals:

- small, focused changes
- modular pull requests or patches
- quick automated checks
- clear experiment tracking
- repeatable seeded runs

Detailed agent-specific instructions, coding guidelines, and task templates will be added separately once the project goals and structure are fully settled.

### Environment note

If you keep this repository in an iCloud-backed location, treat `.venv/` as a local-only environment directory.

Recommended pattern:

```bash
python3.11 -m venv .venv
touch .venv/.noindex
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The goal is to keep package binaries and Python import trees out of normal cloud-sync/indexing paths. The environment should be recreated from `requirements.txt`, not relied on as synced project state.

---

## Status

This repository is in the middle of the baseline-assembly stage.

Implemented and verified so far:

- repository scaffold and packaging
- core utilities under `src/utils/`
- typed experiment configuration under `src/config/`
- shared data abstractions under `src/data/`
- optional-context masking and feature containers under `src/features/`
- synthetic GBM environment, portfolio constraints, and reward foundation
- analytic oracle benchmark under `src/algos/`
- actor / critic model interfaces under `src/models/`
- CTRL rollout collection and deterministic policy-evaluation scaffolding
- CTRL objective, re-evaluation, and scalar loss plumbing under `src/algos/ctrl.py`
- initial trainer-side CTRL slices under `src/train/`, including:
  - single-trajectory update step
  - fixed-length inner trainer run
  - single outer `w` update
  - single combined outer iteration
  - fixed-length outer-loop schedule
  - stateful trainer shell with validation, snapshot, history, and reset boundaries
  - in-memory checkpoint payload export/restore plus file-based save/load helpers
  - in-memory scalar logging-record extraction plus log-record file save/load helpers
- first evaluation-side slice under `src/eval/`, including:
  - typed deterministic evaluation summaries
- smoke and long-verification tooling under `scripts/`
- minimal CTRL demo script under `scripts/run_ctrl_demo.py`
- focused unit-test coverage, with the current unit suite at `555 passed`

Not yet implemented:

- full trainer-loop and experiment-management infrastructure beyond the current bounded trainer helpers
- offline / online trainers
- evaluation/backtesting infrastructure beyond the current single-episode deterministic summary helper
- plotting and reporting utilities

Current validation utilities include:

- `scripts/run_smoke_test.py` for fast integration checks
- `scripts/run_ctrl_demo.py` for a tiny end-to-end CTRL demo run
- `scripts/run_long_verification.py` for stage-based longer verification with saved output artifacts

If your local `.venv` exposes only `python3` rather than `python`, use whichever interpreter exists. For example:

```bash
.venv/bin/python3 -m pytest tests/unit -q
.venv/bin/python3 scripts/run_smoke_test.py
```

Relative to the reference notes, the repository now covers the paper-aligned environment/model/oracle foundation, the approved CTRL math plumbing, and the first bounded trainer-side integration slices, but it still stops short of full trainer-loop implementation of the papers' learning procedures.

---

## Summary

This repository aims to provide a strong, modular foundation for **mean-variance portfolio allocation with neural networks**, with special attention to:

- quick iteration
- testability
- optional conditional information
- offline and online learning
- clean extensibility
- efficiency in both memory and runtime

The first objective is not to build the most complex model possible, but to build a foundation that can be trusted, tested, and improved rapidly.
