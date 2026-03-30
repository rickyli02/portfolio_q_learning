# Portfolio Q-Learning

A research repository for mean-variance portfolio allocation with neural and continuous-time-inspired reinforcement-learning baselines.

The current codebase is centered on a synthetic GBM environment, a Zhou-Li oracle benchmark, a Huang-Jia-Zhou-style CTRL baseline, deterministic evaluation and backtest helpers, and strong pytest coverage so changes can be made safely.

## Current Status

As of 2026-03-30, the repository includes:

- typed config, data, and feature foundations under `src/config/`, `src/data/`, and `src/features/`
- a synthetic GBM portfolio environment plus portfolio-constraint utilities under `src/envs/`
- analytic oracle mean-variance policy code under `src/algos/oracle_mv.py`
- Gaussian actor and quadratic critic implementations under `src/models/`
- CTRL rollout, objective, loss, deterministic execution, and trainer-state code under `src/algos/` and `src/train/`
- deterministic evaluation records, aggregates, bundles, and IO helpers under `src/eval/`
- CTRL-vs-oracle comparison, train-and-compare bridging, compact scalar reporting, and a config-backed experiment runner under `src/backtest/`
- diagnostic dtype-comparison utilities under `src/utils/dtype_compare.py`
- small demo, verification, and YAML-driven experiment entrypoints under `scripts/`

The repo now makes the policy-role split explicit:

- training data collection uses a stochastic behavior policy
- evaluation and comparison use a deterministic execution policy

## Quick Start

Create a local virtual environment and install dependencies:

```bash
python3.11 -m venv .venv
touch .venv/.noindex
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the main verification entrypoints:

```bash
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/run_smoke_test.py
.venv/bin/python3 scripts/run_ctrl_demo.py
.venv/bin/python3 scripts/run_ctrl_oracle_demo.py
.venv/bin/python3 scripts/run_config_experiment.py configs/experiments/ctrl_baseline_tiny.yaml
```

If your local environment exposes `.venv/bin/python` instead of `.venv/bin/python3`, use whichever interpreter exists.

## Repository Map

```text
portfolio_q_learning/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── experiments/
├── outputs/
├── references/
├── scripts/
│   ├── run_ctrl_demo.py
│   ├── run_ctrl_oracle_demo.py
│   ├── run_config_experiment.py
│   ├── run_dtype_compare_demo.py
│   ├── run_long_verification.py
│   └── run_smoke_test.py
├── shared_agent_files/
│   ├── claude_code_todo.md
│   ├── dialogue.txt
│   └── dialogue_brainstorm.txt
├── src/
│   ├── algos/
│   ├── backtest/
│   ├── config/
│   ├── data/
│   ├── envs/
│   ├── eval/
│   ├── features/
│   ├── models/
│   ├── train/
│   └── utils/
└── tests/
    ├── smoke/
    ├── unit/
    └── regression/
```

## Verification Snapshot

Current directly checked results on this branch:

- latest directly checked full-suite snapshot remains historical: `.venv/bin/python3 -m pytest -q` -> `895 passed, 1 warning`
- `.venv/bin/python3 scripts/run_smoke_test.py` -> `9/9 passed`
- `.venv/bin/python3 -m pytest tests/unit/test_demo_scripts.py -q --tb=short` -> `14 passed`
- `.venv/bin/python3 -m pytest tests/unit/test_policy_contract.py -q --tb=short` -> `13 passed`
- `.venv/bin/python3 -m pytest tests/unit/test_config.py -q --tb=short` -> `33 passed`
- `.venv/bin/python3 -m pytest tests/unit/test_experiment_runner.py -q --tb=short` -> `22 passed`
- `.venv/bin/python3 -m pytest tests/unit/test_config_experiment_script.py -q --tb=short` -> `20 passed`
- `.venv/bin/python3 scripts/run_config_experiment.py configs/experiments/ctrl_baseline_tiny.yaml` -> exits `0`

The single current warning in the full suite is from the intentional numerics-boundary test around `QuadraticCritic.quad`.

## What Exists Today

- `src/algos/ctrl.py` contains the approved CTRL math plumbing for stochastic rollout collection and deterministic execution evaluation.
- `src/train/` contains the stateful trainer shell, outer-loop execution helpers, snapshots, history, checkpoint IO, and log-record IO.
- `src/eval/` contains deterministic summary, aggregate, record, record-set, scalar-derivation, and bundle helpers.
- `src/backtest/comparison.py` compares CTRL and oracle policies over explicit evaluation seeds.
- `src/backtest/train_compare.py` and `src/backtest/train_compare_report.py` provide a train-then-compare seam plus a compact scalar report.
- `src/backtest/experiment_runner.py` provides a config-backed experiment runner for the currently supported GBM + CTRL baseline path.
- `scripts/run_config_experiment.py` loads a YAML config, runs the approved experiment runner, and prints a stable scalar summary for manual use.

## Not Yet Implemented

- saved experiment/report artifacts beyond console summaries
- real-data ingestion paths such as WRDS
- plotting/report artifact generation beyond console summaries
- transaction-cost-aware evaluation and execution realism
- richer synthetic stress environments beyond the current GBM baseline
- the later practical online improvements from the reference papers

## Working Notes

Use these project files for orientation:

- `shared_agent_files/claude_code_todo.md` for the roadmap and implementation snapshot
- `shared_agent_files/dialogue.txt` for the current bounded task and review history
- `shared_agent_files/dialogue_brainstorm.txt` for non-executable design discussion
- `references/portfolio_mv_papers_algorithm_summary.md` and `references/portfolio_mv_papers_companion_implementation_notes.md` for paper-to-code notes

This repository is optimized for research iteration and verification, not production trading infrastructure.
