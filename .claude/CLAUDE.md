# CLAUDE.md — Portfolio Q-Learning

## Project summary

Modular research repository for **mean-variance portfolio allocation with neural networks and continuous-time RL**. Supports both offline and online learning, with optional conditional inputs that may be missing at inference time.

**Current state:** scaffold only — `src/`, `tests/`, `scripts/`, `configs/` directories exist but contain no source files. Implementation has not begun.

---

## Primary reference hierarchy

Use sources in this order:

1. Codebase files already in this repository
2. `shared_agent_files/claude_code_todo.md` — implementation roadmap and tasking document (read this first when starting work)
3. `shared_agent_files/setup_and_utilities_guidelines.md` — setup, pytest, utilities, style standards
4. `references/portfolio_mv_papers_algorithm_summary.md` — algorithm summaries for EMV and CTRL
5. `references/portfolio_mv_papers_companion_implementation_notes.md` — implementation details, notation clashes, update-order guidance
6. `references/papers/` — original PDFs, only when the markdown summaries are insufficient

Do not repeatedly re-read the original papers if the markdown summaries already contain the implementation detail.

---

## Task order

Execute in roughly this sequence — do not jump ahead:

1. Repository scaffold (`requirements.txt`, `.gitignore`, `pyproject.toml`, `__init__.py` files)
2. Utilities foundation (`src/utils/`: seed, device, io, paths, logging)
3. Config system (typed config objects or YAML-backed parsing)
4. Data layer (synthetic datasets, replay buffer, transitions)
5. Features and masking (`src/features/`)
6. Synthetic environments (`src/envs/`, start with GBM)
7. Reward and objectives
8. Model interfaces (`src/models/base.py`)
9. Baseline models (Gaussian actor, compact critic)
10. Algorithms: oracle MV → EMV → CTRL (`src/algos/`)
11. Offline trainer (`src/train/offline_trainer.py`)
12. Online trainer (`src/train/online_trainer.py`)
13. Evaluation and backtest (`src/eval/`, `src/backtest/`)
14. Test suite (`tests/unit/`, `tests/smoke/`, `tests/regression/`)
15. Efficiency and quality pass

Run `scripts/run_smoke_test.py` frequently before marking phases complete.

---

## Architecture principles

- Small, single-purpose modules — no monolithic training scripts
- Offline and online trainers share the same core interfaces and batch schema
- Optional conditional features use explicit masks — model must run when all optional inputs are missing
- Training objective (mean-variance) must stay separate from evaluation metrics
- Every major component must be testable in isolation
- Future model classes can be swapped without rewriting the training loop

**First build scope:** mean-variance objective, Gaussian behavior policy, deterministic execution policy, multi-asset allocation, optional conditional feature masking, leverage/weight-constraint wrappers.

**Do not start with:** transformers, neural operators, large sequence models, distributed backtesting, complex microstructure simulators.

---

## Implementation style

Use:
- Python type hints on all public APIs
- Docstrings on public modules, classes, and functions
- Dataclasses or typed config objects
- Deterministic seeding utilities
- Explicit device placement
- Numerically stable tensor operations
- `pathlib.Path` for all file paths

Avoid:
- Large notebook-only logic
- Hidden global state
- Duplicate implementations for offline vs online paths
- Silent fallback behavior
- Over-engineering before the first working baseline

---

## Config naming conventions

Avoid ambiguous single-letter names. Use:
- `entropy_temp` (not `lambda`)
- `trace_decay`
- `jump_intensity`
- `target_return`
- `rebalance_interval`
- `online_update_interval`

---

## Expected repository structure

```
repo_root/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base/
│   ├── experiments/
│   └── tests/
├── src/
│   ├── __init__.py
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
└── outputs/
```

---

## Testing

Framework: pytest. Run from repo root.

- **Unit:** tensor shapes, masking logic, reward functions, environment stepping, log-prob/entropy, covariance validity, replay buffer
- **Smoke:** tiny end-to-end offline run, tiny end-to-end online run, deterministic seeded GBM run, one evaluation pass
- **Regression:** fixed-seed benchmark outputs, stable metrics, no severe runtime/memory regressions

```bash
pytest
pytest tests/unit
pytest tests/smoke -q
```

Tests must be deterministic. Smoke tests must stay short. Prefer synthetic data over large data files.

---

## Utilities (`src/utils/`)

Cross-cutting helpers only:
- `seed.py` — deterministic seeding
- `device.py` — device selection
- `io.py` — checkpoint save/load
- `paths.py` — output directory creation, timestamped paths
- `logging.py` — training step, losses, entropy, wealth stats, eval metrics, runtime, seed

Do not put portfolio reward logic, environment stepping, actor/critic code, or algorithm update rules in `utils/`.

---

## Output management

```
outputs/
└── experiment_name/
    └── run_YYYYMMDD_HHMMSS/
        ├── config_snapshot.yaml
        ├── logs/
        ├── checkpoints/
        └── results/
```

Use `pathlib.Path`. No hardcoded absolute paths. Outputs, checkpoints, and logs are gitignored.

---

## Definition of done (first usable version)

- Repository installs and imports cleanly
- `pytest` passes from repo root
- Synthetic GBM environment works
- Gaussian behavior policy works
- Deterministic execution policy works
- Offline trainer works
- Online trainer works
- Evaluation metrics work
- Optional context masking works in at least one model path
- `scripts/run_smoke_test.py` passes
- Outputs save reproducibly under fixed seed

---

## Agent coordination

- `shared_agent_files/dialogue.txt` — communication log between Claude Code and Codex agents; record identity and timestamp when writing
- Codex agent writes to `codex_files/`, Claude Code writes to `claude_files/`
- Both agents may read/write `shared_agent_files/`

---

## Python environment

- Python 3.11
- Virtual environment: `python3.11 -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Initial dependencies: `numpy scipy pandas torch pyyaml pydantic tqdm matplotlib pytest`
