# CLAUDE.md — Portfolio Q-Learning

## Project summary

Modular research repository for **mean-variance portfolio allocation with neural networks and continuous-time RL**. Supports both offline and online learning, with optional conditional inputs that may be missing at inference time.

**Current state:** Phase 2 foundation is partially implemented, but the coordination state is not yet clean.

Implemented now:
- `requirements.txt`, `pyproject.toml`, `.gitignore`
- package scaffold under `src/` with subpackages for `data`, `features`, `envs`, `models`, `algos`, `train`, `eval`, `backtest`, and `utils`
- initial utility modules in `src/utils/`: `seed.py`, `device.py`, `io.py`, `paths.py`, `logging.py`
- config scaffold under `src/config/` and YAML configs under `configs/`
- data-layer scaffold under `src/data/`
- utility, config, and data-layer tests under `tests/`
- `scripts/run_smoke_test.py`
- placeholder directory structure for `outputs/`

Not implemented yet:
- environments
- reward/objective logic
- model/algorithm/trainer/eval code

Current caution:
- The repo has moved ahead into Phase 2 work, but documentation, dialogue discipline, and review gating must be tightened before additional implementation proceeds.

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
Frequently used skills or commands should be saved in `.claude/skills` and `.claude/commands`.
When additional planning is required, ask questions in `dialogue.txt`.
Commit only when the user explicitly requests a commit.

## Agent governance

- `.claude/CLAUDE.md` and `.claude/commands/` are repo-owned collaboration files and may be maintained by Codex in its project-manager role.
- `.claude/settings.json` and `.codex/config.toml` are user-owned. Do not edit them directly. Request changes through `shared_agent_files/dialogue.txt`.
- Use `shared_agent_files/dialogue.txt` for decisions, blockers, and requests that affect other agents or user-owned config.
- Agents may create planning or memory notes in `claude_files/` and `codex_files/`.
- All scripts and tests must be run in the project `.venv` environment.

## Dialogue Protocol

Every Claude implementation block must be governed through `shared_agent_files/dialogue.txt`.

### Task assignment syntax

Use this form when Codex or the user assigns a bounded task to Claude:

```text
[Codex | YYYY-MM-DD HH:MM:SS TZ]
To Claude:
- Task: <short task name>
- Scope: <what Claude is allowed to change>
- Expected files: <files or folders likely to change>
- Verification: <tests/scripts Claude should run in .venv>
- Stop conditions: <task-specific stop rules>
- Review required: <Codex, user, or none>
- Go/No-Go: GO
```

### Stop-condition syntax

Use this form when Codex or the user wants explicit stop rules for the task:

```text
- Stop conditions:
  1. <condition that must halt implementation>
  2. <condition that requires dialogue update before proceeding>
  3. <condition that requires user or Codex review>
```

### Post-change verification request syntax

Claude should request verification in dialogue with this form after finishing a bounded task block:

```text
[Claude Code | YYYY-MM-DD HH:MM:SS TZ]
Verification request:
- Task completed: <task name>
- Files changed: <files>
- Verification run in .venv: <commands and results>
- Known risks / open questions: <items>
- Requesting review from: <Codex or user>
```

### Stop-event syntax

If Claude encounters a stop condition, Claude must log it immediately in dialogue:

```text
[Claude Code | YYYY-MM-DD HH:MM:SS TZ]
Stop event:
- Active task: <task name>
- Triggered condition: <which stop condition fired>
- Reason: <why work stopped>
- Files touched so far: <files or none>
- Requested next input: <what Claude needs from Codex or user>
```

## Generic Stop Rules

Claude must stop and post a `Stop event` in dialogue when any of the following occurs:

- the next implementation step would begin a new phase or a materially larger scope than what was explicitly assigned
- documentation or governance files are stale enough that they would mislead further implementation
- tests or smoke checks required for the active task fail
- a requested verification step cannot be run in the project `.venv`
- a contradiction appears between repo state, dialogue instructions, and current task scope
- the task would require editing a user-owned file
- the correct next step depends on a design choice that has not yet been approved by Codex or the user
- Claude has completed one bounded task block and the task was marked as requiring review before continuing

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

Run `scripts/run_smoke_test.py` from the project `.venv` once the script exists.

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
Current test coverage is packaging/utilities only; do not treat a green test suite yet as evidence that research logic is correct.
Current governance rule: green tests do not authorize Claude to continue into the next task block unless the dialogue entry for the task allowed continued execution without review.

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

- `shared_agent_files/dialogue.txt` — communication log between Claude Code and Codex agents; record identity and datetime when writing
- dialogue entries must include a full timestamp, must be appended at the bottom of the file, and must follow the task / stop / verification conventions above
- Codex agent writes to `codex_files/`, Claude Code writes to `claude_files/`
- Both agents may read/write `shared_agent_files/`

---

## Python environment

- Python 3.11
- Virtual environment: `python3.11 -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Initial dependencies: `numpy scipy pandas torch pyyaml pydantic tqdm matplotlib pytest`
