# Claude Code TODO and Implementation Guide

This file is intended as the primary tasking document for Claude Code while building the repository.

The project goal is to build a **modular, testable, memory-conscious repository for mean-variance portfolio allocation with neural networks and continuous-time-inspired reinforcement learning**, with support for both **offline** and **online** learning, and with **optional conditional inputs** that may be missing at inference or training time.

This file should be treated as an implementation roadmap, not a paper note.

## Current snapshot

As of 2026-03-27:

- Phase 1 scaffold exists.
- Phase 2 config/data foundation exists.
- Phase 2 bugfix cleanup is complete.
- Phase 3A features and masking foundation under `src/features/` is complete and verified.
- Phase 3B environment foundation and constraint layer are approved.
- Phase 4A config schema extension for algorithm selection and plotting is complete and verified.
- Phase 5A oracle benchmark core is approved.
- Phase 6A actor / critic model interface foundation is approved.
- Phase 7A CTRL rollout / deterministic-evaluation scaffolding is complete.
- Phase 7B CTRL smoke integration and long-verification tooling is approved.
- Phase 8A CTRL objective and loss-primitive foundation is approved.
- Phase 8B CTRL gradient-tracked re-evaluation foundation is approved.
- Phase 8C CTRL scalar loss assembly foundation is approved.
- Phase 9A CTRL single-trajectory trainer step foundation is approved.
- Phase 9B CTRL fixed-length trainer-run foundation is approved.
- Phase 10A CTRL outer-loop w update primitive is approved.
- Phase 10B CTRL single outer-iteration foundation is approved.
- Latest verified outputs with current repo state are:
  - model-focused verification:
    - `tests/unit/test_models.py -q -> 49 passed`
  - current full unit-suite verification:
    - `tests/unit -q -> 386 passed`
  - current trainer-step verification:
    - `tests/unit/test_ctrl_trainer.py -q -> 67 passed`
  - current smoke verification:
    - `scripts/run_smoke_test.py -> 7/7 passed`
  - current long-verification artifacts:
    - `outputs/verification/2026-03-27_141812_verification.txt`
    - `outputs/verification/2026-03-27_143247_verification.txt`
  - current normalized import-timing artifact:
    - `numpy ~= 0.069s`
    - `torch ~= 0.859s`
- The currently active bounded task in dialogue is:
  - Phase 10C CTRL fixed-length outer-loop schedule foundation
- The active RL implementation target is now:
  - oracle benchmark from known synthetic parameters first
  - Huang–Jia–Zhou (2025) theorem-aligned CTRL baseline next
  - Huang–Jia–Zhou (2022) / 2025 practical online improvements after baseline stability
- Wang–Zhou (2019/2020) remains a mathematical and derivational reference, not a required implementation layer.
- Plotting and visualization remain required implementation tasks, and the config schema now includes `PlottingConfig` for YAML-driven plotting preferences.
- The config schema now also includes algorithm selection via `AlgorithmConfig`.
- Pending follow-up TODO items identified by review and not yet implemented:
  - expand the CTRL pseudocode note with a more explicit trace-formula pointer from the companion notes
  - add concrete memory-pressure capture guidance to logging / plotting work
  - normalize trainer/test module naming and headers before broader trainer infrastructure hardens
- Phase 2 planning notes were archived to:
  - `references/archive/2026-03-26_phase2_execution_brief.md`
  - `codex_files/archive/2026-03-26_phase2_manager_notes.md`
- New work beyond the currently assigned bounded task blocks requires a formal GO task assignment in `shared_agent_files/dialogue.txt`.
- Use `.claude/CLAUDE.md` only for stable baseline rules. Use this file plus dialogue for current-state details.
- For `.venv` command examples, use whichever interpreter exists locally:
  - `.venv/bin/python`
  - or `.venv/bin/python3`
  and prefer `-m pytest` style invocation when command-entrypoint layout differs across machines.

### Numerical safety planning (captured 2026-03-27, pre-CTRL implementation)

Before the CTRL algorithm layer depends on `src/models/`, the following failure
modes must be addressed in a future bounded task.  Do not add silent clamps.
Per Ricky's guidance: operations that may fail or diverge should **log/warn with
the offending values** and/or **raise informative errors** — not silently correct.

**Operations at risk in current model code:**

| Location | Operation | Failure mode |
|---|---|---|
| `gaussian_actor.py` | `exp(log_phi1)` | overflow → φ₁ → inf → mean_action inf |
| `gaussian_actor.py` | `exp(-log_phi2_inv)` | overflow/underflow → φ₂ → inf or 0 |
| `gaussian_actor.py` | `phi2 * exp(phi3 * time_to_go)` | variance overflow (phi3 > 0 large) or underflow to 0 (phi3 < 0 large) |
| `gaussian_actor.py` | `log(var)` in entropy | -inf if variance underflows to float32 denormal |
| `quadratic_critic.py` | `exp(-theta3 * time_to_go)` | overflow if theta3 < 0 and time_to_go large |
| `oracle_mv.py` | `linalg.solve(cov, B)` | LinAlgError on singular/near-singular cov |

**Planned guard approach (implement in future bounded task):**
- Add `warn_if_unstable(tensor, name, threshold)` utility in `src/utils/` that
  calls `warnings.warn` with the offending values when `tensor.abs().max() > threshold`
  or when `tensor` contains inf/nan.  Do not clamp; just warn.
- Call this utility from model `forward()` / property accessors on computed
  intermediate values (not on raw parameters, to avoid noise during early training).
- For `oracle_mv.py`, wrap `linalg.solve` in a try/except that re-raises
  `ValueError` with the condition number and input mu/sigma values.
- Consider adding a `validate_parameters()` method to `ActorBase`/`CriticBase`
  that checks for inf/nan in `self.parameters()` and raises `ValueError`.
  This is cheap to call at the start of each training iteration.
- PSD/PD check guidance: before calling `linalg.solve` or `linalg.cholesky`,
  check `torch.linalg.eigvalsh(cov).min() > eps` and warn if conditioning is poor.
- Define threshold convention: absolute values > 1e6 or inf/nan warrant a warning;
  only inf/nan warrant an immediate raise in forward passes.

Historical roadmap blocks below may describe tasks that are already complete. Treat this snapshot plus dialogue as the source of truth for current state.

---

## 0. Primary reference hierarchy

Claude Code should use the following sources in this order:

1. **Codebase files already created in this repository**
2. **Markdown references in `/references/`**
3. **Original papers in `/references/papers/` only when necessary**

### Primary markdown references
Read these first before querying any paper PDFs:

- `/references/portfolio_mv_papers_algorithm_summary.md`
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

### Secondary repository references
Use these when present:

- `/README.md`
- `/references/README_portfolio_management_nn.md` if it exists in the repo
- config files
- existing test files
- docstrings and type hints already present in the codebase

### Instruction on paper usage
Minimize direct queries to the original papers in `/references/papers/`.

Use the papers only when:
- a formula is missing from the markdown summaries,
- there is a notation ambiguity that materially affects implementation,
- a theorem-backed baseline detail is needed for a specific module,
- a discrepancy appears between code and the markdown summaries.

Do **not** repeatedly re-read the papers if the markdown summaries already contain the implementation detail.

---

## 1. General coding guidelines

### 1.1 Architectural principles
Claude Code should preserve the following principles:

- keep modules small and single-purpose,
- separate research logic from engineering utilities,
- avoid monolithic training scripts,
- prefer simple baseline implementations first,
- make every major component testable in isolation,
- keep offline and online learning under shared interfaces where possible,
- support optional missing conditional features through explicit masks or equivalent mechanisms,
- write code so future model classes can be swapped without rewriting the training loop.

### 1.2 Implementation style
Use:

- Python type hints everywhere practical,
- concise docstrings for public functions/classes,
- dataclasses or typed config objects where appropriate,
- deterministic seeding utilities,
- explicit device placement,
- numerically stable tensor operations,
- minimal hidden state.

Avoid:

- large notebook-only logic,
- hidden global state,
- duplicate implementations for offline vs online paths,
- silent fallback behavior,
- over-engineering before the first working baseline.

### 1.3 Research implementation priorities
Prioritize in this order:

1. correctness,
2. modularity,
3. testability,
4. runtime/memory efficiency,
5. extensibility,
6. elegance.

### 1.4 Baseline mathematical scope
The first build should support:

- mean-variance objective,
- stochastic Gaussian behavior policy,
- deterministic execution policy when desired,
- offline learning,
- online learning,
- multi-asset portfolio allocation,
- optional conditional feature inputs,
- leverage / weight-constraint wrappers as modular post-processing layers.

Do not start with:
- full option-surface encoders,
- transformers,
- neural operators,
- production-grade distributed backtesting,
- complex market microstructure simulators.

---

## 2. Expected repository structure

Claude Code should build toward this structure unless a small deviation is clearly better:

```text
repo_root/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .venv/                      # local environment, ignored by git
├── references/
│   ├── portfolio_mv_papers_algorithm_summary.md
│   ├── portfolio_mv_papers_companion_implementation_notes.md
│   └── papers/
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
│   ├── eval/                  # metrics, evaluator, plots
│   ├── backtest/
│   └── utils/
├── scripts/
│   ├── train_offline.py
│   ├── train_online.py
│   ├── evaluate.py
│   ├── plot_results.py
│   └── run_smoke_test.py
├── tests/
│   ├── unit/
│   ├── smoke/
│   └── regression/
└── outputs/
```

If Claude Code creates additional folders, they should be justified by a real separation of concerns.

---

## 3. Global task order

Claude Code should execute work in roughly this order:

1. scaffold repository structure,
2. create setup files (`requirements.txt`, `.gitignore`, `pyproject.toml`),
3. create utilities foundation,
4. create config system,
5. create feature / masking support and synthetic environment foundations,
6. create baseline portfolio constraints and reward logic,
7. create the analytic oracle benchmark for synthetic known-parameter cases,
8. create base model interfaces,
9. create the Huang–Jia–Zhou (2025) baseline CTRL actor / critic,
10. create the modified online CTRL path with 2022/2025 practical improvements,
11. create evaluation, backtest, and plotting logic,
12. create test suite,
13. create configuration-driven experiment / plotting scripts,
14. refine for efficiency and code quality.

Do not skip directly to complex algorithms before the scaffolding and tests exist.

---

## 4. Functionality block: setup and project scaffold

### Goal
Create the initial repository structure and packaging setup.

### Tasks
- create folder tree under `src/`, `configs/`, `scripts/`, and `tests/`,
- create `__init__.py` files where needed,
- create `README.md` if not already present,
- create `requirements.txt`,
- create `.gitignore`,
- create `pyproject.toml`,
- ensure imports work from the project root.

### Acceptance criteria
- `python -m pytest` runs without import-path failures,
- top-level package imports resolve,
- no circular imports in scaffolding,
- repository can be installed or used in editable local mode.

### Reference files
- `/references/portfolio_mv_papers_algorithm_summary.md`
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

---

## 5. Functionality block: configs

### Goal
Create a clean, typed configuration system for experiments.

### Tasks
Create config objects or YAML-backed config parsing for:

- environment settings,
- asset universe settings,
- reward and mean-variance coefficients,
- entropy/exploration settings,
- trace decay settings,
- training loop settings,
- batch sizes,
- learning rates,
- evaluation settings,
- logging/output settings,
- random seeds,
- behavior-policy vs execution-policy settings.

### Requirements
- avoid ambiguous names such as a single `lambda`,
- use names like:
  - `entropy_temp`,
  - `trace_decay`,
  - `jump_intensity`,
  - `target_return`,
  - `rebalance_interval`,
  - `online_update_interval`.

### Acceptance criteria
- one base config can run end-to-end smoke tests,
- experiment config overrides are simple,
- config objects validate required fields,
- config names are consistent with code and tests.

### Reference files
- `/references/portfolio_mv_papers_companion_implementation_notes.md`
- especially notation and ambiguity notes

---

## 6. Functionality block: data layer

### Goal
Create the data abstractions needed for offline and online training.

### Tasks
Implement:

- synthetic dataset generation,
- optional historical dataset adapters,
- trajectory containers,
- replay buffer or transition buffer interfaces,
- batch samplers,
- normalization hooks if needed,
- deterministic seed-aware synthetic path generation.

### Suggested modules
- `src/data/datasets.py`
- `src/data/replay_buffer.py`
- `src/data/synthetic.py`
- `src/data/types.py`

### Important design requirements
- support both step-wise transitions and full trajectories,
- define a stable batch schema,
- keep tensors compact,
- avoid storing redundant derived values if they can be recomputed cheaply,
- make offline and online modes return compatible transition structures.

### Acceptance criteria
- offline trainer and online trainer can consume a shared batch protocol,
- replay buffer unit tests pass,
- synthetic data generation is deterministic under fixed seed.

### Reference files
- use markdown summaries first,
- only query papers for unresolved notation about state/action definitions.

---

## 7. Functionality block: features and optional conditional inputs

### Goal
Support base features plus optional auxiliary conditioning information with missingness.

### Tasks
Implement:

- base state feature builder,
- optional context feature builder,
- context mask generation,
- missing-value handling strategy,
- feature-group dropout for robustness during training if added later,
- consistent tensor shapes for base input, optional context, and mask.

### Suggested modules
- `src/features/base_features.py`
- `src/features/context_features.py`
- `src/features/masking.py`

### Requirements
- the model must still run when all optional features are missing,
- mask semantics must be explicit and tested,
- base-only and all-missing-context cases should be easy to compare in tests.

### Acceptance criteria
- forward pass works with:
  - no optional features,
  - partially missing optional features,
  - fully present optional features,
- tests verify shape stability and no crash behavior.

### Reference files
- `/references/README_portfolio_management_nn.md` if present
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

---

## 8. Functionality block: environments

### Goal
Create synthetic portfolio environments that support initial experiments and theorem-backed sanity checks.

### Priority order
1. multi-asset Black-Scholes / GBM environment,
2. leverage-constrained variant,
3. optional regime-switching or factor-driven environment,
4. optional jump-diffusion environment later.

### Tasks
Implement:

- portfolio environment with explicit nominal-vs-discounted wealth choice,
- transition stepping,
- reward computation,
- action application and rebalance timing,
- stochastic behavior-policy stepping,
- deterministic execution evaluation mode,
- optional transaction cost hooks,
- optional risk-free asset inclusion/exclusion modes.

### Suggested modules
- `src/envs/base_env.py`
- `src/envs/gbm_env.py`
- `src/envs/jump_env.py` later
- `src/envs/constraints.py`

### Important requirements
- action convention must be documented clearly:
  - dollar allocation or portfolio weights,
- discounted vs nominal wealth must be explicit,
  - current `GBMPortfolioEnv` uses nominal wealth,
- execution schedule must be separable from learning update schedule.

### Acceptance criteria
- small deterministic smoke run passes,
- environment produces transitions compatible with trainers,
- leverage wrapper works and is separately tested.

### Reference files
- `/references/portfolio_mv_papers_algorithm_summary.md`
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

---

## 9. Functionality block: reward and objectives

### Goal
Implement the mean-variance objective cleanly and separately from evaluation metrics.

### Tasks
Implement:

- terminal mean-variance loss or proxy,
- optional entropy regularization terms,
- helper functions for:
  - terminal wealth target penalty,
  - variance-like penalties,
  - entropy contribution,
  - per-step approximation if required.

### Requirements
- keep training objective separate from reporting metrics,
- allow the Lagrange multiplier `w` or related constraint parameter to be updated on a slower schedule,
- document exact discretization convention used.

### Acceptance criteria
- unit tests verify finite outputs,
- unit tests verify expected signs and dimensions,
- offline and online trainers call the same objective functions.

### Reference files
- `/references/portfolio_mv_papers_algorithm_summary.md`
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

---

## 10. Functionality block: model interfaces

### Goal
Define stable interfaces before implementing specific model classes.

### Tasks
Create interfaces or abstract base classes for:

- actor / behavior policy,
- execution policy,
- critic / value model,
- optional joint actor-critic wrappers,
- portfolio projection / constraint wrapper.

### Suggested modules
- `src/models/base.py`
- `src/models/policy_base.py`
- `src/models/value_base.py`

### Requirements
The actor interface should support:
- stochastic sampling,
- deterministic action extraction,
- log-prob computation,
- entropy computation,
- optional mask-aware forward pass.

The critic interface should support:
- value evaluation,
- parameter access for training,
- optional structured parameterizations.

### Acceptance criteria
- trainer does not depend on one concrete model implementation,
- interfaces are tested with toy implementations.

### Reference files
- `/references/portfolio_mv_papers_algorithm_summary.md`
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

---

## 11. Functionality block: baseline model implementations

### Goal
Implement the first working model family.

### Phase 1 preferred models
- compact Gaussian actor,
- compact critic,
- optional quadratic critic for theorem-aligned baselines,
- optional masked MLP encoders for richer state inputs.

### Suggested modules
- `src/models/gaussian_policy.py`
- `src/models/quadratic_value.py`
- `src/models/masked_mlp.py`

### Requirements
- stochastic behavior policy and deterministic execution policy must be separable,
- covariance must be parameterized safely,
- avoid naive unconstrained covariance matrices,
- consider precision or Cholesky parameterization,
- preserve a simple pathway for theorem-aligned Black-Scholes baselines.

### Acceptance criteria
- actor sampling works,
- deterministic extraction works,
- log-prob and entropy are correct up to basic tests,
- covariance remains valid under repeated updates.

### Reference files
- `/references/portfolio_mv_papers_companion_implementation_notes.md`
- use the paper PDFs only if covariance-update details are not sufficiently clear from the summaries.

---

## 12. Functionality block: algorithms

### Goal
Implement learning algorithms incrementally, not all at once.

### Phase order
1. oracle / analytic benchmark layer,
2. simple offline mean-variance sanity baseline,
3. CTRL-style actor-critic baseline from Huang–Jia–Zhou (2025),
4. modified online CTRL variant with 2022/2025 practical improvements,
5. jump-diffusion reuse later.

Do not make Wang–Zhou (2019/2020) EMV a required implementation layer in this plan.
Use it only as a derivational and mathematical reference for the exploratory Gaussian structure and the role of the outer-loop $w$ update.

### Suggested modules
- `src/algos/oracle_mv.py`
- `src/algos/ctrl.py`
- `src/algos/ctrl_online.py`

### Requirements
- mark each algorithm clearly as:
  - analytic benchmark,
  - theorem-aligned baseline research implementation,
  - practical online variant,
- document which pieces are theorem-backed and which are engineering choices.
- The oracle strategy must be callable through configuration in synthetic-data runs.
- The CTRL baseline should expose both stochastic behavior-policy rollout and deterministic execution-policy evaluation.

### Acceptance criteria
- each algorithm has at least one smoke test,
- baseline configs run end-to-end,
- logs are interpretable.

### Reference files
- Oracle, CTRL, and practical-modification details in:
  - `/references/portfolio_mv_papers_algorithm_summary.md`
  - `/references/portfolio_mv_papers_companion_implementation_notes.md`
  - `/references/portfolio_mv_ctrl_complete_pseudocode.md`

---

## 13. Functionality block: training modules

### Goal
Create reusable training loops.

### Tasks
Implement:

- offline trainer,
- online trainer,
- shared metrics/logging hooks,
- checkpoint saving,
- deterministic seed control,
- output directory management.

### Suggested modules
- `src/train/offline_trainer.py`
- `src/train/online_trainer.py`
- `src/train/checkpoints.py`
- `src/train/logging.py`

### Requirements
- offline and online trainers should share helper utilities where sensible,
- `w` updates should be handled explicitly and at a configurable cadence,
- logging should distinguish:
  - critic loss,
  - actor loss / gradient signal,
  - entropy,
  - terminal wealth stats,
  - evaluation metrics,
  - gradient norms,
  - training time,
  - memory pressure when available,
  - behavior-policy vs execution-policy performance.

### Acceptance criteria
- both trainers can run on toy GBM data,
- checkpoints reload correctly,
- output folders are reproducible and organized.

### Reference files
- both `/references/*.md` summaries
- only query papers for exact update forms if the algorithm module being implemented requires it.

---

## 14. Functionality block: evaluation and backtesting

### Goal
Implement separate evaluation and visualization utilities.

### Tasks
Implement:

- mean return,
- volatility,
- Sharpe ratio,
- Sortino ratio,
- max drawdown,
- turnover,
- optional transaction-cost-adjusted returns,
- seed aggregation,
- deterministic execution evaluation,
- training diagnostics plots,
- wealth-trajectory comparison plots,
- portfolio-weight path plots,
- optional metric overlays on plots.

### Suggested modules
- `src/eval/metrics.py`
- `src/eval/evaluator.py`
- `src/eval/plots.py`
- `src/backtest/portfolio_paths.py`

### Requirements
- training objective and evaluation metrics must remain separate,
- evaluation should support both stochastic behavior policy and deterministic execution policy,
- report settings clearly in saved results,
- plotting must be configurable through YAML,
- plotting should work from saved logs / checkpoints rather than only live training state,
- training plots should cover at least:
  - losses,
  - gradient magnitudes,
  - training time,
  - memory pressure when collected.

### Acceptance criteria
- evaluator works from saved checkpoints,
- metrics are independently unit tested,
- path summaries and aggregate tables are consistent,
- plot generation works from a saved run directory,
- wealth and portfolio plots are reproducible under fixed inputs.

### Reference files
- `/references/README_portfolio_management_nn.md` if present
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

---

## 15. Functionality block: tests

### Goal
Establish fast confidence loops.

### Required test categories

#### Unit tests
Test:
- tensor shapes,
- masking logic,
- environment stepping,
- reward functions,
- log-prob and entropy,
- covariance validity,
- projection / leverage constraints,
- replay buffer behavior.

#### Smoke tests
Test:
- end-to-end tiny offline run,
- end-to-end tiny online run,
- deterministic seeded Black-Scholes run,
- one evaluation pass.

#### Regression tests
Test:
- small known benchmark outputs,
- stability of key metrics under fixed seed,
- no severe runtime/memory regressions on toy tasks.

### Acceptance criteria
- `pytest` passes locally from the project root,
- smoke tests complete quickly,
- regression tests are deterministic.

### Reference files
- `/references/portfolio_mv_papers_companion_implementation_notes.md`

---

## 16. Script-specific TODOs

### 16.1 `scripts/train_offline.py`
Purpose:
- run offline training from historical or precomputed trajectories.

Tasks:
- parse config,
- build dataset or replay source,
- initialize models and trainer,
- run training loop,
- save checkpoints and logs,
- optionally run final evaluation.

Should reference:
- offline trainer,
- config loaders,
- baseline algorithm modules.

### 16.2 `scripts/train_online.py`
Purpose:
- run online or sequential-update training in a simulator or rolling environment.

Tasks:
- parse config,
- initialize environment,
- initialize behavior and execution policies,
- run online updates,
- update `w` at configured cadence,
- log evaluation periodically.

Should reference:
- online trainer,
- environment modules,
- policy modules,
- evaluation hooks.

### 16.3 `scripts/evaluate.py`
Purpose:
- load config and checkpoint,
- run evaluation-only backtests,
- output metric tables and path summaries.

Tasks:
- support deterministic execution mode,
- support stochastic behavior-policy evaluation if desired,
- save outputs cleanly.

Should reference:
- evaluator,
- metrics,
- checkpoint loaders.

### 16.4 `scripts/plot_results.py`
Purpose:
- load config and saved run outputs,
- generate configured diagnostic and portfolio plots.

Tasks:
- support training-diagnostics plots,
- support wealth-trajectory comparison plots,
- support portfolio-weight path plots,
- optionally overlay evaluation metrics or summary tables,
- save outputs cleanly under the run directory.

Should reference:
- evaluator,
- plotting utilities,
- saved logs / checkpoints.

### 16.5 `scripts/run_smoke_test.py`
Purpose:
- very fast integration check.

Tasks:
- load tiny config,
- create tiny environment,
- run a few training updates,
- run one evaluation pass,
- exit with nonzero status on failure.

This script should be used by Claude Code frequently before larger edits are considered complete.

---

## 17. Efficiency guidelines

Claude Code should preserve memory and time efficiency from the beginning.

### Requirements
- keep model sizes small by default,
- avoid storing unnecessary full trajectories when step-wise transitions suffice,
- precompute static features when appropriate,
- keep evaluation batching efficient,
- avoid repeated CPU/GPU transfers,
- do not introduce large dependencies without need.

### Warning
The first version should prefer:
- compact MLPs,
- structured Gaussian actors,
- simple replay buffers,
over large sequence models or attention-heavy architectures.

---

## 18. Documentation expectations

For every major module Claude Code creates, also add:

- a short module docstring,
- clear public API signatures,
- at least one test file if the module is core,
- comments only where they genuinely help.

Do not write long narrative comments in every function. Prefer clear naming.

---

## 19. Definition of done for the first usable version

The first usable version is complete when all of the following hold:

- repository installs and imports cleanly,
- `requirements.txt`, `.gitignore`, and `.venv` workflow are documented,
- tests run under `pytest`,
- synthetic GBM environment works,
- Gaussian behavior policy works,
- deterministic execution policy works,
- offline trainer works,
- online trainer works,
- evaluation metrics work,
- optional context masking works in at least one model path,
- smoke test script passes,
- outputs are saved reproducibly.

---

## 20. Final reminder to Claude Code

Before querying original papers:
1. check `/references/portfolio_mv_papers_algorithm_summary.md`,
2. check `/references/portfolio_mv_papers_companion_implementation_notes.md`,
3. inspect current repo code and tests,
4. only then consult `/references/papers/` if a real ambiguity remains.

The markdown summaries should be treated as the main implementation reference layer.
