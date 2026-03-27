# Project Memory: portfolio_q_learning

Last updated: 2026-03-27

## Current implementation state

- Phase 1 scaffold: complete
- Phase 2 config/data foundation: complete
- Phase 3A features and masking foundation: complete
- Phase 3B synthetic environment and constraints: approved
- Phase 4A config schema extension for algorithm selection and plotting: approved
- Phase 5A oracle benchmark core: approved
- Phase 6A actor / critic model interface foundation: approved
- Phase 7A CTRL rollout and deterministic-evaluation scaffolding: complete
- Phase 7B CTRL smoke integration and long-verification tooling: approved
- Phase 8A CTRL objective and loss-primitive foundation: approved
- Phase 8B CTRL gradient-tracked re-evaluation foundation: approved
- Phase 8C CTRL scalar loss assembly foundation: approved
- Phase 9A CTRL single-trajectory trainer step foundation: approved
- Phase 9B CTRL fixed-length trainer-run foundation: approved
- Phase 10A CTRL outer-loop w update primitive: approved
- Phase 10B CTRL single outer-iteration foundation: approved
- Phase 10C CTRL fixed-length outer-loop schedule foundation: approved
- Active bounded task: Phase 11A stateful CTRL trainer shell foundation

## Current verification snapshot

- `tests/unit -q` has reached `402 passed`
- `tests/unit/test_ctrl_trainer.py -q` has reached `83 passed`
- `scripts/run_smoke_test.py` has reached `7/7 passed`
- long-verification artifacts exist under `outputs/verification/`
- subprocess-isolated import timing normalized after recreating `.venv` locally:
  - `numpy ~= 0.069s`
  - `torch ~= 0.859s`

## Active implementation direction

- Primary learning path:
  - analytic oracle baseline first
  - Huang-Jia-Zhou (2025) CTRL baseline next
  - practical online improvements only after baseline stability
- Current immediate focus:
  - add a thin stateful trainer-facing shell over the approved functional helpers
  - keep the trainer layer bounded and avoid opening full trainer/checkpoint/logging infrastructure yet

## Stable decisions

- `env.mu` is the price-SDE drift, not expected log-return
- prefer `torch` over `numpy` for repo-integrated algorithm code
- prefer linear solve / Cholesky-style paths over explicit inverse when equivalent
- `QuadraticCritic` terminal condition must remain:
  - `J(T, x; w) = (x-w)^2 - (w-z)^2`
- `GaussianActor` docs must distinguish theorem-aligned qualitative structure from repo scaffold choices
- `apply_risky_only_projection()` should raise `ValueError` on zero gross exposure

## Environment / workflow notes

- Treat `.venv/` as local-only rebuildable state
- If the repo lives in an iCloud-backed location, add `.venv/.noindex`
- Either `.venv/bin/python` or `.venv/bin/python3` is acceptable; use whichever interpreter path actually exists locally
- Prefer interpreter-invoked module commands such as:
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python3 -m pytest`

## Coordination

- Stable baseline rules live in `.claude/CLAUDE.md`
- Current scope and implementation state live in:
  - `shared_agent_files/dialogue.txt`
  - `shared_agent_files/claude_code_todo.md`
