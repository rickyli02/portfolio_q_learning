# Project Memory: portfolio_q_learning

Last updated: 2026-03-29

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
- Phase 11A stateful CTRL trainer shell foundation: approved
- Phase 11B stateful trainer validation boundary: approved
- Phase 12A trainer snapshot and scalar-summary foundation: approved
- Phase 12B in-memory trainer history foundation: approved
- Phase 12C trainer reset boundary foundation: approved
- Phase 13A minimal CTRL demo script foundation: approved
- Phase 13B in-memory checkpoint payload foundation: approved
- Phase 13C checkpoint file IO foundation: approved
- Phase 14A in-memory trainer logging record foundation: approved
- Phase 14B trainer log file IO foundation: approved
- Phase 15A deterministic evaluation summary foundation: approved
- Phase 15B evaluation summary file IO foundation: approved
- Phase 15C multi-episode deterministic aggregate: approved
- Phase 15D evaluation aggregate file IO: approved
- Phase 15E deterministic evaluation trajectory record: approved
- Phase 15F evaluation record file IO: approved
- Phase 15G multi-seed deterministic record set: approved
- Phase 15H evaluation record-set file IO: approved
- Phase 15I pure derivation helpers (summary_from_record, aggregate_from_record_set): approved
- Phase 15J typed scalar bundle (CTRLEvalScalarBundle, bundle_from_record_set): approved
- Phase 15K scalar-bundle file IO (save/load_eval_bundles): approved
- Phase 16A deterministic CTRL-vs-oracle scalar comparison (src/backtest/comparison.py): approved
- Phase 16B trainer/demo pipeline stress tests across broader GBM parameter regimes: approved
- Phase 16C trainer-to-backtest bridge and tiny comparison demo: pending Codex review

## Current verification snapshot

- `tests/unit -q` has reached `765 passed`
- `scripts/run_smoke_test.py` has reached `8/8 passed`
- long-verification artifacts exist under `outputs/verification/`
- .venv rebuilt on 2026-03-29 (Python 3.14, torch 2.11.0) after torch._functorch.config import failure

## Active implementation direction

- Primary learning path:
  - analytic oracle baseline first
  - Huang-Jia-Zhou (2025) CTRL baseline next
  - practical online improvements only after baseline stability
- Current immediate focus:
  - Phase 16C bridge is complete (src/backtest/train_compare.py, scripts/run_ctrl_oracle_demo.py)
  - first end-to-end training-to-evaluation workflow is now in place
  - next natural step after 16C approval: broader experiment support or reporting

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
