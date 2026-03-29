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
- Phase 16C trainer-to-backtest bridge and tiny comparison demo: approved
- Phase 16D smoke-level hardening for both demo entrypoints: approved
- Phase 17A numerical-safety diagnostics foundation: approved
- Phase 17B oracle conditioning-warning boundary: approved
- Phase 17C dtype-sensitivity comparison foundation: approved
- Phase 17D dtype-comparison demo/report seam: approved
- Phase 18A train-compare scalar report foundation: approved

## Current verification snapshot

- `python -m pytest` has reached `856 passed, 1 warning`
- `tests/unit/test_numerics.py -q` has reached `38 passed`
- `tests/unit/test_models.py -q` has reached `49 passed`
- `tests/unit/test_oracle.py -q` has reached `42 passed`
- `tests/unit/test_ctrl_trainer.py -q` has reached `222 passed`
- `tests/unit/test_dtype_compare.py -q` has reached `24 passed`
- `tests/unit/test_dtype_compare_demo.py -q` has reached `11 passed`
- `tests/unit/test_train_compare.py -q` has reached `17 passed`
- `tests/unit/test_train_compare_report.py -q` has reached `16 passed`
- `tests/unit/test_backtest_comparison.py -q` has reached `15 passed`
- `scripts/run_smoke_test.py` has reached `9/9 passed`
- long-verification artifacts exist under `outputs/verification/`
- `.venv` was rebuilt again on 2026-03-29 after the interpreter binaries disappeared from `.venv/bin`; the current confirmed full run is under Python 3.14.3

## Active implementation direction

- Primary learning path:
  - analytic oracle baseline first
  - Huang-Jia-Zhou (2025) CTRL baseline next
  - practical online improvements only after baseline stability
- Current immediate focus:
  - Phase 18B active: align `scripts/run_ctrl_oracle_demo.py` with the approved `CTRLTrainCompareReport` seam
  - documentation now explicitly distinguishes implemented theorem-aligned structure from still-missing Tier 1/Tier 2 paper gaps

## Reference-gap snapshot

- Implemented and tested:
  - theorem-aligned model/environment/loss structure
  - stateful trainer shell
  - evaluation/backtest/train-compare/report seams
  - numerics and dtype diagnostics
- Tier 1 gaps still missing:
  - actor/critic parameter projection sets
  - explicit verification of the `φ₂^{-1}` update path against pseudocode
  - stronger named enforcement of behavior-policy versus execution-policy separation
- Tier 2 gaps still missing:
  - TD(λ) traces
  - one-step incremental updates
  - historical mini-batch weighting
  - separate rebalance versus parameter-update cadences
  - leverage-limit layer
  - named off-policy seam
- Tier 3/Tier 4 remain infrastructure/research roadmap items rather than current implementation state

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
