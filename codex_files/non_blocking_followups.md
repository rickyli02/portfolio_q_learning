# Non-Blocking Follow-Ups

Last updated: 2026-03-29
Owner: Codex

This file tracks review notes that were explicitly judged non-blocking during
bounded implementation phases. These items should not interrupt current work,
but they are easy to forget if they live only in `shared_agent_files/dialogue.txt`.

## Open follow-ups

- Keep `tests/` counts and `tests/unit` counts explicitly separated in future verification notes.
  - Context: one earlier dialogue entry mixed the broader `tests/` count with the narrower `tests/unit` count.
  - Impact: coordination/documentation clarity only.

- Consider keeping smoke-facing demo marker checks narrow and stable once Phase 16D lands.
  - Current state: an active bounded task now targets marker-based smoke coverage for both `scripts/run_ctrl_demo.py` and `scripts/run_ctrl_oracle_demo.py`.
  - Impact: avoid brittle formatting assertions while still catching real demo-entrypoint regressions.

- Add stronger optimizer roundtrip coverage once trainer work uses nontrivial optimizer state.
  - Current state: checkpoint roundtrip tests mostly exercise scalar/model restore paths with simple SGD.
  - Impact: future confidence for momentum / Adam-style state restoration.

- Normalize trainer/test module naming and headers before the trainer API hardens further.
  - Current state: this has been noted in roadmap/docs review more than once.
  - Impact: mostly maintenance and readability, but worth cleaning before broader trainer surface grows.

- Expand the CTRL pseudocode note with a more explicit trace-formula pointer from the companion notes.
  - Current state: already listed in `shared_agent_files/claude_code_todo.md` as a pending follow-up.
  - Impact: documentation precision for later theorem-to-code comparisons.

- Add concrete memory-pressure capture guidance when logging/plotting infrastructure is implemented.
  - Current state: already listed in roadmap docs, but not yet turned into a bounded task.
  - Impact: useful for scaling future trainer/evaluation runs.

- Clarify transaction-cost realism early once the repo begins real-data experiment design.
  - Current state: brainstorming and documentation review both concluded that transaction costs affect the optimization problem itself, not just post hoc evaluation.
  - Impact: future WRDS / experiment-design work should not treat costs as a late cosmetic add-on.

- Clarify the first derivatives extension as "options/implied-volatility information as conditioning inputs" before considering options as tradable assets.
  - Current state: brainstorm review narrowed this as the most tractable first derivatives path.
  - Impact: keeps future design work from expanding the action space prematurely.

- Keep the asymmetric-information synthetic toy idea, but rename it away from vague "insider trading" wording.
  - Current state: brainstorm review refined this into a toy setup where one policy observes a latent jump/regime signal before resolution.
  - Impact: preserves the research idea while keeping the design mathematically and ethically cleaner.

- Decide later whether `CTRLEvalRecord` should become immutable or defensively clone tensor payloads at construction/load boundaries.
  - Current state: Phase 15E intentionally kept `CTRLEvalRecord` mutable and returns wrapped tensors directly; review approved this as non-blocking.
  - Impact: later code should not assume record payloads are immutable until this boundary is revisited.

- Decide later whether `CTRLEvalRecordSet` / `CTRLEvalScalarBundle` should also be hardened against mutation or documented more explicitly as mutable transport containers.
  - Current state: review accepted the current minimal container behavior as non-blocking.
  - Impact: future consumer code should not assume immutable container payloads unless the contract is tightened deliberately.

## Resolved notes

- Duplicate `config-dispatch wiring` bullet in `src/train/ctrl_state.py` docstring.
  - Resolved by user on 2026-03-27.
