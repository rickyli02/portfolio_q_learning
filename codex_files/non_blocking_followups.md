# Non-Blocking Follow-Ups

Last updated: 2026-03-27
Owner: Codex

This file tracks review notes that were explicitly judged non-blocking during
bounded implementation phases. These items should not interrupt current work,
but they are easy to forget if they live only in `shared_agent_files/dialogue.txt`.

## Open follow-ups

- Keep `tests/` counts and `tests/unit` counts explicitly separated in future verification notes.
  - Context: one earlier dialogue entry mixed the broader `tests/` count with the narrower `tests/unit` count.
  - Impact: coordination/documentation clarity only.

- Consider strengthening the smoke check for `scripts/run_ctrl_demo.py` to assert specific summary markers rather than only successful completion.
  - Current state: smoke now exercises the real demo entrypoint and treats exit success as sufficient.
  - Impact: slightly stronger regression detection for script output formatting.

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

## Resolved notes

- Duplicate `config-dispatch wiring` bullet in `src/train/ctrl_state.py` docstring.
  - Resolved by user on 2026-03-27.
