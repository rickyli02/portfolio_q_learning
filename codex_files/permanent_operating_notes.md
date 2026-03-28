## Codex Permanent Operating Notes

Last updated: 2026-03-28
Owner: Codex

### Role

- Act as project manager and coordination layer for the repository.
- Help guide Claude on implementation sequencing, scope control, and infrastructure quality.

### Governance

- `.claude/` and `.codex/` are repo-owned collaboration areas.
- `.claude/settings.json` and `.codex/config.toml` are user-owned and must not be edited directly.
- Changes requested for user-owned config files should be routed through `shared_agent_files/dialogue.txt`.
- Claude may request edits to `.claude/CLAUDE.md` or other files under `.claude/`, but those edits should be reviewed by the user and Codex.
- All scripts and tests should be run in the project `.venv`.
- For venv commands, either `python` or `python3` is acceptable; use whichever interpreter path actually exists in the local `.venv`.
- Prefer `.venv/.../python -m ...` style invocation over assuming standalone `pytest` or `pip` entrypoint scripts always exist.
- Prefer checked-in helper scripts for repeated mechanical tasks.

### Dialogue Management

- When `shared_agent_files/dialogue.txt` becomes too long or the project moves to a new stage, archive the current dialogue into `shared_agent_files/archive/`.
- Archived dialogue files should be named with a datetime label.
- Start a fresh `shared_agent_files/dialogue.txt` with a short summary of the most recent decisions and open issues.
- Dialogue entries should include full timestamps and should be appended at the bottom of the file.
- Generic guidance from Ricky in dialogue should be copied into persistent Codex notes.
- Claude timestamp generation should be mechanical via `scripts/get_dialogue_timestamp.sh`.
- Claude long-form reasoning should be saved under `claude_files/` via `scripts/save_claude_reasoning.py`.

### Claude Task Protocol

- Before Claude starts a new bounded implementation block, require a dialogue task-assignment entry that states task, scope, expected files, verification, stop conditions, review target, and `Go/No-Go`.
- Require explicit stop conditions for non-trivial Claude tasks.
- Require Claude to post a verification-request entry after finishing a bounded task block when review is required.
- Require Claude to post a stop-event entry immediately when a stop condition is triggered.
- If a task adds a new dedicated unit-test file, require Claude's verification request to list that file's direct pytest command explicitly, not just the broader `tests/unit` command.
- If the repo has already completed several consecutive infrastructure-only phases in one layer, require the next task-assignment to justify why another extension is needed instead of moving to a consumer workflow.

### Generic Stop Conditions For Claude

- beginning a new phase or materially larger scope than was explicitly assigned
- failing required tests or smoke checks for the active task
- being unable to run required verification in the project `.venv`
- encountering stale docs or governance text that would mislead further work
- hitting a contradiction between dialogue instructions, repo state, and current task scope
- needing to edit a user-owned file
- reaching an unapproved design choice that changes the task materially

### Phase Documentation

- When the project moves into a new phase, create a phase-specific reference note under `references/`.
- Move stale or superseded reference notes into `references/archive/` when they are no longer active.
- Refresh current-state docs when a long run of bounded phases materially changes the repo surface.
- In particular, keep `shared_agent_files/dialogue.txt`, `shared_agent_files/claude_code_todo.md`, and Codex memory notes synchronized with the actual approved repo state rather than letting them lag by multiple phases.

### Memory and Notes

- Treat generic guidance from Ricky in the dialogue as persistent operating policy and copy it into `codex_files/` notes.
- Use `codex_files/` for manager notes, coordination policies, and intermediate planning that may need later inspection.

### Current standing engineering concerns

- Keep documentation synchronized with the real repo state.
- Prevent offline and online training paths from diverging before shared schemas are defined.
- Fix small infrastructure issues early when they affect reproducibility or logging behavior.
- Treat cloud-synced virtual environments as a reproducibility and performance risk; keep `.venv/` local, rebuildable, and out of normal iCloud syncing/indexing paths.
- Avoid overextending infrastructure layers without a consumer.
- After a foundation layer is clearly complete enough to use, bias subsequent tasking toward consuming it in a real workflow before adding more wrappers or persistence helpers around the same layer.
