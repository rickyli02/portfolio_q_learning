## Codex Permanent Operating Notes

Last updated: 2026-03-26
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

### Dialogue Management

- When `shared_agent_files/dialogue.txt` becomes too long or the project moves to a new stage, archive the current dialogue into `shared_agent_files/archive/`.
- Archived dialogue files should be named with a datetime label.
- Start a fresh `shared_agent_files/dialogue.txt` with a short summary of the most recent decisions and open issues.
- Dialogue entries should include full timestamps and should be appended at the bottom of the file.
- Generic guidance from Ricky in dialogue should be copied into persistent Codex notes.

### Claude Task Protocol

- Before Claude starts a new bounded implementation block, require a dialogue task-assignment entry that states task, scope, expected files, verification, stop conditions, review target, and `Go/No-Go`.
- Require explicit stop conditions for non-trivial Claude tasks.
- Require Claude to post a verification-request entry after finishing a bounded task block when review is required.
- Require Claude to post a stop-event entry immediately when a stop condition is triggered.

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

### Memory and Notes

- Treat generic guidance from Ricky in the dialogue as persistent operating policy and copy it into `codex_files/` notes.
- Use `codex_files/` for manager notes, coordination policies, and intermediate planning that may need later inspection.

### Current standing engineering concerns

- Keep documentation synchronized with the real repo state.
- Prevent offline and online training paths from diverging before shared schemas are defined.
- Fix small infrastructure issues early when they affect reproducibility or logging behavior.
