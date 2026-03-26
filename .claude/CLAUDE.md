# CLAUDE.md — Portfolio Q-Learning

## Purpose

This file is for stable baseline behavior only. Current implementation status, active phase progress, and task-specific instructions belong in `shared_agent_files/claude_code_todo.md` and `shared_agent_files/dialogue.txt`.

## Primary references

Use sources in this order:

1. existing code in this repository
2. `shared_agent_files/dialogue.txt`
3. `shared_agent_files/claude_code_todo.md`
4. `shared_agent_files/setup_and_utilities_guidelines.md`
5. `references/portfolio_mv_papers_algorithm_summary.md`
6. `references/portfolio_mv_papers_companion_implementation_notes.md`
7. `references/papers/` only when the markdown references are insufficient

Do not repeatedly re-read original papers when the markdown references already contain the needed detail.

## Governance

- `.claude/settings.json` and `.codex/config.toml` are user-owned. Do not edit them directly.
- Request changes to user-owned config files through `shared_agent_files/dialogue.txt`.
- `.claude/CLAUDE.md` and `.claude/commands/` are repo-owned collaboration files.
- Claude may write long reasoning and planning notes under `claude_files/`.
- Codex may edit reasoning files in `claude_files/`.
- All scripts and tests must be run in the project `.venv`.

## Helper scripts

Use the Claude-specific helper scripts under `claude_files/scripts/`:

- `sh claude_files/scripts/get_dialogue_timestamp.sh`
  Purpose: get a dialogue timestamp from the shell.
- `python claude_files/scripts/save_claude_reasoning.py`
  Purpose: save a long reasoning note under `claude_files/`.

Prefer checked-in helper scripts over repeated manual formatting.

## Dialogue rules

Every nontrivial Claude task must be governed through `shared_agent_files/dialogue.txt`.

Rules:
- append entries at the bottom of the file
- include a full timestamp
- use a shell-derived timestamp from `sh claude_files/scripts/get_dialogue_timestamp.sh`
- do not invent or hand-type timestamps

Task assignment format:

```text
[Codex | YYYY-MM-DD HH:MM:SS TZ]
To Claude:
- Task: <short task name>
- Scope: <allowed scope>
- Expected files: <files or folders>
- Verification: <commands to run in .venv>
- Stop conditions: <task-specific stop rules>
- Review required: <Codex, user, or none>
- Go/No-Go: GO
```

Verification request format:

```text
[Claude Code | YYYY-MM-DD HH:MM:SS TZ]
Verification request:
- Task completed: <task name>
- Files changed: <files>
- Verification run in .venv: <commands and results>
- Known risks / open questions: <items>
- Requesting review from: <Codex or user>
```

Stop event format:

```text
[Claude Code | YYYY-MM-DD HH:MM:SS TZ]
Stop event:
- Active task: <task name>
- Triggered condition: <condition>
- Reason: <why work stopped>
- Files touched so far: <files or none>
- Requested next input: <what is needed>
```

## Generic stop rules

Claude must stop and post a `Stop event` when any of the following occurs:

- the next step would begin a new phase or materially larger scope than assigned
- required verification in `.venv` fails
- required verification in `.venv` cannot be run
- documentation or governance text is stale enough to mislead further work
- a contradiction appears between repo state, dialogue instructions, and task scope
- the task would require editing a user-owned file
- the next step depends on an unapproved design choice
- the bounded task block is complete and review was required before continuing
- a trustworthy timestamp cannot be obtained from the helper script

## Implementation principles

- keep modules small and single-purpose
- keep offline and online paths aligned through shared interfaces
- use explicit masks for optional conditional features
- keep training objective separate from evaluation metrics
- favor correctness, modularity, testability, and fast verification over novelty
- use type hints, concise docstrings, and `pathlib.Path`
- avoid hidden global state and notebook-only logic

## Command guidance

- Prefer simple single-purpose commands over long chained shell commands.
- Avoid compound commands like `ls ... && echo ... && ls ...` when separate commands or a helper script will do.
- Run project verification from `.venv`, for example:

```bash
.venv/bin/pytest
.venv/bin/python scripts/run_smoke_test.py
```

## Coordination

- `shared_agent_files/dialogue.txt` is the coordination log between Claude, Codex, and the user.
- `shared_agent_files/claude_code_todo.md` tracks current implementation status and active roadmap details.
- Claude writes under `claude_files/`; Codex writes under `codex_files/`.
