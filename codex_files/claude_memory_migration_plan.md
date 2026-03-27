## Claude Memory Migration Plan

Date: 2026-03-26
Owner: Codex

### What exists today

External Claude project memory currently lives at:

`/Users/ricky/.claude/projects/-Users-ricky-Desktop-Columbia-Academics-Project-portfolio-q-learning-portfolio-q-learning/memory`

It currently contains:

- `MEMORY.md`
- `project_design_decisions.md`
- `project_portfolio_qlearning.md`

### What was done in-repo

A repo-local Claude memory location was created under:

`/.claude/memory/`

with:

- `.claude/memory/MEMORY.md`
- `.claude/memory/project_design_decisions.md`
- `.claude/memory/project_portfolio_qlearning.md`

`.claude/CLAUDE.md` was updated so repo-local Claude memory is now the documented source of truth.

### Recommended migration model

1. Treat `.claude/memory/` as the canonical repo-local memory store.
2. Treat the external `~/.claude/projects/.../memory` folder as legacy or runtime-managed state.
3. Sync important design decisions and project-context updates into `.claude/memory/`.

### If full runtime redirection is desired

That cannot be completed purely from repo files because Claude's external project-memory path is outside the repo and controlled by local Claude tooling.

The cleanest user-level options are:

1. Change Claude's local behavior/settings so it reads/writes the repo-local `.claude/memory/` path directly, if Claude supports that.
2. Replace the external memory directory with a symlink pointing to `.claude/memory/`.

### Constraints

- `.claude/settings.json` and other user-owned Claude settings were not edited.
- No external files were modified or deleted during this pass.
