# Claude Repo Memory Index

- [Portfolio Q-Learning Project Context](project_portfolio_qlearning.md) — current repo state, algorithm targets, key files, coordination rules
- [Portfolio constraint design decisions](project_design_decisions.md) — durable decisions for `src/envs/constraints.py`

This repo-local memory is the preferred source of truth for Claude project memory.
If external Claude memory exists under `~/.claude/projects/.../memory`, treat it as legacy/tool-managed state and sync important updates back here.
