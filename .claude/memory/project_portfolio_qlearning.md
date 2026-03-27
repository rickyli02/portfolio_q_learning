---
name: Portfolio Q-Learning Project Context
description: Current project memory for the mean-variance portfolio allocation RL repo
type: project
---

Research repo building a modular, testable codebase for mean-variance portfolio allocation with neural networks and continuous-time-inspired reinforcement learning.

**Current state (as of 2026-03-26):**
- Phase 1 scaffold complete
- Phase 2 config/data foundation complete
- Phase 3A features and masking foundation complete
- Phase 3B synthetic environment foundation and constraint fixes submitted for final Codex review

**Active implementation target:**
- oracle benchmark from known synthetic parameters first
- Huang–Jia–Zhou (2025) theorem-aligned CTRL baseline next
- Huang–Jia–Zhou (2022) / 2025 practical online improvements after baseline stability
- Wang–Zhou (2019/2020) remains a mathematical and derivational reference, not a required implementation layer

**Key repo files:**
- `shared_agent_files/dialogue.txt` — coordination log between Claude, Codex, and the user
- `shared_agent_files/claude_code_todo.md` — active implementation roadmap and task inventory
- `references/portfolio_mv_papers_algorithm_summary.md` — paper summary and repo-layer recommendations
- `references/portfolio_mv_papers_companion_implementation_notes.md` — implementation notes and practical caveats
- `references/portfolio_mv_ctrl_complete_pseudocode.md` — repo-aligned oracle + CTRL pseudocode
- `.claude/memory/MEMORY.md` — repo-local Claude memory index

**Coordination rules:**
- Use `.claude/memory/` as the durable Claude memory location inside the repo
- Treat external `~/.claude/projects/.../memory` as legacy/tool-managed state if present
- Follow `shared_agent_files/dialogue.txt` for bounded tasks, stop conditions, and review gates
