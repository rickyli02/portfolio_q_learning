---
name: Portfolio Q-Learning Project Context
description: Current project memory for the mean-variance portfolio allocation RL repo
type: project
---

Research repo building a modular, testable codebase for mean-variance portfolio allocation with neural networks and continuous-time-inspired reinforcement learning.

**Current state (as of 2026-03-27):**
- Phase 1 scaffold complete
- Phase 2 config/data foundation complete
- Phase 3A features and masking foundation complete
- Phase 3B synthetic environment foundation and constraint layer approved
- Phase 4A config schema extension for algorithm selection and plotting complete
- Phase 5A oracle benchmark core approved
- Phase 6A actor / critic model interface foundation complete and verification request posted; awaiting Codex approval
  - `src/models/base.py`, `src/models/gaussian_actor.py`, `src/models/quadratic_critic.py` implemented
  - documentation cleanup completed: terminal condition corrected, theorem-aligned vs scaffold split clarified
  - numerical safety planning note added to `shared_agent_files/claude_code_todo.md`
- Latest verified checks:
  - `.venv/bin/pytest tests/unit/test_models.py -q` -> 49 passed
  - `.venv/bin/pytest tests/unit -q` -> 223 passed
  - `.venv/bin/python scripts/run_smoke_test.py` -> 6/6 passed (latest confirmed smoke status before Phase 5A)

**Active implementation target:**
- Huang–Jia–Zhou (2025) theorem-aligned CTRL baseline next (Phase 7)
- Huang–Jia–Zhou (2022) / 2025 practical online improvements after baseline stability
- Wang–Zhou (2019/2020) remains a mathematical and derivational reference, not a required implementation layer

**Model docstring convention (established Phase 6A):**
- Use "THEOREM-ALIGNED STRUCTURE" for the qualitative form from the papers
- Use "REPO SCAFFOLD CHOICES" for deviations: isotropic scalar φ₂ (vs S_{++}^d matrix), free learnable φ₃ (paper treats it as fixed), optimization-safe parameterization
- Per-asset φ₁ ∈ ℝ^d is theorem-aligned (paper §3.6 states this explicitly); its positivity enforcement is a scaffold choice

**Key correctness note — QuadraticCritic terminal condition:**
- At t=T: J(T, x; w) = (x−w)² − (w−z)²   ← CORRECT
- NOT: J(T, x; w) = −(w−z)²   ← WRONG (exponential factor is 1 at t=T, not 0)
- Only the polynomial corrections θ₁(t−T) and θ₂(t²−T²) vanish at t=T

**Numerical safety design direction (Ricky's explicit requirement):**
- Do NOT add silent clamps or silent fallbacks in model forward passes
- On unstable values: log/warn with the offending values and/or raise informative errors
- Goal: make debugging easier, not hide divergence
- Detail plan is in `shared_agent_files/claude_code_todo.md` under "Numerical safety planning"

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
