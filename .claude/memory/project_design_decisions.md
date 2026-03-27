---
name: Portfolio constraint design decisions
description: Durable design decisions for constraint handling in src/envs/constraints.py
type: project
---

## Zero-gross projection policy

`apply_risky_only_projection()` must raise `ValueError` when the input action has zero gross exposure (Σ|u_i| = 0).

**Why:**
- there is no direction to rescale a zero vector into a risky-only portfolio
- silently returning the zero vector violates the function contract that projected gross exposure should equal wealth
- the caller must decide whether to skip projection, preserve zero action, or inject a benchmark-specific fallback before calling the function

**Current status:**
- this decision is also recorded in `codex_files/repo_design_memory.md`
- if this issue reappears later, revisit the caller logic first before changing the constraint helper
