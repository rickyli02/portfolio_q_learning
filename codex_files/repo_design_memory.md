## Repo Design Memory

Last updated: 2026-03-26
Owner: Codex

### Constraint behavior decisions

- `apply_risky_only_projection()` should raise `ValueError` when the input action has zero gross exposure.
- Rationale:
  - there is no directional information available to rescale a zero vector into a risky-only portfolio,
  - silently returning the zero vector violates the function contract that projected gross exposure should equal wealth,
  - this behavior should be reconsidered only if a later benchmark convention explicitly requires a different fallback.

### Notes for future review

- If this error becomes noisy in practice, revisit whether the caller should:
  - avoid calling risky-only projection on zero actions,
  - inject a benchmark-specific default risky allocation,
  - or use a documented no-op convention instead.
- Until such a benchmark requirement exists, raising is the safer default.
