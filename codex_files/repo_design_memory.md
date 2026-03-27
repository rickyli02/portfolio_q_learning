## Repo Design Memory

Last updated: 2026-03-27
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

### Oracle / linear algebra decisions

- Prefer `torch` over `numpy` for repo-integrated algorithms and analytic benchmarks.
- Rationale:
  - environments, future models, and evaluation code already exchange `torch.Tensor` objects,
  - staying in torch avoids conversion churn and dtype/device mismatches,
  - the oracle benchmark should remain directly composable with future trainer and rollout code.
- For closed-form portfolio calculations, use `torch.float64` for coefficient precomputation, then convert to the caller's wealth/action dtype at evaluation time.
- Avoid explicit matrix inverse operations on covariance systems in the primary computation path when a linear solve is sufficient.
- Rationale:
  - solve-based paths are numerically safer than forming `Sigma^{-1}` explicitly,
  - ill-conditioned covariance matrices are plausible once correlated synthetic assets and parameter sweeps are used,
  - the oracle benchmark is a reference baseline, so silent numerical instability is more harmful than a small amount of extra implementation care.
