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

### Model-layer documentation decisions

- For the current Phase 6A model foundation, the reference markdown files are closer to the original papers than the first-pass model docstrings were.
- `QuadraticCritic` documentation must preserve the paper-aligned terminal condition:
  - `J(T, x; w) = (x-w)^2 - (w-z)^2`
- `GaussianActor` documentation should distinguish:
  - theorem-aligned qualitative structure:
    - mean proportional to `-phi1(x-w)`,
    - covariance scaled by `phi2 * exp(phi3(T-t))`,
    - stochastic behavior policy versus deterministic execution policy
  - repo scaffold choices:
    - isotropic scalar `phi2` instead of full matrix-valued `S_{++}^d`,
    - free learnable `phi3`,
    - optimization-safe parameterization details such as positivity handling
- In the multi-asset 2025 reference path, vector-valued `phi1 ∈ R^d` should not be described as purely a repo invention.

### Numerical-safety planning

- Add explicit planning for operations that can fail or diverge before deeper CTRL implementation starts.
- At minimum, future bounded tasks should consider:
  - positivity guards before `log`, entropy, or variance-derived operations,
  - PSD/PD and conditioning checks for covariance-like objects,
  - preferring solve / Cholesky-style paths over explicit inverse where mathematically equivalent,
  - deciding case-by-case whether failures should:
    - raise immediately,
    - clamp with a documented epsilon,
    - or warn and continue
- These choices should be documented as local design decisions when they first appear in code, rather than left implicit in implementation details.
