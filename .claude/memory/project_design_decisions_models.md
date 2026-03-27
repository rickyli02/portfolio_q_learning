---
name: Model layer design decisions
description: Durable design decisions for src/models/ — docstring conventions, numerical safety, parameterization choices
type: project
---

## Numerical safety guard policy (Ricky's direction, 2026-03-27)

Model forward passes must NOT silently clamp or correct unstable values.

**Why:** silent corrections mask training divergence and make bugs harder to find. Ricky explicitly asked for warn/raise behavior instead.

**How to apply:**
- When an intermediate tensor is out-of-bounds (inf, nan, or abs > threshold): emit `warnings.warn` with the tensor name and offending value
- On severe failures (inf/nan output that would poison downstream computation): raise `ValueError` with context
- Implement guards as a utility in `src/utils/`, not inline clamps in forward()
- PSD/PD checks before `linalg.solve` or Cholesky: check `eigvalsh(cov).min() > eps`, warn if conditioning is poor
- Future plan is recorded in `shared_agent_files/claude_code_todo.md` under "Numerical safety planning"

## QuadraticCritic terminal condition

At t=T the correct value is: J(T, x; w) = (x−w)² − (w−z)²

**Why:** e^{−θ₃·(T−T)} = e^0 = 1, so the quadratic term (x−w)² persists. Only the polynomial corrections θ₁(t−T) and θ₂(t²−T²) vanish. Verified against paper terminal condition v(T,x;w) = (x−w)² − (w−z)².

**How to apply:** if the terminal value formula ever resurfaces in a docstring or test, confirm (x−w)² is present at t=T.

## GaussianActor docstring convention

Module headers should distinguish:
- `THEOREM-ALIGNED STRUCTURE` — qualitative form from the paper (Gaussian mean −φ₁(x−w), covariance φ₂ e^{φ₃(T-t)}, φ₁ ∈ ℝ^d)
- `REPO SCAFFOLD CHOICES` — deviations: scalar isotropic φ₂ (paper uses S_{++}^d matrix), free learnable φ₃ (paper treats it as fixed/coupled to θ₃), positivity enforcement on φ₁

**Why:** Codex review flagged the original "THEOREM-BACKED STRUCTURE" header as misleading because it covered scaffold choices without labeling them.

**How to apply:** use the same two-block convention for any future actor/critic modules that mix paper structure with engineering scaffold choices.
