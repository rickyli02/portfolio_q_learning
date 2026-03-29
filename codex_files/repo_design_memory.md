## Repo Design Memory

Last updated: 2026-03-28
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

### Environment / verification decisions

- The severe multi-minute `import torch` slowdown observed on 2026-03-27 was environmental, not treated as a core repo or pure PyTorch-algorithm bug.
- Evidence:
  - while the repo lived with an iCloud-backed `.venv`, import-time profiling showed pathological delays
  - after recreating `.venv` locally and marking it with `.noindex`, import timing normalized to roughly:
    - `numpy ~= 0.069s`
    - `torch ~= 0.859s`
- Project rule:
  - treat `.venv/` as local-only rebuildable state
  - do not rely on cloud sync for virtual-environment contents
  - add `.venv/.noindex` when the repo is under an iCloud-backed path

### Venv interpreter-path decisions

- Do not assume every machine will expose `.venv/bin/python` and `.venv/bin/pip` convenience entrypoints.
- Ricky approved either `python` or `python3` command usage.
- Preferred practice:
  - use whichever interpreter path actually exists in the venv
  - prefer interpreter-invoked module commands such as:
    - `.venv/bin/python -m pytest`
    - `.venv/bin/python3 -m pytest`
    - `.venv/bin/python -m pip`
    - `.venv/bin/python3 -m pip`
- Do not require a symlink-normalization step just to standardize command names.

### Evaluation-layer sequencing decision

- The Phase 15 eval stack is now deep enough to consume directly:
  - summaries
  - aggregates
  - path records
  - multi-seed record sets
  - scalar derivation helpers
  - scalar bundles
  - file IO for all of the above
- Strategic conclusion from the 2026-03-28 review:
  - additional eval-layer wrappers or persistence helpers should no longer be the default next step
  - the next high-value work should consume this stack in a real workflow
- Preferred next-consumer directions:
  - populate `src/backtest/` with a narrow CTRL-vs-oracle comparison seam
  - add an end-to-end experiment/evaluation script under `scripts/`
  - connect trainer output to evaluation/oracle comparison in a bounded post-run flow
- Rationale:
  - Phases 15E-15K were technically sound but had no in-repo consumer yet
  - the value of saved records/record-sets/bundles is only realized once something loads them and produces comparison output or downstream analysis

### Post-Phase-16 consumer sequencing

- Phase 16A completed the first real eval-stack consumer under `src/backtest/`:
  - deterministic CTRL-vs-oracle scalar comparison
  - approved tests now cover the comparison seam and exact oracle/CTRL scalar-bundle behavior
- The next quality bottleneck initially shifted from "missing consumer" to "testing the correct trainer-facing surface".
- Current sequencing decision:
  - Phase 16B resolved that trainer-facing stress-coverage gap around the approved stateful shell
  - the next bounded consumer should connect `CTRLTrainerState` output to deterministic CTRL-vs-oracle comparison in one small post-run flow
- Review lesson from the Phase 16B follow-up:
  - direct tests of `ctrl_outer_loop(...)` are useful but do not substitute for stateful-shell coverage when the task explicitly names `CTRLTrainerState.run_outer_loop(...)`
  - snapshot/history behavior should be treated as part of the trainer pipeline contract once the stateful shell is the requested seam
