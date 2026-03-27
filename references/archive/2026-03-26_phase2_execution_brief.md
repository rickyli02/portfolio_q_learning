# Phase 2 Execution Brief

Date: 2026-03-26
Scope: config system, shared schemas, data layer preparation

## Objective

Phase 2 should establish the contracts that all later RL components will depend on. The main goal is to define the configuration and data interfaces before environment, trainer, and algorithm code grows around unstable assumptions.

## Recommended sequence

1. Clean up the known low-cost infrastructure issues in `src/utils/` and packaging that can affect reproducibility or workflow.
2. Define typed experiment/config objects.
3. Define the shared transition or batch schema for offline and online learning.
4. Build the first data-layer modules against that schema.
5. Add the first synthetic environment only after config and data contracts are stable.

## Required config surface

The initial config system should cover:

- seed and reproducibility settings
- environment settings
- asset-universe settings
- reward and mean-variance parameters
- optimization and learning-rate settings
- online-update or rebalance cadence
- output/logging settings
- evaluation settings

Use explicit names such as `entropy_temp`, `trace_decay`, `target_return`, `rebalance_interval`, and `online_update_interval`.

## Shared schema expectations

Before trainer code is expanded, define a shared schema for transitions or batches that can support both:

- offline replay batches
- online sequential updates
- optional conditioning features with explicit masks
- portfolio action or weight outputs
- reward and next-state fields

## Acceptance criteria

- One validated base config can be loaded from repo root.
- Config parsing has tests for defaults, overrides, and invalid values.
- The shared transition or batch schema is documented in code and tests.
- The first data abstractions consume the shared schema rather than ad hoc dictionaries.
- `scripts/run_smoke_test.py` exists before later phases rely on it.

## Known risks to watch

- A generic top-level import package name like `src` is acceptable short-term but weak long-term.
- Current utility tests confirm scaffold sanity only; they do not validate portfolio logic.
- Reproducibility and logging issues should be fixed before longer experiment runs become common.
