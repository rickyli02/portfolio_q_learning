## Phase 2 Manager Notes

Date: 2026-03-26
Owner: Codex

### Current priorities before Phase 2 expands

1. Keep the repository instructions synchronized with the actual tree and workflow.
2. Avoid introducing offline/online divergence before shared config and batch schemas exist.
3. Treat the current tests as scaffold checks, not algorithmic validation.

### Recommended order for the next implementation block

1. Fix small reproducibility/logging/documentation issues that were already identified.
2. Define the config surface area.
3. Define the shared batch/transition schema that both offline and online code will consume.
4. Build the data layer against that schema.
5. Add the first synthetic environment only after the schema and config contracts are stable.

### Phase 2 acceptance criteria

- A typed config system exists with clear names for experiment, environment, reward, optimization, output, and seed settings.
- One base config can be loaded and validated from the repo root.
- Config parsing has unit tests for defaults, overrides, and invalid inputs.
- A shared batch or transition schema is documented in code before trainer logic starts.
- `scripts/run_smoke_test.py` exists before later phases depend on it.

### Known engineering risks to keep visible

- `src/utils/paths.py` currently risks run-directory collisions when two launches happen within the same second.
- `src/utils/logging.py` currently reuses logger handlers in a way that can silently skip file logging.
- `src/utils/seed.py` is still lighter than what GPU RL reproducibility typically needs.
- `src/utils/io.py` may need a broader checkpoint policy once optimizer or RNG state is stored.
- `pyproject.toml` still uses the generic import namespace `src`.

### Config and governance constraints

- Do not edit `.claude/settings.json` or `.codex/config.toml` directly.
- Route requests for user-owned config changes through `shared_agent_files/dialogue.txt`.
- Keep Ricky's dialogue guidance copied into persistent notes under `codex_files/`.
- Archive `shared_agent_files/dialogue.txt` when it becomes too long or when the project clearly moves to a new stage.
