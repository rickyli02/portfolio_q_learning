# Setup, Environment, Requirements, Pytest, Git Ignore, and Utilities Guidelines

General note: codex and claude agents are free to communicate over `shared_agent_files/dialogue.txt`.

This file provides the repository setup instructions and implementation standards for foundational project files.

It should be read before creating or modifying:

- `requirements.txt`
- `.gitignore`
- `.venv`
- `pyproject.toml`
- `tests/`
- `src/utils/`

---

## 1. Python version and environment

### Recommended Python version
Use **Python 3.11** unless the repository owner explicitly changes this.

Reason:
- broad package compatibility,
- good PyTorch compatibility,
- fewer build problems than very new Python versions.

### Local environment setup
Create a local virtual environment from the repository root:

```bash
python3.11 -m venv .venv
touch .venv/.noindex
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.11` is not available, use the closest supported version explicitly approved for the repo.
If the created venv exposes only `python3` rather than `python`, use `python3 -m pip ...` and `python3 -m pytest ...`. Do not assume the venv always creates every convenience symlink.

### iCloud / file-sync rule for `.venv`
If the repository lives inside an iCloud-backed path, treat `.venv/` as a local-only working directory, not as synced project content.

Reason:
- cloud file-provider latency can make Python imports pathologically slow,
- large package trees such as `torch` and `numpy` are especially sensitive to filesystem overhead,
- the environment should be reproducible from `requirements.txt`, not preserved via sync.

Required practice:
- create `.venv/` locally from the repo root,
- add `.venv/.noindex` immediately after creation so macOS indexing and iCloud syncing do not treat it like normal document content,
- keep `.venv/` out of git and out of any expected project handoff workflow.

### Environment rules
- `.venv/` must be ignored by git.
- Do not commit environment-specific binaries.
- Do not assume global packages are available.
- All project commands should work from the activated `.venv`.

---

## 2. Requirements.txt guidelines

### Goal
Keep `requirements.txt` small, explicit, and practical for a first research repo.

### Initial core dependencies
A reasonable initial `requirements.txt` should include only what is needed for the first baseline:

```text
numpy
scipy
pandas
torch
pyyaml
pydantic
tqdm
matplotlib
pytest
```

Optional additions if actually used:
```text
hypothesis
ruff
black
jupyter
```

### Dependency rules
- do not add large libraries unless they are clearly needed,
- prefer standard library tools where practical,
- keep optional research dependencies separate if they are not needed for the first smoke runs,
- avoid dependency sprawl early.

### Recommendation
If dev-only dependencies are added later, consider splitting into:
- `requirements.txt`
- `requirements-dev.txt`

But for the first version, a single `requirements.txt` is acceptable if kept small.

---

## 3. .gitignore guidelines

Create a `.gitignore` that at least excludes:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so

# Virtual environment
.venv/
venv/

# Packaging
build/
dist/
*.egg-info/

# Testing / coverage
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store

# Outputs
outputs/
checkpoints/
logs/

# Temporary files
*.tmp
*.swp
```

### Rules
- generated outputs should not be committed by default,
- local editor folders should be ignored,
- keep the ignore file simple and readable.

---

## 4. Pyproject.toml guidelines

Create a minimal `pyproject.toml` to support clean project structure.

### Minimum goals
- define project metadata,
- define Python version requirement,
- support local package imports,
- optionally configure pytest and formatting tools later.

### Suggested initial scope
Keep it lightweight. Do not over-configure the file on day one.

If added, tool config can later include:
- `pytest`
- `ruff`
- `black`

---

## 5. Pytest guidelines

### Goal
Pytest is a required part of the workflow from the start.

### Test folder structure
```text
tests/
├── unit/
├── smoke/
└── regression/
```

### Test priorities

#### Unit tests
Test small isolated components:
- feature masking,
- reward functions,
- environment transitions,
- actor sampling and entropy,
- critic forward pass,
- replay buffer behavior,
- leverage and projection functions.

#### Smoke tests
Run tiny end-to-end workflows:
- one offline run on toy data,
- one online run on toy data,
- one evaluation pass,
- one checkpoint save/load roundtrip.

#### Regression tests
Use only once there is a stable baseline:
- fixed-seed benchmark checks,
- known output ranges,
- runtime sanity checks.

### Pytest rules
- tests should run from the repo root,
- tests should be deterministic where possible,
- smoke tests should stay short,
- do not make all tests depend on large data files,
- synthetic data should be preferred for core test coverage.

### Suggested commands
```bash
pytest
pytest tests/unit
pytest tests/smoke -q
```

---

## 6. Utilities folder guidelines

### Goal
The `src/utils/` folder should contain only genuinely cross-cutting helpers.

### Good utility candidates
- seeding helpers,
- device selection,
- tensor conversion helpers,
- configuration loading helpers,
- small path/output helpers,
- checkpoint path naming,
- logging helpers.

### Poor utility candidates
Do **not** dump unrelated business logic into `utils/`.

The following should **not** go into `utils/`:
- portfolio reward logic,
- environment stepping,
- actor or critic model code,
- algorithm update rules,
- evaluation metrics,
- feature engineering logic specific to one pipeline.

Those belong in their domain-specific modules.

### Suggested utility modules
```text
src/utils/
├── seed.py
├── device.py
├── io.py
├── paths.py
└── logging.py
```

### Utility design rules
- utilities should be small,
- utilities should be pure or nearly pure where possible,
- avoid hidden side effects,
- test utility functions if they are nontrivial.

---

## 7. Output and path management

Create consistent output handling early.

### Recommended behavior
Store outputs under timestamped or config-named directories such as:

```text
outputs/
└── experiment_name/
    └── run_YYYYMMDD_HHMMSS/
```

Save:
- config snapshot,
- logs,
- checkpoints,
- evaluation results.

### Rules
- do not hardcode user-specific absolute paths,
- use `pathlib.Path`,
- create helper functions for output directory creation.

---

## 8. Logging guidelines

### Goal
Make logs useful for debugging and research iteration.

### Minimum logging targets
- training step or episode,
- actor loss / signal,
- critic loss,
- entropy,
- terminal wealth summary,
- evaluation metrics,
- runtime,
- device info,
- seed.

### Rule
Keep logging modular. Logging helpers may live in `src/utils/logging.py`, but algorithm-specific metrics should be produced in the algorithm or trainer modules, not hidden inside a generic utility.

---

## 9. Style and code quality

### Required
- type hints on public APIs,
- docstrings on public modules/classes/functions,
- readable names,
- no unexplained magic constants in core logic,
- minimal but useful comments.

### Strong recommendation
If formatting tools are added later, use:
- `ruff`
- `black`

But do not block initial implementation on tool-chain perfection.

---

## 10. Suggested initial foundational files

Claude Code should create these early:

- `requirements.txt`
- `.gitignore`
- `pyproject.toml`
- `src/__init__.py`
- `src/utils/seed.py`
- `src/utils/device.py`
- `src/utils/io.py`
- `src/utils/paths.py`
- `tests/unit/test_seed.py`
- `tests/smoke/test_imports.py`

These should be created before complex algorithm modules.

---

## 11. Definition of done for setup layer

The setup layer is complete when:

- `.venv` setup works,
- `pip install -r requirements.txt` succeeds,
- imports work from repo root,
- `pytest` runs without path/import issues,
- `.gitignore` excludes expected local/generated files,
- utility modules exist and are not overloaded,
- output path helpers work.

---

## 12. Final reminder

Keep the setup layer boring, reliable, and lightweight.

The repository should be easy to clone, easy to set up, and easy to test before any sophisticated algorithm work begins.
