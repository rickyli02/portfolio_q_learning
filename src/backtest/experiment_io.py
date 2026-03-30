"""Run-artifact persistence helpers for config-backed experiments — Phase 20C.

Provides minimal typed helpers to save a ``CTRLTrainCompareReport`` and a
resolved ``ExperimentConfig`` to disk in simple human-readable formats.

Format conventions:
- Scalar report → JSON (flat dict; easy to parse and human-readable).
- Resolved config → YAML (mirrors the input format; round-trips with load_config).

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- experiment directory management or run-ID generation
- checkpoint or tensor-heavy record persistence
- plotting or report visualization
- config-dispatch or experiment runner logic
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import yaml

from src.backtest.train_compare_report import CTRLTrainCompareReport
from src.config.schema import ExperimentConfig


def save_experiment_report(report: CTRLTrainCompareReport, path: Path | str) -> None:
    """Save a ``CTRLTrainCompareReport`` to a JSON file.

    All scalar fields are written as a flat JSON object.  The file is created
    or overwritten at ``path``.

    Args:
        report: Completed ``CTRLTrainCompareReport`` from ``summarize_train_compare``.
        path:   Destination file path (created or overwritten).
    """
    p = Path(path)
    data = dataclasses.asdict(report)
    p.write_text(json.dumps(data, indent=2) + "\n")


def load_experiment_report(path: Path | str) -> CTRLTrainCompareReport:
    """Load a ``CTRLTrainCompareReport`` from a JSON file.

    Args:
        path: Source file path written by ``save_experiment_report``.

    Returns:
        ``CTRLTrainCompareReport`` with all fields populated.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If the JSON is missing a required field or has wrong types.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment report file not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in experiment report: {exc}") from exc
    try:
        return CTRLTrainCompareReport(**data)
    except TypeError as exc:
        raise ValueError(f"Experiment report has unexpected fields: {exc}") from exc


def save_experiment_config(cfg: ExperimentConfig, path: Path | str) -> None:
    """Save a resolved ``ExperimentConfig`` to a YAML file.

    Serialises all config fields using their dataclass values so the saved file
    reflects the effective config used for the run, not the raw input YAML.
    The file is created or overwritten at ``path``.

    Args:
        cfg:  Resolved ``ExperimentConfig`` (post-validation).
        path: Destination file path (created or overwritten).
    """
    p = Path(path)

    def _to_dict(obj: object) -> object:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, list):
            return [_to_dict(v) for v in obj]
        return obj

    p.write_text(yaml.dump(_to_dict(cfg), default_flow_style=False, sort_keys=False))
