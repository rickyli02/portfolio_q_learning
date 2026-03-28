"""Evaluation record file IO — Phase 15F.

Provides minimal helpers to persist and reload ``CTRLEvalRecord`` entries
using a newline-delimited JSON (JSONL) format.  Each line is one record; the
format is plain-text and human-readable without additional dependencies.
Tensor fields are stored as plain numeric lists and restored as ``torch.Tensor``
objects on load.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- run-directory or artifact-management policy
- config-dispatch wiring
- plotting or report generation
- aggregate / summary file IO changes
- trainer integration
- stochastic evaluation
- compression or binary artifact conventions

These belong in future bounded tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval.record import CTRLEvalRecord

# Float dtype used for all restored tensor fields (matches repo convention).
_EVAL_DTYPE = torch.float32


def _expect_float(value: object, name: str, lineno: int) -> None:
    """Raise ValueError if ``value`` is not a JSON number (int or float, not bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Field '{name}' on line {lineno} must be a number, "
            f"got {type(value).__name__!r}"
        )


def _expect_int(value: object, name: str, lineno: int) -> None:
    """Raise ValueError if ``value`` is not a JSON integer (not bool, not float)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Field '{name}' on line {lineno} must be an integer, "
            f"got {type(value).__name__!r}"
        )


def _expect_list_of_numbers(value: object, name: str, lineno: int) -> None:
    """Raise ValueError if ``value`` is not a list of JSON numbers."""
    if not isinstance(value, list):
        raise ValueError(
            f"Field '{name}' on line {lineno} must be a list, "
            f"got {type(value).__name__!r}"
        )
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"Field '{name}[{i}]' on line {lineno} must be a number, "
                f"got {type(item).__name__!r}"
            )


def _expect_list_of_list_of_numbers(value: object, name: str, lineno: int) -> None:
    """Raise ValueError if ``value`` is not a list of lists of JSON numbers."""
    if not isinstance(value, list):
        raise ValueError(
            f"Field '{name}' on line {lineno} must be a list, "
            f"got {type(value).__name__!r}"
        )
    for i, row in enumerate(value):
        if not isinstance(row, list):
            raise ValueError(
                f"Field '{name}[{i}]' on line {lineno} must be a list, "
                f"got {type(row).__name__!r}"
            )
        for j, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"Field '{name}[{i}][{j}]' on line {lineno} must be a number, "
                    f"got {type(item).__name__!r}"
                )


def save_eval_records(records: list[CTRLEvalRecord], path: Path | str) -> None:
    """Save a list of ``CTRLEvalRecord`` entries to a JSONL file.

    Each record is serialised as one JSON object per line.  Tensor fields are
    stored as nested lists of plain Python floats.  The file is created or
    overwritten at ``path``.

    Args:
        records: List of ``CTRLEvalRecord`` entries to save.
        path:    Destination file path (created or overwritten).
    """
    p = Path(path)
    with p.open("w") as fh:
        for rec in records:
            fh.write(
                json.dumps(
                    {
                        "times": rec.times.tolist(),
                        "wealth_path": rec.wealth_path.tolist(),
                        "actions": rec.actions.tolist(),
                        "terminal_wealth": rec.terminal_wealth,
                        "initial_wealth": rec.initial_wealth,
                        "n_steps": rec.n_steps,
                        "min_wealth": rec.min_wealth,
                        "max_wealth": rec.max_wealth,
                        "target_return_z": rec.target_return_z,
                        "terminal_gap": rec.terminal_gap,
                    }
                )
                + "\n"
            )


def load_eval_records(path: Path | str) -> list[CTRLEvalRecord]:
    """Load ``CTRLEvalRecord`` entries from a JSONL file.

    Tensor fields are restored as ``torch.Tensor`` objects with dtype
    ``torch.float32``.  Consistency checks are applied to path lengths.

    Args:
        path: Source file path.

    Returns:
        List of ``CTRLEvalRecord`` in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If any line contains malformed JSON, is missing a
                           required field, has an incorrect scalar or list type,
                           or has inconsistent path lengths.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval record file not found: {p}")

    results: list[CTRLEvalRecord] = []
    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON on line {lineno}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object on line {lineno}, "
                f"got {type(data).__name__!r}"
            )

        # --- required fields ---
        required = [
            "times", "wealth_path", "actions",
            "terminal_wealth", "initial_wealth", "n_steps",
            "min_wealth", "max_wealth",
        ]
        for field in required:
            if field not in data:
                raise ValueError(
                    f"Missing required field '{field}' on line {lineno}"
                )

        n_steps_raw = data["n_steps"]
        _expect_int(n_steps_raw, "n_steps", lineno)
        n_steps: int = n_steps_raw

        _expect_float(data["terminal_wealth"], "terminal_wealth", lineno)
        _expect_float(data["initial_wealth"], "initial_wealth", lineno)
        _expect_float(data["min_wealth"], "min_wealth", lineno)
        _expect_float(data["max_wealth"], "max_wealth", lineno)

        times_raw = data["times"]
        wealth_path_raw = data["wealth_path"]
        actions_raw = data["actions"]

        _expect_list_of_numbers(times_raw, "times", lineno)
        _expect_list_of_numbers(wealth_path_raw, "wealth_path", lineno)
        _expect_list_of_list_of_numbers(actions_raw, "actions", lineno)

        # --- path-length consistency ---
        if len(times_raw) != n_steps:
            raise ValueError(
                f"Line {lineno}: len(times)={len(times_raw)} != n_steps={n_steps}"
            )
        if len(wealth_path_raw) != n_steps + 1:
            raise ValueError(
                f"Line {lineno}: len(wealth_path)={len(wealth_path_raw)} "
                f"!= n_steps+1={n_steps + 1}"
            )
        if len(actions_raw) != n_steps:
            raise ValueError(
                f"Line {lineno}: len(actions)={len(actions_raw)} != n_steps={n_steps}"
            )
        if n_steps > 0:
            inner_width = len(actions_raw[0])
            for i, row in enumerate(actions_raw):
                if len(row) != inner_width:
                    raise ValueError(
                        f"Line {lineno}: actions[{i}] has width {len(row)}, "
                        f"expected {inner_width} (inconsistent action rows)"
                    )

        # --- optional nullable fields ---
        target_return_z = data.get("target_return_z")
        terminal_gap = data.get("terminal_gap")

        if target_return_z is not None:
            _expect_float(target_return_z, "target_return_z", lineno)
        if terminal_gap is not None:
            _expect_float(terminal_gap, "terminal_gap", lineno)

        results.append(
            CTRLEvalRecord(
                times=torch.tensor(times_raw, dtype=_EVAL_DTYPE),
                wealth_path=torch.tensor(wealth_path_raw, dtype=_EVAL_DTYPE),
                actions=torch.tensor(actions_raw, dtype=_EVAL_DTYPE),
                terminal_wealth=float(data["terminal_wealth"]),
                initial_wealth=float(data["initial_wealth"]),
                n_steps=n_steps,
                min_wealth=float(data["min_wealth"]),
                max_wealth=float(data["max_wealth"]),
                target_return_z=float(target_return_z) if target_return_z is not None else None,
                terminal_gap=float(terminal_gap) if terminal_gap is not None else None,
            )
        )

    return results
