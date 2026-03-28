"""Evaluation record-set file IO — Phase 15H.

Provides minimal helpers to persist and reload ``CTRLEvalRecordSet`` entries
using a newline-delimited JSON (JSONL) format.  Each line is one record set
serialised as a JSON object; embedded ``CTRLEvalRecord`` tensor fields are
stored as plain numeric lists and restored as ``torch.Tensor`` objects on load.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- run-directory or artifact-management policy
- config-dispatch wiring
- plotting or report generation
- aggregate / summary / single-record file IO changes
- compression or binary artifact conventions
- trainer integration
- stochastic evaluation

These belong in future bounded tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval.record import CTRLEvalRecord
from src.eval.record_set import CTRLEvalRecordSet

_EVAL_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Scalar and list validators
# ---------------------------------------------------------------------------


def _expect_float(value: object, name: str, lineno: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Field '{name}' on line {lineno} must be a number, "
            f"got {type(value).__name__!r}"
        )


def _expect_int(value: object, name: str, lineno: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Field '{name}' on line {lineno} must be an integer, "
            f"got {type(value).__name__!r}"
        )


def _expect_list_of_ints(value: object, name: str, lineno: int) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"Field '{name}' on line {lineno} must be a list, "
            f"got {type(value).__name__!r}"
        )
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"Field '{name}[{i}]' on line {lineno} must be an integer, "
                f"got {type(item).__name__!r}"
            )


def _expect_list_of_numbers(value: object, name: str, lineno: int) -> None:
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


# ---------------------------------------------------------------------------
# Embedded-record helpers
# ---------------------------------------------------------------------------


def _record_to_dict(rec: CTRLEvalRecord) -> dict:
    return {
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


def _dict_to_record(data: object, rec_idx: int, lineno: int) -> CTRLEvalRecord:
    """Parse one embedded record dict into a ``CTRLEvalRecord``.

    Args:
        data:    Parsed JSON value for this record entry.
        rec_idx: Zero-based record index within the set (for error messages).
        lineno:  JSONL line number (for error messages).
    """
    ctx = f"records[{rec_idx}]"
    if not isinstance(data, dict):
        raise ValueError(
            f"Line {lineno}: {ctx} must be a JSON object, "
            f"got {type(data).__name__!r}"
        )

    required = [
        "times", "wealth_path", "actions",
        "terminal_wealth", "initial_wealth", "n_steps",
        "min_wealth", "max_wealth",
    ]
    for field in required:
        if field not in data:
            raise ValueError(
                f"Line {lineno}: missing required field '{field}' in {ctx}"
            )

    n_steps_raw = data["n_steps"]
    _expect_int(n_steps_raw, f"{ctx}.n_steps", lineno)
    n_steps: int = n_steps_raw

    _expect_float(data["terminal_wealth"], f"{ctx}.terminal_wealth", lineno)
    _expect_float(data["initial_wealth"], f"{ctx}.initial_wealth", lineno)
    _expect_float(data["min_wealth"], f"{ctx}.min_wealth", lineno)
    _expect_float(data["max_wealth"], f"{ctx}.max_wealth", lineno)

    times_raw = data["times"]
    wealth_path_raw = data["wealth_path"]
    actions_raw = data["actions"]

    _expect_list_of_numbers(times_raw, f"{ctx}.times", lineno)
    _expect_list_of_numbers(wealth_path_raw, f"{ctx}.wealth_path", lineno)
    _expect_list_of_list_of_numbers(actions_raw, f"{ctx}.actions", lineno)

    if len(times_raw) != n_steps:
        raise ValueError(
            f"Line {lineno}: {ctx} len(times)={len(times_raw)} != n_steps={n_steps}"
        )
    if len(wealth_path_raw) != n_steps + 1:
        raise ValueError(
            f"Line {lineno}: {ctx} len(wealth_path)={len(wealth_path_raw)} "
            f"!= n_steps+1={n_steps + 1}"
        )
    if len(actions_raw) != n_steps:
        raise ValueError(
            f"Line {lineno}: {ctx} len(actions)={len(actions_raw)} != n_steps={n_steps}"
        )
    if n_steps > 0:
        inner_width = len(actions_raw[0])
        for i, row in enumerate(actions_raw):
            if len(row) != inner_width:
                raise ValueError(
                    f"Line {lineno}: {ctx} actions[{i}] has width {len(row)}, "
                    f"expected {inner_width} (inconsistent action rows)"
                )

    target_return_z = data.get("target_return_z")
    terminal_gap = data.get("terminal_gap")
    if target_return_z is not None:
        _expect_float(target_return_z, f"{ctx}.target_return_z", lineno)
    if terminal_gap is not None:
        _expect_float(terminal_gap, f"{ctx}.terminal_gap", lineno)

    return CTRLEvalRecord(
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


# ---------------------------------------------------------------------------
# Public IO helpers
# ---------------------------------------------------------------------------


def save_eval_record_sets(
    record_sets: list[CTRLEvalRecordSet], path: Path | str
) -> None:
    """Save a list of ``CTRLEvalRecordSet`` entries to a JSONL file.

    Each record set is serialised as one JSON object per line.  The file is
    created or overwritten at ``path``.

    Args:
        record_sets: List of ``CTRLEvalRecordSet`` entries to save.
        path:        Destination file path (created or overwritten).
    """
    p = Path(path)
    with p.open("w") as fh:
        for rs in record_sets:
            fh.write(
                json.dumps(
                    {
                        "seeds": rs.seeds,
                        "records": [_record_to_dict(r) for r in rs.records],
                    }
                )
                + "\n"
            )


def load_eval_record_sets(path: Path | str) -> list[CTRLEvalRecordSet]:
    """Load ``CTRLEvalRecordSet`` entries from a JSONL file.

    Args:
        path: Source file path.

    Returns:
        List of ``CTRLEvalRecordSet`` in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If any line contains malformed JSON, is missing a
                           required field, has an incorrect type, has
                           ``len(seeds) != len(records)``, or contains embedded
                           records with inconsistent path lengths.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval record-set file not found: {p}")

    results: list[CTRLEvalRecordSet] = []
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

        for field in ("seeds", "records"):
            if field not in data:
                raise ValueError(
                    f"Missing required field '{field}' on line {lineno}"
                )

        seeds_raw = data["seeds"]
        records_raw = data["records"]

        _expect_list_of_ints(seeds_raw, "seeds", lineno)

        if not isinstance(records_raw, list):
            raise ValueError(
                f"Field 'records' on line {lineno} must be a list, "
                f"got {type(records_raw).__name__!r}"
            )

        if len(seeds_raw) != len(records_raw):
            raise ValueError(
                f"Line {lineno}: len(seeds)={len(seeds_raw)} != "
                f"len(records)={len(records_raw)}"
            )

        records = [
            _dict_to_record(rec_data, i, lineno)
            for i, rec_data in enumerate(records_raw)
        ]

        results.append(CTRLEvalRecordSet(seeds=list(seeds_raw), records=records))

    return results
