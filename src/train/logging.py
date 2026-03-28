"""Trainer log record file IO — Phase 14B.

Provides minimal helpers to persist and reload ``CTRLLogRecord`` entries using
a newline-delimited JSON (JSONL) format.  Each line is one record; the format
is plain-text and human-readable without requiring additional dependencies.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- run-directory or artifact-management policy
- config-dispatch wiring
- plotting or DataFrame output
- checkpoint metadata sidecars
- auto-log cadence / callback systems

These belong in future bounded tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.train.log_record import CTRLLogRecord

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


def save_log_records(records: list[CTRLLogRecord], path: Path | str) -> None:
    """Save a list of ``CTRLLogRecord`` entries to a JSONL file.

    Each record is serialised as one JSON object per line.  The file is
    created or overwritten at ``path``.

    Args:
        records: List of ``CTRLLogRecord`` entries to save.
        path:    Destination file path (created or overwritten).
    """
    p = Path(path)
    with p.open("w") as fh:
        for rec in records:
            fh.write(
                json.dumps(
                    {
                        "current_w": rec.current_w,
                        "target_return_z": rec.target_return_z,
                        "w_step_size": rec.w_step_size,
                        "last_terminal_wealth": rec.last_terminal_wealth,
                        "last_w_prev": rec.last_w_prev,
                        "last_n_updates": rec.last_n_updates,
                    }
                )
                + "\n"
            )


def load_log_records(path: Path | str) -> list[CTRLLogRecord]:
    """Load ``CTRLLogRecord`` entries from a JSONL file.

    Args:
        path: Source file path.

    Returns:
        List of ``CTRLLogRecord`` in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If any line contains malformed JSON, is missing a
                           required field, or has an incorrect scalar type.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Log file not found: {p}")

    records: list[CTRLLogRecord] = []
    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON on line {lineno}: {exc}") from exc
        try:
            current_w = data["current_w"]
            target_return_z = data["target_return_z"]
            w_step_size = data["w_step_size"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required field {exc} on line {lineno}"
            ) from exc

        _expect_float(current_w, "current_w", lineno)
        _expect_float(target_return_z, "target_return_z", lineno)
        _expect_float(w_step_size, "w_step_size", lineno)

        last_terminal_wealth = data.get("last_terminal_wealth")
        last_w_prev = data.get("last_w_prev")
        last_n_updates = data.get("last_n_updates")

        if last_terminal_wealth is not None:
            _expect_float(last_terminal_wealth, "last_terminal_wealth", lineno)
        if last_w_prev is not None:
            _expect_float(last_w_prev, "last_w_prev", lineno)
        if last_n_updates is not None:
            _expect_int(last_n_updates, "last_n_updates", lineno)

        records.append(
            CTRLLogRecord(
                current_w=current_w,
                target_return_z=target_return_z,
                w_step_size=w_step_size,
                last_terminal_wealth=last_terminal_wealth,
                last_w_prev=last_w_prev,
                last_n_updates=last_n_updates,
            )
        )

    return records
