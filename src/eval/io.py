"""Evaluation summary file IO — Phase 15B.

Provides minimal helpers to persist and reload ``CTRLEvalSummary`` entries
using a newline-delimited JSON (JSONL) format.  Each line is one record; the
format is plain-text and human-readable without additional dependencies.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- run-directory or artifact-management policy
- config-dispatch wiring
- plotting or report generation
- multi-episode backtesting summaries
- trainer integration

These belong in future bounded tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.summary import CTRLEvalSummary


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


def save_eval_summaries(
    summaries: list[CTRLEvalSummary], path: Path | str
) -> None:
    """Save a list of ``CTRLEvalSummary`` entries to a JSONL file.

    Each summary is serialised as one JSON object per line.  The file is
    created or overwritten at ``path``.

    Args:
        summaries: List of ``CTRLEvalSummary`` entries to save.
        path:      Destination file path (created or overwritten).
    """
    p = Path(path)
    with p.open("w") as fh:
        for s in summaries:
            fh.write(
                json.dumps(
                    {
                        "terminal_wealth": s.terminal_wealth,
                        "initial_wealth": s.initial_wealth,
                        "target_return_z": s.target_return_z,
                        "terminal_gap": s.terminal_gap,
                        "n_steps": s.n_steps,
                        "min_wealth": s.min_wealth,
                        "max_wealth": s.max_wealth,
                    }
                )
                + "\n"
            )


def load_eval_summaries(path: Path | str) -> list[CTRLEvalSummary]:
    """Load ``CTRLEvalSummary`` entries from a JSONL file.

    Args:
        path: Source file path.

    Returns:
        List of ``CTRLEvalSummary`` in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If any line contains malformed JSON, is missing a
                           required field, or has an incorrect scalar type.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval summary file not found: {p}")

    summaries: list[CTRLEvalSummary] = []
    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON on line {lineno}: {exc}") from exc

        # Extract and validate required fields
        try:
            terminal_wealth = data["terminal_wealth"]
            initial_wealth = data["initial_wealth"]
            n_steps = data["n_steps"]
            min_wealth = data["min_wealth"]
            max_wealth = data["max_wealth"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required field {exc} on line {lineno}"
            ) from exc

        _expect_float(terminal_wealth, "terminal_wealth", lineno)
        _expect_float(initial_wealth, "initial_wealth", lineno)
        _expect_int(n_steps, "n_steps", lineno)
        _expect_float(min_wealth, "min_wealth", lineno)
        _expect_float(max_wealth, "max_wealth", lineno)

        # Optional fields
        target_return_z = data.get("target_return_z")
        terminal_gap = data.get("terminal_gap")

        if target_return_z is not None:
            _expect_float(target_return_z, "target_return_z", lineno)
        if terminal_gap is not None:
            _expect_float(terminal_gap, "terminal_gap", lineno)

        summaries.append(
            CTRLEvalSummary(
                terminal_wealth=terminal_wealth,
                initial_wealth=initial_wealth,
                target_return_z=target_return_z,
                terminal_gap=terminal_gap,
                n_steps=n_steps,
                min_wealth=min_wealth,
                max_wealth=max_wealth,
            )
        )

    return summaries
