"""Evaluation aggregate file IO — Phase 15D.

Provides minimal helpers to persist and reload ``CTRLEvalAggregate`` entries
using a newline-delimited JSON (JSONL) format.  Each line is one aggregate
record; the format is plain-text and human-readable without additional
dependencies.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- run-directory or artifact-management policy
- config-dispatch wiring
- plotting or report generation
- benchmark comparison tables
- trainer integration
- file IO for per-episode summary bundles

These belong in future bounded tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.aggregate import CTRLEvalAggregate


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


def save_eval_aggregates(
    aggregates: list[CTRLEvalAggregate], path: Path | str
) -> None:
    """Save a list of ``CTRLEvalAggregate`` entries to a JSONL file.

    Each aggregate is serialised as one JSON object per line.  The file is
    created or overwritten at ``path``.

    Args:
        aggregates: List of ``CTRLEvalAggregate`` entries to save.
        path:       Destination file path (created or overwritten).
    """
    p = Path(path)
    with p.open("w") as fh:
        for agg in aggregates:
            fh.write(
                json.dumps(
                    {
                        "n_episodes": agg.n_episodes,
                        "mean_terminal_wealth": agg.mean_terminal_wealth,
                        "min_terminal_wealth": agg.min_terminal_wealth,
                        "max_terminal_wealth": agg.max_terminal_wealth,
                        "mean_terminal_gap": agg.mean_terminal_gap,
                        "target_hit_rate": agg.target_hit_rate,
                    }
                )
                + "\n"
            )


def load_eval_aggregates(path: Path | str) -> list[CTRLEvalAggregate]:
    """Load ``CTRLEvalAggregate`` entries from a JSONL file.

    Args:
        path: Source file path.

    Returns:
        List of ``CTRLEvalAggregate`` in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If any line contains malformed JSON, is missing a
                           required field, or has an incorrect scalar type.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval aggregate file not found: {p}")

    aggregates: list[CTRLEvalAggregate] = []
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

        try:
            n_episodes = data["n_episodes"]
            mean_terminal_wealth = data["mean_terminal_wealth"]
            min_terminal_wealth = data["min_terminal_wealth"]
            max_terminal_wealth = data["max_terminal_wealth"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required field {exc} on line {lineno}"
            ) from exc

        _expect_int(n_episodes, "n_episodes", lineno)
        _expect_float(mean_terminal_wealth, "mean_terminal_wealth", lineno)
        _expect_float(min_terminal_wealth, "min_terminal_wealth", lineno)
        _expect_float(max_terminal_wealth, "max_terminal_wealth", lineno)

        mean_terminal_gap = data.get("mean_terminal_gap")
        target_hit_rate = data.get("target_hit_rate")

        if mean_terminal_gap is not None:
            _expect_float(mean_terminal_gap, "mean_terminal_gap", lineno)
        if target_hit_rate is not None:
            _expect_float(target_hit_rate, "target_hit_rate", lineno)

        aggregates.append(
            CTRLEvalAggregate(
                n_episodes=n_episodes,
                mean_terminal_wealth=mean_terminal_wealth,
                min_terminal_wealth=min_terminal_wealth,
                max_terminal_wealth=max_terminal_wealth,
                mean_terminal_gap=mean_terminal_gap,
                target_hit_rate=target_hit_rate,
            )
        )

    return aggregates
