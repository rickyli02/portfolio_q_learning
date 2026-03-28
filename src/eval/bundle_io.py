"""Scalar-bundle file IO — Phase 15K.

Provides minimal helpers to persist and reload ``CTRLEvalScalarBundle`` entries
using a newline-delimited JSON (JSONL) format.  Each line is one bundle
serialised as a JSON object.  Because the bundle is already scalar-only there
are no tensor fields; the format is plain-text and human-readable without
additional dependencies.

SCOPE BOUNDARY
--------------
The following are NOT implemented here:
- plotting or report generation
- broader run-directory or artifact-management policy
- record / record-set / aggregate / summary logic changes
- compression or binary artifact conventions
- trainer integration
- stochastic evaluation

These belong in future bounded tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.aggregate import CTRLEvalAggregate
from src.eval.derive import CTRLEvalScalarBundle
from src.eval.summary import CTRLEvalSummary


# ---------------------------------------------------------------------------
# Type validators
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


# ---------------------------------------------------------------------------
# Embedded-object helpers
# ---------------------------------------------------------------------------


def _summary_to_dict(s: CTRLEvalSummary) -> dict:
    return {
        "terminal_wealth": s.terminal_wealth,
        "initial_wealth": s.initial_wealth,
        "target_return_z": s.target_return_z,
        "terminal_gap": s.terminal_gap,
        "n_steps": s.n_steps,
        "min_wealth": s.min_wealth,
        "max_wealth": s.max_wealth,
    }


def _aggregate_to_dict(a: CTRLEvalAggregate) -> dict:
    return {
        "n_episodes": a.n_episodes,
        "mean_terminal_wealth": a.mean_terminal_wealth,
        "min_terminal_wealth": a.min_terminal_wealth,
        "max_terminal_wealth": a.max_terminal_wealth,
        "mean_terminal_gap": a.mean_terminal_gap,
        "target_hit_rate": a.target_hit_rate,
    }


def _parse_summary(data: object, idx: int, lineno: int) -> CTRLEvalSummary:
    ctx = f"summaries[{idx}]"
    if not isinstance(data, dict):
        raise ValueError(
            f"Line {lineno}: {ctx} must be a JSON object, "
            f"got {type(data).__name__!r}"
        )
    required = [
        "terminal_wealth", "initial_wealth", "n_steps", "min_wealth", "max_wealth"
    ]
    for field in required:
        if field not in data:
            raise ValueError(
                f"Line {lineno}: missing required field '{field}' in {ctx}"
            )
    _expect_float(data["terminal_wealth"], f"{ctx}.terminal_wealth", lineno)
    _expect_float(data["initial_wealth"], f"{ctx}.initial_wealth", lineno)
    _expect_int(data["n_steps"], f"{ctx}.n_steps", lineno)
    _expect_float(data["min_wealth"], f"{ctx}.min_wealth", lineno)
    _expect_float(data["max_wealth"], f"{ctx}.max_wealth", lineno)

    target_return_z = data.get("target_return_z")
    terminal_gap = data.get("terminal_gap")
    if target_return_z is not None:
        _expect_float(target_return_z, f"{ctx}.target_return_z", lineno)
    if terminal_gap is not None:
        _expect_float(terminal_gap, f"{ctx}.terminal_gap", lineno)

    return CTRLEvalSummary(
        terminal_wealth=float(data["terminal_wealth"]),
        initial_wealth=float(data["initial_wealth"]),
        target_return_z=float(target_return_z) if target_return_z is not None else None,
        terminal_gap=float(terminal_gap) if terminal_gap is not None else None,
        n_steps=int(data["n_steps"]),
        min_wealth=float(data["min_wealth"]),
        max_wealth=float(data["max_wealth"]),
    )


def _parse_aggregate(data: object, lineno: int) -> CTRLEvalAggregate:
    ctx = "aggregate"
    if not isinstance(data, dict):
        raise ValueError(
            f"Line {lineno}: {ctx} must be a JSON object, "
            f"got {type(data).__name__!r}"
        )
    required = [
        "n_episodes", "mean_terminal_wealth", "min_terminal_wealth", "max_terminal_wealth"
    ]
    for field in required:
        if field not in data:
            raise ValueError(
                f"Line {lineno}: missing required field '{field}' in {ctx}"
            )
    _expect_int(data["n_episodes"], f"{ctx}.n_episodes", lineno)
    _expect_float(data["mean_terminal_wealth"], f"{ctx}.mean_terminal_wealth", lineno)
    _expect_float(data["min_terminal_wealth"], f"{ctx}.min_terminal_wealth", lineno)
    _expect_float(data["max_terminal_wealth"], f"{ctx}.max_terminal_wealth", lineno)

    mean_terminal_gap = data.get("mean_terminal_gap")
    target_hit_rate = data.get("target_hit_rate")
    if mean_terminal_gap is not None:
        _expect_float(mean_terminal_gap, f"{ctx}.mean_terminal_gap", lineno)
    if target_hit_rate is not None:
        _expect_float(target_hit_rate, f"{ctx}.target_hit_rate", lineno)

    return CTRLEvalAggregate(
        n_episodes=int(data["n_episodes"]),
        mean_terminal_wealth=float(data["mean_terminal_wealth"]),
        min_terminal_wealth=float(data["min_terminal_wealth"]),
        max_terminal_wealth=float(data["max_terminal_wealth"]),
        mean_terminal_gap=float(mean_terminal_gap) if mean_terminal_gap is not None else None,
        target_hit_rate=float(target_hit_rate) if target_hit_rate is not None else None,
    )


# ---------------------------------------------------------------------------
# Public IO helpers
# ---------------------------------------------------------------------------


def save_eval_bundles(bundles: list[CTRLEvalScalarBundle], path: Path | str) -> None:
    """Save a list of ``CTRLEvalScalarBundle`` entries to a JSONL file.

    Each bundle is serialised as one JSON object per line.  The file is
    created or overwritten at ``path``.

    Args:
        bundles: List of ``CTRLEvalScalarBundle`` entries to save.
        path:    Destination file path (created or overwritten).
    """
    p = Path(path)
    with p.open("w") as fh:
        for b in bundles:
            fh.write(
                json.dumps(
                    {
                        "seeds": b.seeds,
                        "summaries": [_summary_to_dict(s) for s in b.summaries],
                        "aggregate": _aggregate_to_dict(b.aggregate),
                    }
                )
                + "\n"
            )


def load_eval_bundles(path: Path | str) -> list[CTRLEvalScalarBundle]:
    """Load ``CTRLEvalScalarBundle`` entries from a JSONL file.

    Args:
        path: Source file path.

    Returns:
        List of ``CTRLEvalScalarBundle`` in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If any line contains malformed JSON, is missing a
                           required field, has an incorrect type, has
                           ``len(seeds) != len(summaries)``, or has
                           ``aggregate.n_episodes != len(summaries)``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval bundle file not found: {p}")

    results: list[CTRLEvalScalarBundle] = []
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

        for field in ("seeds", "summaries", "aggregate"):
            if field not in data:
                raise ValueError(
                    f"Missing required field '{field}' on line {lineno}"
                )

        seeds_raw = data["seeds"]
        summaries_raw = data["summaries"]

        _expect_list_of_ints(seeds_raw, "seeds", lineno)

        if not isinstance(summaries_raw, list):
            raise ValueError(
                f"Field 'summaries' on line {lineno} must be a list, "
                f"got {type(summaries_raw).__name__!r}"
            )

        if len(seeds_raw) != len(summaries_raw):
            raise ValueError(
                f"Line {lineno}: len(seeds)={len(seeds_raw)} != "
                f"len(summaries)={len(summaries_raw)}"
            )

        summaries = [
            _parse_summary(s, i, lineno) for i, s in enumerate(summaries_raw)
        ]
        aggregate = _parse_aggregate(data["aggregate"], lineno)

        if aggregate.n_episodes != len(summaries):
            raise ValueError(
                f"Line {lineno}: aggregate.n_episodes={aggregate.n_episodes} != "
                f"len(summaries)={len(summaries)}"
            )

        results.append(
            CTRLEvalScalarBundle(
                seeds=list(seeds_raw),
                summaries=summaries,
                aggregate=aggregate,
            )
        )

    return results
