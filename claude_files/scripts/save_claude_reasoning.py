#!/usr/bin/env python3
"""Save a Claude reasoning note under claude_files/ with a timestamped name."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_FILES = REPO_ROOT / "claude_files"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "note"


def main() -> int:
    parser = argparse.ArgumentParser(description="Save a Claude reasoning note")
    parser.add_argument("--title", required=True, help="Short note title")
    parser.add_argument("--text", help="Inline note text")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read note body from stdin instead of --text",
    )
    args = parser.parse_args()

    if args.stdin:
        body = sys.stdin.read()
    elif args.text is not None:
        body = args.text
    else:
        parser.error("provide either --text or --stdin")

    CLAUDE_FILES.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{_slugify(args.title)}.md"
    path = CLAUDE_FILES / filename
    content = (
        f"# {args.title}\n\n"
        f"- Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"- Author: Claude Code\n\n"
        f"{body.rstrip()}\n"
    )
    path.write_text(content)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
