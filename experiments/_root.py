"""Repo-root helpers for offline experiment scripts.

Importing this module puts the repository root on sys.path so `nanovllm`
resolves without requiring the caller cwd to be the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Keep the script directory (experiments/) at sys.path[0] so sibling
# imports like `align_hf` still resolve. Put the repo root next.
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))


def result_path(name: str) -> Path:
    """Map a --save/--out value onto results/ under the repo root.

    Absolute paths are kept. Paths that already start with results/ are
    rooted at the repo. Bare names land in results/<name>.json.
    """
    path = Path(name)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "results":
        return ROOT / path
    if path.suffix != ".json":
        path = path.with_name(path.name + ".json")
    return RESULTS / path


def resolve_report(name: str) -> Path:
    """Resolve a --compare JSON path, preferring the repo root."""
    path = Path(name)
    if path.is_absolute():
        return path
    rooted = ROOT / path
    if rooted.exists() or (path.parts and path.parts[0] == "results"):
        return rooted
    return path
