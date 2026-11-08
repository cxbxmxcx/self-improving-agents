"""helix is the framework: it never imports the book's chapters or the SDK built on top of it."""

from __future__ import annotations

import ast
from pathlib import Path

HELIX = Path(__file__).resolve().parents[1] / "helix"
ABOVE = ("metacog", "chapters", "agents")


def imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_helix_imports_nothing_above_it():
    offenders = {}
    for path in HELIX.rglob("*.py"):
        found = imported_roots(path.read_text(encoding="utf-8")) & set(ABOVE)
        if found:
            offenders[str(path.relative_to(HELIX))] = sorted(found)
    assert not offenders, offenders
