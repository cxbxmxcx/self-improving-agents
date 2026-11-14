"""Shared test helpers.

Parity tests replay the book's chapter 6 recordings through metacog. They
find the book repository through helix itself (linked by
scripts/link_helix.py) and skip when a recording is missing.
"""

from __future__ import annotations

from pathlib import Path

import helix
import pytest

BOOK_ROOT = Path(helix.__file__).resolve().parents[1]
CH06_RECORDED = BOOK_ROOT / "chapters" / "ch06" / "assistant" / "recorded"


@pytest.fixture
def ch06_recording():
    def path(name: str) -> Path:
        recorded = CH06_RECORDED / name
        if not recorded.exists():
            pytest.skip(f"book recording missing: {recorded}")
        return recorded

    return path
