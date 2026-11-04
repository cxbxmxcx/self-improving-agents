"""Content diffs as SequenceMatcher opcodes, rendered by the client.

The Streamlit app shipped difflib's HTML three separate times; here the
server sends structured opcodes once and one JS component draws them. The
line cap guards against SequenceMatcher's quadratic worst case.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def diff_payload(
    a_text: str,
    b_text: str,
    a_label: str,
    b_label: str,
    max_lines: int = 4000,
) -> dict[str, Any]:
    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()
    truncated = len(a_lines) > max_lines or len(b_lines) > max_lines
    a_lines = a_lines[:max_lines]
    b_lines = b_lines[:max_lines]

    matcher = SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)
    opcodes = [
        {"tag": tag, "a1": a1, "a2": a2, "b1": b1, "b2": b2}
        for tag, a1, a2, b1, b2 in matcher.get_opcodes()
    ]
    stats = {
        "added": sum(o["b2"] - o["b1"] for o in opcodes if o["tag"] in ("insert", "replace")),
        "removed": sum(o["a2"] - o["a1"] for o in opcodes if o["tag"] in ("delete", "replace")),
        "unchanged": sum(o["a2"] - o["a1"] for o in opcodes if o["tag"] == "equal"),
    }
    return {
        "a": {"label": a_label, "lines": a_lines},
        "b": {"label": b_label, "lines": b_lines},
        "opcodes": opcodes,
        "stats": stats,
        "truncated": truncated,
    }
