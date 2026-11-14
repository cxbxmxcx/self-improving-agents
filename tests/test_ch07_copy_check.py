"""Chapter 7's copy check: the builder never repeats wording that only the gate or the held-out requests contain.

The builder writes every rewrite, every component's memories and programs, and every relearned function, so its
recorded replies hold all the code a search produced. A six-word phrase that appears in a gate or held-out request but
in no request the builder could see (the experience, the search's requests, and the served week) would be copying.
Phrases those visible requests also contain, such as a run of colleagues' names, prove nothing and are left out.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from chapters.ch07.assistant import requests

RECORDED = Path(__file__).resolve().parents[1] / "chapters" / "ch07" / "assistant" / "recorded" / "models.jsonl"
HIDDEN = ("gate", "changed_gate", "held_out", "second_gate", "second_held_out")
VISIBLE = ("experience", "search", "second_experience", "second_search")
WORD = re.compile(r"[a-z0-9']+")


def six_word_runs(text: str) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {tuple(words[i:i + 6]) for i in range(len(words) - 5)}


def phrases(sets: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {run for name in sets for request in requests.load(name) for run in six_word_runs(request["text"])}


def builder_replies() -> list[str]:
    rows = [json.loads(line) for line in RECORDED.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row.get("content") or "" for row in rows if str(row.get("model", "")).startswith("claude")]


def test_no_builder_reply_repeats_a_six_word_phrase_found_only_in_gate_or_held_out_requests() -> None:
    visible = phrases(VISIBLE) | {run for day in requests.week().values()
                                  for request in day for run in six_word_runs(request["text"])}
    hidden_only = phrases(HIDDEN) - visible
    replies = builder_replies()
    assert replies, "the recording should hold the builder's replies"
    assert hidden_only, "the gate and held-out requests should have phrases of their own"
    copied = {" ".join(run) for reply in replies for run in six_word_runs(reply) & hidden_only}
    assert not copied, f"builder replies repeat hidden wording: {sorted(copied)[:5]}"
