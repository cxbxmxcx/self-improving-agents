"""Decision models: fast, typed judgments for the harness (chapter 6).

`decider_from_name` picks an adapter by name, so scripts can switch between
Jev, OpenAI's Decisions API, an LLM fallback, and a recorded replay with one
argument:

    jev-latest                 TypeSafe's Jev
    openai/gpt-6-luna          OpenAI's Decisions API
    llm/<litellm model id>     an LLM answering without reasoning
    replay:<path to .jsonl>    recorded answers, no key needed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helix.deciders.base import (
    Answer,
    Choice,
    Decider,
    DecisionUsage,
    Probability,
    Question,
    Score,
    ask_measured,
    question_from_dict,
    question_key,
    question_to_dict,
    score_position,
)
from helix.deciders.llm import LLMDecider
from helix.deciders.openai_decisions import OpenAIDecisionsDecider
from helix.deciders.replay import RecordingDecider, ReplayDecider
from helix.deciders.base import MissingAnswer
from helix.deciders.typesafe import TypeSafeDecider


def decider_from_name(name: str, **kwargs: Any) -> Decider:
    """The decision model a name refers to."""
    if name.startswith("replay:"):
        return ReplayDecider(Path(name.split(":", 1)[1]), **kwargs)
    if name.startswith("llm/"):
        return LLMDecider(name.split("/", 1)[1], **kwargs)
    if name.startswith("openai/"):
        return OpenAIDecisionsDecider(name.split("/", 1)[1], **kwargs)
    if name.startswith("jev"):
        return TypeSafeDecider(name, **kwargs)
    raise ValueError(f"unknown decision model: {name!r} (expected jev-*, openai/*, llm/*, or replay:<path>)")


def recorded_decider(path: Path | str, *, mode: str = "replay", model: str = "jev-latest", **kwargs: Any) -> Decider:
    """A decision model behind a recording. `replay` needs no key; `record` asks the model only for what the
    recording lacks and appends it; `live` asks the model every time and appends."""
    if mode == "replay":
        return ReplayDecider(path)
    if mode not in ("record", "live"):
        raise ValueError(f"mode must be replay, record or live, not {mode!r}")
    recording = RecordingDecider(decider_from_name(model, **kwargs), path)
    return recording if mode == "live" else ReplayDecider(path, fallback=recording)


__all__ = [
    "MissingAnswer", "recorded_decider",
    "Answer", "Choice", "Decider", "DecisionUsage", "ask_measured", "LLMDecider", "OpenAIDecisionsDecider", "Probability",
    "Question", "RecordingDecider", "ReplayDecider", "Score", "TypeSafeDecider", "decider_from_name",
    "question_from_dict", "question_key", "question_to_dict", "score_position",
]
