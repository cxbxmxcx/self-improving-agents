"""Effort ladders: the reasoning settings a model family accepts, in order, and what each one sends.

The harness reasons about reasoning in rungs, cheapest first. The study used
four (`none`, `low`, `medium`, `high`); providers accept a few more. A ladder
turns a rung into the call parameters for one model family. Both families
supported so far take LiteLLM's `reasoning_effort`, which LiteLLM translates:

- OpenAI (GPT-5.x and the o-series): sent unchanged.
- Anthropic (Claude): LiteLLM 1.106 sends `none` as no thinking. Other rungs
  become adaptive thinking at that effort on the Claude 5 family, or a thinking
  budget (1,024, 2,048 or 4,096 tokens) on earlier models. tests/test_a1.py
  pins this translation.

A model from any other family gets the default ladder, which sends
`reasoning_effort` and leaves the translation to LiteLLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Every rung any family accepts, cheapest first: the order rungs compare in.
ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# The study's four settings: what calibration sweeps by default.
RUNGS = ("none", "low", "medium", "high")


@dataclass(frozen=True)
class EffortLadder:
    """One model family's reasoning settings and the parameter that carries them."""

    family: str
    rungs: tuple[str, ...] = ORDER
    param: str = "reasoning_effort"

    def params(self, rung: str) -> dict[str, Any]:
        """The call parameters that set `rung`."""
        if rung not in self.rungs:
            raise ValueError(f"{rung!r} is not a reasoning setting for {self.family} models; use one of {self.rungs}")
        return {self.param: rung}


LADDERS: dict[str, EffortLadder] = {
    "openai": EffortLadder("openai", ("none", "minimal", "low", "medium", "high", "xhigh")),
    "anthropic": EffortLadder("anthropic", ORDER),
    "default": EffortLadder("default"),
}


def family(model: str) -> str:
    """The family a model belongs to, read from its provider prefix or its name."""
    name = (model or "").lower()
    provider, _, rest = name.partition("/")
    if rest:
        if provider in ("openai", "azure"):
            return "openai"
        if "claude" in rest and provider in ("anthropic", "bedrock", "vertex_ai"):
            return "anthropic"
        name = rest
    if name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    return "default"


def ladder_for(model: str) -> EffortLadder:
    return LADDERS[family(model)]


def bare(model: str | None) -> str:
    """A model's name without its provider prefix, so floors measured on `openai/x` also serve `x`."""
    return (model or "").split("/", 1)[-1]


def rank(rung: str) -> int:
    return ORDER.index(rung) if rung in ORDER else len(ORDER)


def most_demanding(rungs: Iterable[str | None]) -> str | None:
    """The highest of several rungs. None (the agent's own setting, unknown here) wins over any rung."""
    chosen = list(rungs)
    if not chosen or any(rung is None for rung in chosen):
        return None
    return max(chosen, key=rank)  # type: ignore[arg-type]
