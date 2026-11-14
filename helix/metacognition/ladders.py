"""Effort ladders: the reasoning settings a model family accepts, in order, and what each one sends.

The harness reasons about reasoning in rungs, cheapest first. The study used
four (`none`, `low`, `medium`, `high`); providers accept a few more. A ladder
turns a rung into the call parameters for one model family. Every ladder sends
`reasoning_effort`, and something downstream translates it:

- OpenAI (GPT-5.x and the o-series): LiteLLM sends it unchanged.
- Anthropic (Claude): helix.metacognition.claude sends Claude models to Anthropic's SDK and
  translates the rung there. `none` turns thinking off where the model can
  (`disabled` on Haiku 5.5, `between_tools` on Sonnet 5.5); every other rung is
  adaptive thinking at that effort. Models that can't turn thinking off (Opus 5.5,
  Fable 5.1, Fable 5.2) have no `none` rung. LiteLLM's own translation, which tests/test_a1.py
  pins, sends nothing for `none`, and that leaves thinking on.

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
    "anthropic": EffortLadder("anthropic", ("none", "low", "medium", "high", "xhigh", "max")),
    # Claude models whose thinking can't be turned off start at low.
    "anthropic-thinking": EffortLadder("anthropic-thinking", ("low", "medium", "high", "xhigh", "max")),
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
    name = family(model)
    if name == "anthropic":
        from helix.metacognition.claude import can_turn_thinking_off

        return LADDERS["anthropic" if can_turn_thinking_off(bare(model)) else "anthropic-thinking"]
    return LADDERS[name]


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
