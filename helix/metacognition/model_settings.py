"""Model call settings, shaped like the OpenAI Agents SDK's ModelSettings.

Settings reach each model call as a PRE_MODEL RequestPatch on the helix loop,
so a run's settings never change the loop itself. `resolve` layers a run's
settings over an agent's: any value the override sets wins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any

# The model a run, a writer, or a search uses when nothing names one.
DEFAULT_MODEL = os.environ.get("METACOG_DEFAULT_MODEL", "gpt-5.4-mini")
# A search for code needs the strongest model there is. A small model only iterates, so it serves as the agent and no more.
BUILDER_MODEL = os.environ.get("METACOG_BUILDER_MODEL", "claude-fable-5-2")


@dataclass
class Reasoning:
    """How hard a reasoning model thinks. `effort` takes the model's own setting names,
    such as "none", "low", "medium", and "high" for gpt-5.4-mini."""

    effort: str | None = None
    summary: str | None = None


def effort_of(reasoning: Any) -> str | None:
    """The effort named by a Reasoning, an OpenAI `Reasoning`, or a dict."""
    if reasoning is None:
        return None
    if isinstance(reasoning, dict):
        return reasoning.get("effort")
    return getattr(reasoning, "effort", None)


# Settings passed to the model under their own names when set.
_PASSED = ("temperature", "top_p", "max_tokens", "tool_choice", "parallel_tool_calls", "seed", "verbosity")


@dataclass
class ModelSettings:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: Reasoning | Any | None = None
    verbosity: str | None = None
    seed: int | None = None
    # Anything else LiteLLM accepts, passed through unchanged.
    extra_args: dict[str, Any] | None = None

    def resolve(self, override: ModelSettings | None) -> ModelSettings:
        """These settings with every value `override` sets layered on top."""
        if override is None:
            return self
        changes = {f.name: getattr(override, f.name) for f in fields(self) if getattr(override, f.name) is not None}
        if self.extra_args and override.extra_args:
            changes["extra_args"] = {**self.extra_args, **override.extra_args}
        return replace(self, **changes)

    def to_params(self) -> dict[str, Any]:
        """The call parameters these settings add, as LiteLLM names them."""
        params = {name: getattr(self, name) for name in _PASSED if getattr(self, name) is not None}
        effort = effort_of(self.reasoning)
        if effort is not None:
            params["reasoning_effort"] = effort
        params.update(self.extra_args or {})
        return params
