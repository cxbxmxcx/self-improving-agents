"""Agent spec loader.

Resolves agent names to modules in the top-level `agents/` package and
invokes their `build()` factory. Used by the chat UI, eval harness, and any
script that wants to operate on a named agent without hardcoding its
construction.

Usage:
    from helix.agents import load_agent, list_agents, load_genesis_prompt

    agent = load_agent("helpdesk")          # uses agents/helpdesk.py
    agent = load_agent("helpdesk", model="claude-sonnet-4-6")  # with overrides

    # Without instantiating:
    seed = load_genesis_prompt("helpdesk")  # returns the Artifact

The loader does not cache agents; each load_agent() returns a fresh Agent.
Memory tiers inside the Agent may still be shared singletons (their job).
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helix.agent import Agent
    from helix.artifact import Artifact


AGENTS_PACKAGE = "agents"


def list_agents() -> list[str]:
    """Return the names of every agent spec module in `agents/`.

    Excludes private modules (those starting with `_`) and the package
    init file.
    """
    try:
        pkg = importlib.import_module(AGENTS_PACKAGE)
    except ImportError:
        return []
    names: list[str] = []
    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        return []
    for finder, name, ispkg in pkgutil.iter_modules(pkg_path):
        if name.startswith("_"):
            continue
        names.append(name)
    return sorted(names)


def _import_spec(name: str):
    """Import the agent module by name."""
    try:
        return importlib.import_module(f"{AGENTS_PACKAGE}.{name}")
    except ImportError as exc:
        available = ", ".join(list_agents()) or "(none found)"
        raise ValueError(
            f"No agent spec named '{name}'. Available: {available}."
        ) from exc


def load_agent(name: str, **overrides: Any) -> "Agent":
    """Resolve the agent spec and return a fresh Agent instance.

    Overrides pass through to the spec's build() function. Common overrides:
    model, system_prompt, memory_tiers (to inject shared singletons), bus.
    """
    spec = _import_spec(name)
    build = getattr(spec, "build", None)
    if build is None or not callable(build):
        raise ValueError(
            f"Agent spec '{name}' does not export a build() function. "
            f"Every spec module must export build(system_prompt=None, **overrides) -> Agent."
        )
    return build(**overrides)


def load_genesis_prompt(name: str) -> "Artifact":
    """Return the spec's default system-prompt Artifact, unwrapped.

    Useful when you want to seed an Improver with the spec's genesis prompt
    without constructing the agent yet.
    """
    spec = _import_spec(name)
    builder = getattr(spec, "build_genesis_prompt", None)
    if builder is None or not callable(builder):
        raise ValueError(
            f"Agent spec '{name}' does not export build_genesis_prompt() -> Artifact."
        )
    return builder()


def describe_agent(name: str) -> dict[str, Any]:
    """Return basic metadata about a spec without instantiating it.

    Reads the module's docstring and any optional module-level constants
    (DESCRIPTION, DEFAULT_MODEL, SYSTEM_PROMPT_ID, etc.). For the dashboard
    or a "pick an agent" UI.
    """
    spec = _import_spec(name)
    return {
        "name": name,
        "module": spec.__name__,
        "docstring": (spec.__doc__ or "").strip(),
        "description": getattr(spec, "DESCRIPTION", ""),
        "default_model": getattr(spec, "DEFAULT_MODEL", ""),
        "system_prompt_id": getattr(spec, "SYSTEM_PROMPT_ID", ""),
        "has_build": callable(getattr(spec, "build", None)),
        "has_genesis": callable(getattr(spec, "build_genesis_prompt", None)),
        "has_improvers": callable(getattr(spec, "build_improvers", None)),
        "has_list_artifacts": callable(getattr(spec, "list_improvable_artifacts", None)),
    }


def load_improvers(name: str, agent: "Agent", *, archive_path: str | None = None) -> list:
    """Return the improvers declared by the agent spec, if any.

    Specs declare improvers by exporting:
        def build_improvers(agent, *, archive_path=None) -> list[Improver]: ...

    If the spec doesn't export this, returns an empty list. Specs decide
    their own defaults (a helpdesk might ship one prompt-improvement plan;
    a multi-agent system might ship one improver per artifact).

    The chat UI calls this when the user clicks "Attach default improvers from
    spec" and stacks the returned improvers in its panel. The user can still
    pause/detach any of them individually after attach.
    """
    spec = _import_spec(name)
    builder = getattr(spec, "build_improvers", None)
    if builder is None or not callable(builder):
        return []
    return list(builder(agent, archive_path=archive_path))


def list_improvable_artifacts(name: str, agent: "Agent") -> list:
    """Return the Artifacts on this agent that can be put under improvement.

    Specs declare improvable artifacts by exporting:
        def list_improvable_artifacts(agent) -> list[Artifact]: ...

    Useful for a UI that wants to show "this agent has 3 improvable
    artifacts" and let the user pick which to target. If not exported,
    falls back to the agent's system prompt — every agent has one and
    that's the canonical improvable artifact.
    """
    spec = _import_spec(name)
    lister = getattr(spec, "list_improvable_artifacts", None)
    if lister is not None and callable(lister):
        return list(lister(agent))
    return [agent.system_prompt]
