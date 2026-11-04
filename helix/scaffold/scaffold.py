"""The metacognitive scaffold: a decision model directing hook wrappers.

`MetacognitiveScaffold` holds what the meta level reads (the decision model,
the decision configuration, the thresholds, the planner and reflector
prompts, and the two paths' models) and registers four wrappers on an
agent's hooks. It adds no loop branch: the router and task state act at
PRE_MODEL, the monitor at PRE_TOOL and PRE_OUTPUT, task state again at
END_OF_TURN, and the reflector at SESSION_END. SPEC §6.2 and §16.2.4.

The configuration, planner, and reflector are bound in a metacognition
composite (L3), improved jointly offline. The thresholds stay outside it,
an L1 artifact an online improver may tune.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from helix.artifact import Artifact, Subtype, compose, composite_content
from helix.hooks import HookPoint
from helix.scaffold.config import DecisionConfig, load_config, load_thresholds
from helix.scaffold.monitor import Monitor
from helix.scaffold.reflector import Reflector
from helix.scaffold.router import Router
from helix.scaffold.task_state import TaskState
from helix.trajectory import Trajectory

if TYPE_CHECKING:
    from helix.agent import Agent
    from helix.deciders.base import Decider

# Composite roles, in binding order. The thresholds are never among them.
ROLES = ("config", "planner", "reflector")


def _expect(artifact: Artifact, subtype: Subtype, slot: str) -> None:
    if artifact.subtype is not subtype:
        raise ValueError(f"the scaffold's {slot} must be a text:{subtype.value} artifact, "
                         f"got {artifact.kind.value}:{artifact.subtype.value if artifact.subtype else None}")


@dataclass
class MetacognitiveScaffold:
    """The meta level an `Agent(scaffold=...)` registers on its hooks."""

    decider: Decider
    config: Artifact
    thresholds: Artifact
    planner: Artifact | None = None
    reflector: Artifact | None = None
    # The fast path is the default; None keeps the agent's own model.
    fast_model: str | None = None
    fast_params: dict[str, Any] = field(default_factory=dict)
    deliberate_model: str | None = None
    deliberate_params: dict[str, Any] = field(default_factory=dict)
    # Revision and reflection are System 2 calls; None uses the deliberate path.
    revision_model: str | None = None
    reflection_model: str | None = None
    # Tools whose calls may affect other people; only these are checked.
    watch_tools: frozenset[str] = frozenset()
    # Whether the user already confirmed this call: (trajectory, tool, arguments).
    confirmed: Callable[[Trajectory, str, dict[str, Any]], bool | Awaitable[bool]] | None = None
    # The run's reward, as in chapter 5; the reflector learns only from runs below 1.
    reward_fn: Callable[[Trajectory], float | None | Awaitable[float | None]] | None = None
    # Adds application state (the calendar, the inbox) to a hook's decision state.
    state_fn: Callable[[str, dict[str, Any], Trajectory], Any] | None = None
    composite_id: str | None = "metacognition"
    composite: Artifact | None = None
    # "raise" surfaces a failed decision; "safe" records it and takes the safe default.
    on_error: str = "raise"
    decision_config: DecisionConfig = field(init=False, repr=False)
    threshold_values: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _expect(self.config, Subtype.MONITOR, "config")
        _expect(self.thresholds, Subtype.MONITOR, "thresholds")
        if self.planner is not None:
            _expect(self.planner, Subtype.PLANNER, "planner")
        if self.reflector is not None:
            _expect(self.reflector, Subtype.REFLECTOR, "reflector")
        if self.thresholds.id in {a.id for a in self.constituents().values()}:
            raise ValueError("the thresholds must be their own artifact, outside the composite")
        if self.on_error not in ("raise", "safe"):
            raise ValueError(f"on_error must be 'raise' or 'safe', got {self.on_error!r}")
        self.watch_tools = frozenset(self.watch_tools)
        self.decision_config = load_config(self.config)
        self.threshold_values = load_thresholds(self.thresholds)
        missing = self.decision_config.threshold_names() - set(self.threshold_values)
        if missing:
            raise ValueError(f"rules name thresholds that {self.thresholds.id} does not hold: {sorted(missing)}")
        bound = {role: a.ref for role, a in self.constituents().items()}
        if self.composite is None and self.composite_id:
            self.composite = compose(self.composite_id, [(a, r) for r, a in self.constituents().items()],
                                     subtype=Subtype.METACOGNITION)
        elif self.composite is not None:
            declared = {c["role"]: (c["id"], int(c["version"])) for c in self.composite.constituents}
            if declared != bound:
                raise ValueError(f"composite {self.composite.id}:v{self.composite.version} binds {declared}, "
                                 f"but the scaffold holds {bound}")

    def constituents(self) -> dict[str, Artifact]:
        """The composite's members by role."""
        out = {"config": self.config, "planner": self.planner, "reflector": self.reflector}
        return {role: a for role, a in out.items() if a is not None}

    def artifacts(self) -> list[Artifact]:
        """What the wrappers read. The agent declares the composite itself."""
        return [*self.constituents().values(), self.thresholds]

    def system2(self, agent: Agent, override: str | None) -> tuple[str, dict[str, Any]]:
        """The model and parameters of a System 2 call."""
        deliberate = self.deliberate_model or agent.model
        model = override or deliberate
        return model, (dict(self.deliberate_params) if model == deliberate else {})

    def register(self, agent: Agent) -> None:
        """Register the wrappers on the agent's hooks. The router precedes task
        state at PRE_MODEL, so task state's patch refines the route's."""
        router, monitor = Router(self, agent), Monitor(self, agent)
        task_state, reflector = TaskState(self, agent), Reflector(self, agent)
        agent.hooks.register(HookPoint.PRE_MODEL, router.pre_model)
        agent.hooks.register(HookPoint.PRE_MODEL, task_state.pre_model)
        agent.hooks.register(HookPoint.PRE_TOOL, monitor.pre_tool)
        agent.hooks.register(HookPoint.END_OF_TURN, task_state.end_of_turn)
        agent.hooks.register(HookPoint.PRE_OUTPUT, monitor.pre_output)
        agent.hooks.register(HookPoint.SESSION_END, reflector.session_end)

    def with_overrides(self, overrides: dict[str, Artifact]) -> MetacognitiveScaffold:
        """This scaffold with any of its artifacts swapped, for `Agent.with_artifacts`.

        A composite override brings its constituents with it: each version it
        binds must be one the scaffold holds or one passed in `overrides`.
        Swapping a constituent alone rebinds the composite as a derived child
        version, so the composite always names what actually runs.
        """
        slots: dict[str, Artifact | None] = {"thresholds": self.thresholds, **{r: getattr(self, r) for r in ROLES}}
        changes: dict[str, Any] = {slot: overrides[a.id] for slot, a in slots.items()
                                   if a is not None and a.id in overrides}
        composite = self.composite
        if composite is not None and composite.id in overrides:
            composite = overrides[composite.id]
            known = {a.ref: a for a in [*slots.values(), *overrides.values()] if a is not None}
            roles = {c["role"]: (c["id"], int(c["version"])) for c in composite.constituents}
            unknown = set(roles) - set(ROLES)
            if unknown:
                raise ValueError(f"composite {composite.id} binds roles the scaffold does not have: {sorted(unknown)}")
            for role in ROLES:
                ref = roles.get(role)
                if ref is None:
                    changes[role] = None
                elif ref in known:
                    changes[role] = known[ref]
                else:
                    raise LookupError(f"composite {composite.id}:v{composite.version} binds {ref[0]}:v{ref[1]}; "
                                      "pass that version in the overrides too")
            changes["composite"] = composite
        elif composite is not None and any(r in changes for r in ROLES):
            members = {r: changes.get(r, getattr(self, r)) for r in ROLES}
            binding = composite.content.get("binding") if isinstance(composite.content, dict) else None
            changes["composite"] = composite.mutate(
                composite_content([(a, r) for r, a in members.items() if a is not None], binding),
                created_by="override",
                metadata={"derived_from": list(composite.ref), "overridden": sorted(r for r in ROLES if r in changes)},
            )
        if not changes:
            return self
        return dataclasses.replace(self, **changes)
