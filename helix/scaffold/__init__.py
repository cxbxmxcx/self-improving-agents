"""The metacognitive scaffold (chapter 6): a decision model directs hook
wrappers that route, confirm, track progress, revise, and reflect.

    scaffold = MetacognitiveScaffold(decider, config, thresholds, planner=..., reflector=...)
    agent = Agent(system_prompt, tools, scaffold=scaffold, memory_tiers=...)

Each wrapper records its decisions in `trajectory.metadata["metacognition"]`.
"""

from __future__ import annotations

from helix.scaffold.common import META_KEY, decider_usage, record
from helix.scaffold.config import (
    ACTIONS_BY_HOOK,
    HOOKS,
    SAFETY_ACTIONS,
    DecisionConfig,
    QuestionSpec,
    Rule,
    Thresholds,
    check_config,
    config_artifact,
    load_config,
    load_thresholds,
    thresholds_artifact,
)
from helix.scaffold.monitor import REVISION_PROMPT, Monitor
from helix.scaffold.reflector import Reflector
from helix.scaffold.router import ASK_NOTE, PLANNER_HEADING, Router
from helix.scaffold.scaffold import ROLES, MetacognitiveScaffold
from helix.scaffold.task_state import FINISH_NOTE, TaskState

__all__ = [
    "ACTIONS_BY_HOOK", "ASK_NOTE", "DecisionConfig", "FINISH_NOTE", "HOOKS", "META_KEY", "MetacognitiveScaffold",
    "Monitor", "PLANNER_HEADING", "QuestionSpec", "REVISION_PROMPT", "ROLES", "Reflector", "Router", "Rule",
    "SAFETY_ACTIONS", "TaskState", "Thresholds", "check_config", "config_artifact", "decider_usage",
    "load_config", "load_thresholds", "record", "thresholds_artifact",
]
