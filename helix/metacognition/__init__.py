"""Metacognition: procedures an agent learns, and the online loop that keeps them trustworthy."""

from helix.metacognition.api import SkillAPI
from helix.metacognition.events import MetacognitionEvent
from helix.metacognition.judge import Judge
from helix.metacognition.mechanisms import (
    CONTROL,
    MECHANISMS,
    Mechanism,
    NoExperience,
    Reflection,
    SkillSynthesis,
    SuccessDistillation,
    TransitionDistillation,
    ValueLearning,
    resolve_mechanism,
)
from helix.metacognition.parts import (
    ADEQUATE,
    EFFORTS,
    SOLID,
    Controller,
    Detector,
    Gate,
    GateResult,
    LearningBudget,
    Practice,
    Router,
    Verdict,
)
from helix.metacognition.parts import Recorder
from helix.metacognition.procedure import Procedure, procedure
from helix.metacognition.prompts import CALENDAR, GENERIC, PromptPack
from helix.metacognition.replay import RecordedWorld, ReplayPractice
from helix.metacognition.runtime import AgentPractice, Metacognition, MetaReport, RunPlan
from helix.metacognition.sandbox import sandboxed_tool
from helix.metacognition.scopes import PerUser
from helix.metacognition.selection import DesignPolicy, LearnerChoice, LearnerSelection, LearningProblem, run_tournament
from helix.metacognition.sources import SEEDS, SourceLearner, seed_source
from helix.metacognition.store import MetaStore

__all__ = [
    "Recorder",
    "DesignPolicy",
    "LearnerChoice",
    "LearnerSelection",
    "LearningProblem",
    "run_tournament",
    "SEEDS",
    "SourceLearner",
    "seed_source",
    "ADEQUATE", "CALENDAR", "CONTROL", "EFFORTS", "GENERIC", "MECHANISMS", "SOLID", "AgentPractice", "Controller",
    "Detector", "Gate", "GateResult", "Judge", "LearningBudget", "Mechanism", "MetaReport", "MetaStore",
    "Metacognition", "MetacognitionEvent", "NoExperience", "PerUser", "Practice", "Procedure", "PromptPack",
    "RecordedWorld", "Reflection", "ReplayPractice", "Router", "RunPlan", "SkillAPI", "SkillSynthesis",
    "SuccessDistillation", "TransitionDistillation", "ValueLearning", "Verdict", "procedure", "resolve_mechanism",
    "sandboxed_tool",
]

from helix.metacognition.agent import Agent  # noqa: E402 - the agent and runner sit above the parts they use
from helix.metacognition.model_settings import ModelSettings, Reasoning  # noqa: E402
from helix.metacognition.result import RunResult  # noqa: E402
from helix.metacognition.run import DEFAULT_MAX_TURNS, RunConfig, Runner  # noqa: E402
from helix.metacognition.run_context import RunContextWrapper  # noqa: E402
from helix.metacognition.tool import FunctionTool, function_tool  # noqa: E402

__all__ += ["Agent", "DEFAULT_MAX_TURNS", "FunctionTool", "ModelSettings", "Reasoning", "RunConfig", "RunContextWrapper",
            "RunResult", "Runner", "function_tool"]
