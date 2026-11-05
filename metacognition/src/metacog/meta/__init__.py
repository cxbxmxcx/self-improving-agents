"""Metacognition: procedures an agent learns, and the online loop that keeps them trustworthy."""

from metacog.meta.api import SkillAPI
from metacog.meta.events import MetacognitionEvent
from metacog.meta.judge import Judge
from metacog.meta.mechanisms import (
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
from metacog.meta.parts import (
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
from metacog.meta.procedure import Procedure, procedure
from metacog.meta.prompts import CALENDAR, GENERIC, PromptPack
from metacog.meta.replay import RecordedWorld, ReplayPractice
from metacog.meta.runtime import AgentPractice, Metacognition, MetaReport, RunPlan
from metacog.meta.sandbox import sandboxed_tool
from metacog.meta.scopes import PerUser
from metacog.meta.store import MetaStore

__all__ = [
    "ADEQUATE", "CALENDAR", "CONTROL", "EFFORTS", "GENERIC", "MECHANISMS", "SOLID", "AgentPractice", "Controller",
    "Detector", "Gate", "GateResult", "Judge", "LearningBudget", "Mechanism", "MetaReport", "MetaStore",
    "Metacognition", "MetacognitionEvent", "NoExperience", "PerUser", "Practice", "Procedure", "PromptPack",
    "RecordedWorld", "Reflection", "ReplayPractice", "Router", "RunPlan", "SkillAPI", "SkillSynthesis",
    "SuccessDistillation", "TransitionDistillation", "ValueLearning", "Verdict", "procedure", "resolve_mechanism",
    "sandboxed_tool",
]
