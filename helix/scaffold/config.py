"""The decision configuration and the thresholds the scaffold acts on.

The configuration names the questions the decision model answers at each
hook and the rules that turn answers into control actions. It is a JSON
text artifact (`text:monitor`), so a text search can propose a new version
and this schema decides whether the proposal is usable. The thresholds are
a separate artifact holding only named numbers, so an online improver can
move them without touching the questions or the rules.

Confirmation rules may not read a named threshold. That keeps every safety
rule out of reach of online tuning: no change to the thresholds artifact
can relax a confirmation.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator

from helix.artifact import Artifact, Subtype, genesis
from helix.deciders.base import Answer, Choice, Probability, Question, Score

HOOKS = ("route", "tool", "progress", "output", "reflect")
ACTIONS_BY_HOOK: dict[str, frozenset[str]] = {
    "route": frozenset({"ask", "deliberate"}),
    "tool": frozenset({"confirm"}),
    "progress": frozenset({"finish"}),
    "output": frozenset({"revise"}),
    "reflect": frozenset({"reflect"}),
}
# Actions that protect people other than the user; online tuning never reaches them.
SAFETY_ACTIONS = frozenset({"confirm"})


class QuestionSpec(BaseModel):
    """One typed question, asked at one hook."""

    model_config = ConfigDict(extra="forbid")

    hook: Literal["route", "tool", "progress", "output", "reflect"]
    type: Literal["choice", "score", "probability"]
    instructions: str = Field(min_length=1)
    # Choice options and what each covers; for a probability question,
    # optional meanings of "true" and "false".
    criteria: dict[str, str | None] | None = None
    # Score levels, lowest first.
    levels: list[str] | None = None

    @model_validator(mode="after")
    def _shape_matches_type(self) -> QuestionSpec:
        if self.type == "choice":
            if not self.criteria or len(self.criteria) < 2:
                raise ValueError("a choice question needs at least two options in criteria")
            if self.levels:
                raise ValueError("only score questions have levels")
        elif self.type == "score":
            if not self.levels or len(self.levels) < 2 or len(set(self.levels)) != len(self.levels):
                raise ValueError("a score question needs at least two distinct levels")
            if self.criteria:
                raise ValueError("a score question takes levels, not criteria")
        else:
            if self.levels:
                raise ValueError("only score questions have levels")
            if self.criteria and set(self.criteria) - {"true", "false"}:
                raise ValueError("a probability question's criteria may only define 'true' and 'false'")
        return self

    def to_question(self) -> Question:
        if self.type == "choice":
            return Choice(self.instructions, dict(self.criteria or {}))
        if self.type == "score":
            return Score(self.instructions, list(self.levels or []))
        criteria = {k: v for k, v in (self.criteria or {}).items() if v}
        return Probability(self.instructions, criteria or None)


class Rule(BaseModel):
    """Turn one answer into one action.

    `>=` and `<` compare a number with a threshold: the probability, the
    score's position on [0, 1], or, for a choice question, the probability
    of `option`. `in` and `not_in` test the chosen option against `value`.
    A string threshold names an entry in the thresholds artifact; a number
    is fixed in the configuration itself.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    op: Literal[">=", "<", "in", "not_in"]
    action: str
    threshold: str | float | None = None
    value: list[str] | None = None
    option: str | None = None

    @field_validator("threshold")
    @classmethod
    def _fixed_threshold_in_range(cls, v: str | float | None) -> str | float | None:
        if isinstance(v, (int, float)) and not 0.0 <= float(v) <= 1.0:
            raise ValueError("a fixed threshold must lie in [0, 1]")
        if isinstance(v, str) and not v.strip():
            raise ValueError("a named threshold needs a name")
        return v

    @model_validator(mode="after")
    def _operands_match_op(self) -> Rule:
        if self.op in (">=", "<"):
            if self.threshold is None:
                raise ValueError(f"rule on {self.question!r}: {self.op} needs a threshold")
            if self.value is not None:
                raise ValueError(f"rule on {self.question!r}: {self.op} takes a threshold, not a value list")
        else:
            if not self.value:
                raise ValueError(f"rule on {self.question!r}: {self.op} needs a non-empty value list")
            if self.threshold is not None or self.option is not None:
                raise ValueError(f"rule on {self.question!r}: {self.op} takes a value list only")
        if self.action in SAFETY_ACTIONS and isinstance(self.threshold, str):
            raise ValueError(
                f"rule on {self.question!r}: a {self.action} rule may not read the online thresholds; "
                "use a fixed number or an option list"
            )
        return self

    @property
    def threshold_name(self) -> str | None:
        return self.threshold if isinstance(self.threshold, str) else None

    def resolved_threshold(self, thresholds: dict[str, float]) -> float | None:
        if self.threshold is None:
            return None
        if isinstance(self.threshold, str):
            return thresholds[self.threshold]
        return float(self.threshold)

    def observed(self, answer: Answer) -> float | str | None:
        """The part of the answer this rule compares."""
        if self.op in ("in", "not_in"):
            return answer.value if isinstance(answer.value, str) else None
        if answer.kind == "choice":
            return float(answer.probabilities.get(self.option or "", 0.0))
        if answer.kind == "score":
            return answer.position
        return float(answer.value) if isinstance(answer.value, (int, float)) else None

    def fires(self, answer: Answer | None, thresholds: dict[str, float]) -> bool:
        """Whether the action applies. An unreadable answer fires a safety
        rule and nothing else: when in doubt, confirm."""
        observed = self.observed(answer) if answer is not None else None
        if observed is None:
            return self.action in SAFETY_ACTIONS
        if self.op == "in":
            return observed in (self.value or [])
        if self.op == "not_in":
            return observed not in (self.value or [])
        threshold = self.resolved_threshold(thresholds)
        return observed >= threshold if self.op == ">=" else observed < threshold


class DecisionConfig(BaseModel):
    """The questions and rules that direct the metacognitive scaffold."""

    model_config = ConfigDict(extra="forbid")

    questions: dict[str, QuestionSpec]
    rules: list[Rule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rules_fit_their_questions(self) -> DecisionConfig:
        for rule in self.rules:
            spec = self.questions.get(rule.question)
            if spec is None:
                raise ValueError(f"rule names an unknown question: {rule.question!r}")
            allowed = ACTIONS_BY_HOOK[spec.hook]
            if rule.action not in allowed:
                raise ValueError(
                    f"rule on {rule.question!r}: action {rule.action!r} is not available at the "
                    f"{spec.hook} hook (expected one of {sorted(allowed)})"
                )
            if rule.op in ("in", "not_in"):
                if spec.type != "choice":
                    raise ValueError(f"rule on {rule.question!r}: {rule.op} needs a choice question")
                unknown = set(rule.value or []) - set(spec.criteria or {})
                if unknown:
                    raise ValueError(f"rule on {rule.question!r}: unknown options {sorted(unknown)}")
            elif spec.type == "choice":
                if rule.option not in (spec.criteria or {}):
                    raise ValueError(f"rule on {rule.question!r}: {rule.op} on a choice question needs one of its options")
            elif rule.option is not None:
                raise ValueError(f"rule on {rule.question!r}: only choice questions take an option")
        return self

    def questions_for(self, hook: str) -> dict[str, Question]:
        return {name: spec.to_question() for name, spec in self.questions.items() if spec.hook == hook}

    def rules_for(self, hook: str) -> list[Rule]:
        return [r for r in self.rules if self.questions[r.question].hook == hook]

    def threshold_names(self) -> set[str]:
        return {r.threshold_name for r in self.rules if r.threshold_name}


def apply_rules(config: DecisionConfig, hook: str, answers: dict[str, Answer],
                thresholds: dict[str, float]) -> tuple[set[str], list[dict[str, Any]]]:
    """The actions a hook's rules take on these answers, and the rules that fired."""
    fired: list[dict[str, Any]] = []
    for rule in config.rules_for(hook):
        if rule.fires(answers.get(rule.question), thresholds):
            fired.append({"question": rule.question, "action": rule.action, "op": rule.op,
                          "threshold": rule.resolved_threshold(thresholds), "named": rule.threshold_name,
                          "value": rule.value, "option": rule.option})
    return {f["action"] for f in fired}, fired


class Thresholds(RootModel[dict[str, float]]):
    """Named cut points on [0, 1], the only numbers online tuning may move."""

    @field_validator("root")
    @classmethod
    def _in_unit_range(cls, values: dict[str, float]) -> dict[str, float]:
        for name, v in values.items():
            if not name.strip():
                raise ValueError("a threshold needs a name")
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"threshold {name!r} must lie in [0, 1], got {v}")
        return {k: float(v) for k, v in values.items()}


def _parsed(content: Any) -> Any:
    return json.loads(content) if isinstance(content, str) else content


def load_config(artifact: Artifact) -> DecisionConfig:
    """The configuration an artifact holds; raises ValueError when it is unusable."""
    try:
        return DecisionConfig.model_validate(_parsed(artifact.content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact.id}:v{artifact.version} is not valid JSON: {exc}") from exc


def load_thresholds(artifact: Artifact) -> dict[str, float]:
    try:
        return Thresholds.model_validate(_parsed(artifact.content)).root
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact.id}:v{artifact.version} is not valid JSON: {exc}") from exc


def check_config(content: str | dict[str, Any], thresholds: dict[str, float] | None = None) -> str | None:
    """None when a proposed configuration is usable, else the reason it is
    not, so a search can discard a broken proposal and say why."""
    try:
        config = DecisionConfig.model_validate(_parsed(content))
    except json.JSONDecodeError as exc:
        return f"not valid JSON: {exc}"
    except ValidationError as exc:
        # One short line per problem: this text goes back to a proposer.
        problems = []
        for error in exc.errors():
            message = str(error.get("msg", "")).removeprefix("Value error, ")
            where = ".".join(str(part) for part in error.get("loc", ()))
            problems.append(f"{where}: {message}" if where else message)
        return "; ".join(problems)
    except (ValueError, TypeError) as exc:
        return str(exc)
    if thresholds is not None:
        missing = config.threshold_names() - set(thresholds)
        if missing:
            return f"rules name thresholds that do not exist: {sorted(missing)}"
    return None


def config_artifact(id: str, config: DecisionConfig | dict[str, Any], created_by: str = "human") -> Artifact:
    """A genesis `text:monitor` artifact holding a validated configuration."""
    model = config if isinstance(config, DecisionConfig) else DecisionConfig.model_validate(config)
    text = json.dumps(model.model_dump(exclude_none=True), indent=2, ensure_ascii=False)
    return genesis(id, Subtype.MONITOR, text, created_by=created_by, metadata={"format": "decision_config"})


def thresholds_artifact(id: str, values: dict[str, float], created_by: str = "human") -> Artifact:
    """A genesis `text:monitor` artifact holding named thresholds."""
    checked = Thresholds.model_validate(values).root
    return genesis(id, Subtype.MONITOR, json.dumps(checked, indent=2), created_by=created_by,
                   metadata={"format": "thresholds"})
