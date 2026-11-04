"""Learning mechanisms: five ways to turn experience into a remembered function.

Each mechanism reads the same experience, the agent's own runs recalled from
episodic memory, and writes one function to a fixed contract. The function
is the learned memory: it is stored in procedural memory and recalled for
later requests. Mechanisms differ in what they learn from and how they
update, which is what a search over mechanisms compares.

- **Value learning (RL)** writes a candidate from each successful run, then
  scores candidates with chapter 5's fixed rule as it uses them on the task
  stream, and keeps the highest-valued one.
- **Reflection** uses its current function on the task stream, writes a
  lesson from each rejection, and revises the function to follow every
  lesson so far.
- **Success distillation** writes one function from all successful runs.
- **Transition distillation** writes one function from every step of every
  run, rejected bookings and errors included.
- **Skill synthesis** writes from the successful runs, executes the draft on
  the observed requests, and repairs it from what went wrong.

A sixth, **no experience**, writes from the contract alone; it is the
control a mechanism must beat. Nothing here knows about calendars: the
caller supplies the contract, the episodes, and `act`, which tries a
function on one request and returns the user's verdict.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from helix.agent import _usage_of
from helix.llm_call import acompletion as helix_acompletion
from helix.memory.code_skills import SAFE_BUILTINS, UnsafeSkill, check_source
from helix.trajectory import StepKind, Trajectory

# Chapter 5's learning rule: q_new = q + ALPHA * (credit - q), from a neutral start.
ALPHA = 0.5
INITIAL_VALUE = 0.5
MAX_RESULT_CHARS = 1500  # one tool result, as a writer sees it


@dataclass
class Contract:
    """What every learned function must be: its name, signature, and promise."""

    name: str
    signature: str  # the def line, with type hints
    description: str
    api: str  # what the bound `api` object offers, in one or two sentences
    returns: str


@dataclass
class Episode:
    """One run the agent made: the request, every tool step, and the verdict."""

    task_id: str
    request: str
    steps: list[dict[str, Any]]  # {"tool", "arguments", "result"}
    success: bool
    reason: str

    @classmethod
    def from_trajectory(cls, task_id: str, trajectory: Trajectory, success: bool, reason: str) -> Episode:
        steps, pending = [], None
        for step in trajectory.steps:
            if step.kind == StepKind.TOOL_CALL:
                pending = {"tool": step.payload.get("name"), "arguments": step.payload.get("arguments", {})}
            elif step.kind == StepKind.TOOL_RESULT and pending is not None:
                steps.append({**pending, "result": step.payload.get("result")})
                pending = None
        return cls(task_id, trajectory.task, steps, success, reason)

    def to_content(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "user_message": self.request, "steps": self.steps,
                "success": self.success, "reason": self.reason,
                "final_output": "accepted" if self.success else f"rejected: {self.reason}"}

    @classmethod
    def from_content(cls, content: dict[str, Any]) -> Episode:
        return cls(content["task_id"], content["user_message"], content["steps"], content["success"],
                   content["reason"])


def render_episode(episode: Episode) -> str:
    lines = [f"Request: {episode.request}"]
    for step in episode.steps:
        result = json.dumps(step["result"], default=str)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + " ..."
        lines.append(f"- {step['tool']}({json.dumps(step['arguments'])}) -> {result}")
    verdict = "accepted" if episode.success else f"rejected: {episode.reason}"
    lines.append(f"User's verdict: {verdict}")
    return "\n".join(lines)


# ----------------------------------------------------------------- prompts
# Module constants so a later chapter can put them under search.

CONTRACT_PROMPT = """You are writing a Python function that a calendar assistant will call
instead of working the answer out itself. Its contract:

```python
{signature}
    \"\"\"{description}\"\"\"
```

{api} {returns}

Rules: no imports, no names that begin with an underscore, and plain Python
only: functions, loops, comprehensions, methods of str, int, list, and dict,
and only these builtins: {builtins}. Reply with one ```python block that
holds the whole function."""

NO_EXPERIENCE_PROMPT = "{contract}\n\nWrite the function."

SUCCESS_PROMPT = """{contract}

Below are runs where the assistant did this job itself and the user
accepted the booking: each request, every tool call with its result, and
the verdict. Write the function so that it would have found what these
runs found.

{episodes}"""

TRANSITION_PROMPT = """{contract}

Below are all of the assistant's runs at this job, accepted and rejected:
each request, every tool call with its result, including rejected bookings
and errors, and the user's verdict with its reason. Write the function so
it finds what the accepted runs found and avoids every mistake the
rejected runs made.

{episodes}"""

LESSON_PROMPT = """A calendar assistant used the function below for this request:
{request}

It went wrong: {reason}

The calendars it read:
{calendars}

```python
{source}
```

In one or two sentences, state the lesson: what the function gets wrong
about calendars like these. Name the general rule, not this request."""

REVISE_PROMPT = """{contract}

The current function:

```python
{source}
```

Lessons learned from requests where it went wrong:
{lessons}

Rewrite the function so it follows every lesson."""

SYNTHESIS_PROMPT = """{contract}

Below are runs where the assistant did this job itself and the user
accepted the booking: each request, every tool call with its result, and
the verdict. Write the function so that it would have found what these
runs found. It will then be executed on these same requests, and you will
see where it goes wrong.

{episodes}"""

REPAIR_PROMPT = """{contract}

The current function:

```python
{source}
```

Executed on the requests the assistant has seen, it went wrong on these:
{failures}

Fix the function so these are accepted, without breaking the others."""


# ----------------------------------------------------------------- the writer

@dataclass
class Writer:
    """The model a mechanism writes with. Its settings are part of the mechanism."""

    model: str = "gpt-5.4-mini"
    params: dict[str, Any] = field(default_factory=lambda: {"reasoning_effort": "medium"})
    # Writing from every run is a long prompt and a long think; the default
    # two minutes is too short for it.
    timeout: int = 600
    calls: int = 0
    # Token counts per call, so a mechanism's learning cost can be priced from a replay too.
    usage: list[dict[str, int]] = field(default_factory=list)

    async def ask(self, prompt: str, *, effort: str | None = None) -> str:
        # A mechanism may write some calls with less reasoning; chapter 7 counts that as cost.
        params = self.params if effort is None else {**self.params, "reasoning_effort": effort}
        self.calls += 1
        response = await helix_acompletion(model=self.model, messages=[{"role": "user", "content": prompt}],
                                           timeout=self.timeout, num_retries=2, **params)
        self.usage.append(_usage_of(response) or {})
        return response.choices[0].message.content or ""

    async def write(self, prompt: str, contract: Contract, *, effort: str | None = None) -> str:
        """A function that passes the guard, asking once more if the first does not."""
        reply = await self.ask(prompt, effort=effort)
        source = extract_source(reply)
        try:
            check_source(source, contract.name)
            return source
        except UnsafeSkill as exc:
            retry = f"{prompt}\n\nYour last reply was refused: {exc}. Reply again with one ```python block."
            source = extract_source(await self.ask(retry, effort=effort))
            check_source(source, contract.name)
            return source


def extract_source(reply: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", reply, flags=re.S)
    return (match.group(1) if match else reply).strip() + "\n"


def contract_text(contract: Contract) -> str:
    return CONTRACT_PROMPT.format(signature=contract.signature, description=contract.description,
                                  api=contract.api, returns=contract.returns,
                                  builtins=", ".join(sorted(SAFE_BUILTINS)))


# ----------------------------------------------------------------- mechanisms

# Try a function on one request: (reward, the user's reason, what it read).
Act = Callable[[str, str], Awaitable[tuple[float, str, str]]]


@dataclass
class LearnedMemory:
    mechanism: str
    source: str
    writer_calls: int
    notes: list[str] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=dict)
    usage: list[dict[str, int]] = field(default_factory=list)


class NoExperience:
    name = "no_experience"

    async def learn(self, episodes: list[Episode], act: Act, writer: Writer, contract: Contract) -> LearnedMemory:
        source = await writer.write(NO_EXPERIENCE_PROMPT.format(contract=contract_text(contract)), contract)
        return LearnedMemory(self.name, source, writer.calls)


class SuccessDistillation:
    name = "success_distillation"

    async def learn(self, episodes: list[Episode], act: Act, writer: Writer, contract: Contract) -> LearnedMemory:
        shown = "\n\n".join(render_episode(e) for e in episodes if e.success)
        source = await writer.write(SUCCESS_PROMPT.format(contract=contract_text(contract), episodes=shown), contract)
        return LearnedMemory(self.name, source, writer.calls)


class TransitionDistillation:
    name = "transition_distillation"

    async def learn(self, episodes: list[Episode], act: Act, writer: Writer, contract: Contract) -> LearnedMemory:
        shown = "\n\n".join(render_episode(e) for e in episodes)
        source = await writer.write(TRANSITION_PROMPT.format(contract=contract_text(contract), episodes=shown),
                                    contract)
        return LearnedMemory(self.name, source, writer.calls)


def _stream(episodes: list[Episode]) -> list[str]:
    """The distinct requests in the order they were experienced."""
    return list(dict.fromkeys(e.task_id for e in episodes))


class ValueLearning:
    """Chapter 5's reinforcement learning, applied to remembered functions."""

    name = "value_learning"

    def __init__(self, epochs: int = 2) -> None:
        self.epochs = epochs

    async def learn(self, episodes: list[Episode], act: Act, writer: Writer, contract: Contract) -> LearnedMemory:
        # One candidate per request the agent solved, written from that run alone.
        candidates: dict[str, str] = {}
        for episode in episodes:
            if episode.success and episode.task_id not in candidates:
                prompt = SUCCESS_PROMPT.format(contract=contract_text(contract), episodes=render_episode(episode))
                try:
                    candidates[episode.task_id] = await writer.write(prompt, contract)
                except UnsafeSkill:
                    continue
        if not candidates:
            raise ValueError("value learning needs at least one successful run")
        values = {key: INITIAL_VALUE for key in candidates}
        notes = []
        for _ in range(self.epochs):
            for task_id in _stream(episodes):
                # Recall the highest-valued candidate; ties go to the earliest written.
                chosen = max(values, key=lambda k: values[k])
                reward, reason, _ = await act(candidates[chosen], task_id)
                values[chosen] += ALPHA * (reward - values[chosen])
                notes.append(f"{task_id}: used {chosen}, reward {reward:g}")
        best = max(values, key=lambda k: values[k])
        return LearnedMemory(self.name, candidates[best], writer.calls, notes,
                             {k: round(v, 4) for k, v in values.items()})


class Reflection:
    name = "reflection"

    def __init__(self, max_revisions: int = 5) -> None:
        # Each revision is a long write; past a handful, lessons start to
        # repeat and the function drifts rather than improves.
        self.max_revisions = max_revisions

    async def learn(self, episodes: list[Episode], act: Act, writer: Writer, contract: Contract) -> LearnedMemory:
        first = next((e for e in episodes if e.success), episodes[0])
        prompt = SUCCESS_PROMPT.format(contract=contract_text(contract), episodes=render_episode(first))
        source = await writer.write(prompt, contract)
        lessons: list[str] = []
        for task_id in _stream(episodes):
            if len(lessons) >= self.max_revisions:
                break
            reward, reason, calendars = await act(source, task_id)
            if reward >= 1.0:
                continue
            request = next(e.request for e in episodes if e.task_id == task_id)
            lesson = (await writer.ask(LESSON_PROMPT.format(request=request, reason=reason, calendars=calendars,
                                                            source=source))).strip()
            lessons.append(lesson)
            revised = REVISE_PROMPT.format(contract=contract_text(contract), source=source,
                                           lessons="\n".join(f"- {item}" for item in lessons))
            try:
                source = await writer.write(revised, contract)
            except UnsafeSkill:
                continue
        return LearnedMemory(self.name, source, writer.calls, lessons)


class SkillSynthesis:
    name = "skill_synthesis"

    def __init__(self, rounds: int = 3) -> None:
        self.rounds = rounds

    async def learn(self, episodes: list[Episode], act: Act, writer: Writer, contract: Contract) -> LearnedMemory:
        shown = "\n\n".join(render_episode(e) for e in episodes if e.success)
        source = await writer.write(SYNTHESIS_PROMPT.format(contract=contract_text(contract), episodes=shown),
                                    contract)
        notes = []
        for _ in range(self.rounds):
            failures = []
            for task_id in _stream(episodes):
                reward, reason, _ = await act(source, task_id)
                if reward < 1.0:
                    request = next(e.request for e in episodes if e.task_id == task_id)
                    failures.append(f"- {request} {reason}")
            notes.append(f"{len(failures)} rejected")
            if not failures:
                break
            prompt = REPAIR_PROMPT.format(contract=contract_text(contract), source=source,
                                          failures="\n".join(failures))
            try:
                source = await writer.write(prompt, contract)
            except UnsafeSkill:
                break
        return LearnedMemory(self.name, source, writer.calls, notes)


# The search proposes them in this order: chapter 5's RL first.
MECHANISMS = {m.name: m for m in (ValueLearning(), Reflection(), SuccessDistillation(), TransitionDistillation(),
                                  SkillSynthesis())}
CONTROL = NoExperience()
