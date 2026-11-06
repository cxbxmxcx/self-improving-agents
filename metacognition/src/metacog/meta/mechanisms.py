"""Learning mechanisms: turn an agent's runs into a procedure's source.

These are chapter 6's five mechanisms and its control, ported from
`helix.memory.learning_mechanisms` line for line, with one change: the
prompts come from a PromptPack instead of module constants, so an agent in
any domain can learn. With the CALENDAR pack they send exactly the book's
prompts in exactly the book's order, which the parity tests check.

Every mechanism has the same inputs: the episodes, `act` (try a function on
one request: source, case id -> reward, reason, observations), a helix
Writer, and the procedure's Contract.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from helix.memory.code_skills import SAFE_BUILTINS, UnsafeSkill
from helix.memory.learning_mechanisms import ALPHA, INITIAL_VALUE, Contract, Episode, LearnedMemory, Writer
from helix.memory.learning_mechanisms import render_episode as render

from metacog.meta.prompts import GENERIC, PACKS, PromptPack

Act = Callable[[str, str], Awaitable[tuple[float, str, str]]]


def contract_text(contract: Contract, prompts: PromptPack) -> str:
    return prompts.contract.format(signature=contract.signature, description=contract.description,
                                   api=contract.api, returns=contract.returns,
                                   builtins=", ".join(sorted(SAFE_BUILTINS)))


def stream(episodes: list[Episode]) -> list[str]:
    """The distinct requests in the order they were experienced."""
    return list(dict.fromkeys(e.task_id for e in episodes))


class Mechanism:
    name = "mechanism"

    def __init__(self, prompts: PromptPack = GENERIC) -> None:
        self.prompts = prompts

    async def learn(self, episodes: list[Episode], act: Act, writer: Writer, contract: Contract) -> LearnedMemory:
        raise NotImplementedError


class NoExperience(Mechanism):
    """The control: write from the contract alone."""

    name = "no_experience"

    async def learn(self, episodes, act, writer, contract):
        prompt = self.prompts.no_experience.format(contract=contract_text(contract, self.prompts))
        return LearnedMemory(self.name, await writer.write(prompt, contract), writer.calls, usage=writer.usage)


class SuccessDistillation(Mechanism):
    name = "success_distillation"

    async def learn(self, episodes, act, writer, contract):
        shown = "\n\n".join(render(e) for e in episodes if e.success)
        prompt = self.prompts.success.format(contract=contract_text(contract, self.prompts), episodes=shown)
        return LearnedMemory(self.name, await writer.write(prompt, contract), writer.calls, usage=writer.usage)


class TransitionDistillation(Mechanism):
    name = "transition_distillation"

    async def learn(self, episodes, act, writer, contract):
        shown = "\n\n".join(render(e) for e in episodes)
        prompt = self.prompts.transition.format(contract=contract_text(contract, self.prompts), episodes=shown)
        return LearnedMemory(self.name, await writer.write(prompt, contract), writer.calls, usage=writer.usage)


class ValueLearning(Mechanism):
    """Chapter 5's reinforcement learning, applied to remembered functions.

    One candidate per request the agent solved, written from that run alone;
    then q <- q + ALPHA * (reward - q) as candidates are used on the stream,
    always using the highest-valued one (ties go to the earliest written).
    """

    name = "value_learning"

    def __init__(self, prompts: PromptPack = GENERIC, epochs: int = 2) -> None:
        super().__init__(prompts)
        self.epochs = epochs

    async def learn(self, episodes, act, writer, contract):
        candidates: dict[str, str] = {}
        for episode in episodes:
            if episode.success and episode.task_id not in candidates:
                prompt = self.prompts.success.format(contract=contract_text(contract, self.prompts),
                                                     episodes=render(episode))
                try:
                    candidates[episode.task_id] = await writer.write(prompt, contract)
                except UnsafeSkill:
                    continue
        if not candidates:
            raise ValueError("value learning needs at least one successful run")
        values = {key: INITIAL_VALUE for key in candidates}
        notes = []
        for _ in range(self.epochs):
            for task_id in stream(episodes):
                chosen = max(values, key=lambda k: values[k])
                reward, _reason, _ = await act(candidates[chosen], task_id)
                values[chosen] += ALPHA * (reward - values[chosen])
                notes.append(f"{task_id}: used {chosen}, reward {reward:g}")
        best = max(values, key=lambda k: values[k])
        return LearnedMemory(self.name, candidates[best], writer.calls, notes,
                             {k: round(v, 4) for k, v in values.items()}, writer.usage)


class Reflection(Mechanism):
    """Use the function on the stream, write a lesson from each failure, revise to follow every lesson."""

    name = "reflection"

    def __init__(self, prompts: PromptPack = GENERIC, max_revisions: int = 5) -> None:
        super().__init__(prompts)
        self.max_revisions = max_revisions

    async def learn(self, episodes, act, writer, contract):
        first = next((e for e in episodes if e.success), episodes[0])
        prompt = self.prompts.success.format(contract=contract_text(contract, self.prompts), episodes=render(first))
        source = await writer.write(prompt, contract)
        lessons: list[str] = []
        for task_id in stream(episodes):
            if len(lessons) >= self.max_revisions:
                break
            reward, reason, observations = await act(source, task_id)
            if reward >= 1.0:
                continue
            request = next(e.request for e in episodes if e.task_id == task_id)
            lesson = (await writer.ask(self.prompts.lesson.format(
                request=request, reason=reason, calendars=observations, observations=observations,
                source=source))).strip()
            lessons.append(lesson)
            revised = self.prompts.revise.format(contract=contract_text(contract, self.prompts), source=source,
                                                 lessons="\n".join(f"- {item}" for item in lessons))
            try:
                source = await writer.write(revised, contract)
            except UnsafeSkill:
                continue
        return LearnedMemory(self.name, source, writer.calls, lessons, usage=writer.usage)


class SkillSynthesis(Mechanism):
    """Write from the successful runs, execute on the observed requests, repair what went wrong."""

    name = "skill_synthesis"

    def __init__(self, prompts: PromptPack = GENERIC, rounds: int = 3) -> None:
        super().__init__(prompts)
        self.rounds = rounds

    async def learn(self, episodes, act, writer, contract):
        shown = "\n\n".join(render(e) for e in episodes if e.success)
        source = await writer.write(self.prompts.synthesis.format(contract=contract_text(contract, self.prompts),
                                                                  episodes=shown), contract)
        notes = []
        for _ in range(self.rounds):
            failures = []
            for task_id in stream(episodes):
                reward, reason, _ = await act(source, task_id)
                if reward < 1.0:
                    request = next(e.request for e in episodes if e.task_id == task_id)
                    failures.append(f"- {request} {reason}")
            notes.append(f"{len(failures)} rejected")
            if not failures:
                break
            prompt = self.prompts.repair.format(contract=contract_text(contract, self.prompts), source=source,
                                                failures="\n".join(failures))
            try:
                source = await writer.write(prompt, contract)
            except UnsafeSkill:
                break
        return LearnedMemory(self.name, source, writer.calls, notes, usage=writer.usage)


# The order a mechanism search proposes them in: chapter 5's RL first.
MECHANISMS: dict[str, type[Mechanism]] = {cls.name: cls for cls in (
    ValueLearning, Reflection, SuccessDistillation, TransitionDistillation, SkillSynthesis)}
CONTROL = NoExperience


def resolve_mechanism(learner: str | Mechanism, prompts: PromptPack | str = GENERIC) -> Mechanism:
    """A mechanism instance from a name ("value_learning", "no_experience", ...) or an instance."""
    if isinstance(learner, Mechanism):
        return learner
    pack = PACKS[prompts] if isinstance(prompts, str) else prompts
    if learner == NoExperience.name:
        return NoExperience(pack)
    if learner not in MECHANISMS:
        raise ValueError(f"unknown learner {learner!r}; choose from {sorted([*MECHANISMS, NoExperience.name])}")
    return MECHANISMS[learner](pack)


def spent(outcome: Any, writer: Writer) -> tuple[int, list[dict[str, Any]]]:
    """Writer calls and usage a learning attempt spent. A mechanism run in helix's lab writes through the
    lab's own metered writer and reports its spend on the learned memory, or on the MechanismStopped that
    ended it; a class mechanism reports the runtime writer's, which is the fallback."""
    calls, usage = getattr(outcome, "writer_calls", None), getattr(outcome, "usage", None)
    return (calls if calls else writer.calls), (usage if usage else writer.usage)


def writer_cost(usage: list[dict[str, Any]], model: str, prices: dict[str, tuple[float, ...]] | None) -> float:
    from metacog.pricing import call_dollars

    return sum(call_dollars(model, u, prices) or 0.0 for u in usage)
