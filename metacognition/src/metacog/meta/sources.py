"""Learning mechanisms held as source (chapter 7), so a search can rewrite them and the loop can run any of them.

Chapter 6's mechanisms are classes (metacog.meta.mechanisms). Chapter 7 holds the same mechanisms as source:
top-level constants, the writer prompts among them, and one entry point, `async def learn(lab)`. The lab
(`helix.memory.mechanism_runtime`) is a mechanism's only capability. It meters writer calls, practice calls
and time, and its guard refuses imports, private names, bare `except` and `str.format` before anything runs.
Like chapter 6's code-skill guard it catches honest mistakes and is not a security sandbox.

`seed_source(name, prompts)` is one of chapter 7's seeds (copied verbatim in `seeds/`) with a prompt pack's
texts in its prompt constants and the code otherwise chapter 7's. `SourceLearner` runs any mechanism source
as a learner, so a seed, a hand-written mechanism and one a search designed all serve the loop the same way.

    meta = Metacognition([...], learners={"parse_invoice": SourceLearner.seed("skill_synthesis")})
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import weakref
from pathlib import Path
from typing import Any

from helix.memory.learning_mechanisms import Contract, Episode, LearnedMemory, Writer
from helix.memory.mechanism_runtime import Lab, MechanismBudget, check_mechanism, run_mechanism

from metacog.meta.mechanisms import Mechanism, contract_text
from metacog.meta.prompts import GENERIC, PACKS, PromptPack

SEEDS_DIR = Path(__file__).parent / "seeds"
# Chapter 7's seeds, in the order chapter 6's search proposed them, and the control every learner must beat.
SEEDS = ("value_learning", "reflection", "success_distillation", "transition_distillation", "skill_synthesis")
CONTROL = "no_experience"
# Each prompt constant a seed holds, and the prompt-pack field its text comes from.
PROMPT_FIELDS = {"NO_EXPERIENCE_PROMPT": "no_experience", "SUCCESS_PROMPT": "success",
                 "TRANSITION_PROMPT": "transition", "LESSON_PROMPT": "lesson", "REVISE_PROMPT": "revise",
                 "SYNTHESIS_PROMPT": "synthesis", "REPAIR_PROMPT": "repair"}
# The seeds fill a lesson's observations as {calendars}, chapter 6's name, where the domain-neutral pack says
# {observations}. Renaming the placeholder keeps the seed's code chapter 7's and the prompt's text the pack's.
RENAMED = {"{observations}": "{calendars}"}

# run_mechanism traces the thread it runs in to enforce a mechanism's deadline, so one mechanism runs at a
# time per event loop (background learning could otherwise start two).
_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


def _one_at_a_time() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _LOCKS.get(loop)
    if lock is None:
        lock = _LOCKS[loop] = asyncio.Lock()
    return lock


def _literal(text: str) -> str:
    """A Python string literal for `text`: triple-quoted when that reads back exactly, else repr."""
    candidate = f'"""{text}"""'
    if '"""' not in text and not text.endswith('"') and "\\" not in text:
        try:
            if ast.literal_eval(candidate) == text:
                return candidate
        except (SyntaxError, ValueError):
            pass
    return repr(text)


def prompt_constants(source: str) -> dict[str, str]:
    """The prompt constants a mechanism's source assigns at top level, by name."""
    found = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in PROMPT_FIELDS and isinstance(node.value, ast.Constant)):
            found[node.targets[0].id] = node.value.value
    return found


def with_prompts(source: str, prompts: PromptPack | str) -> str:
    """`source` with its prompt constants set to a pack's texts; every other line is left as it was."""
    pack = PACKS[prompts] if isinstance(prompts, str) else prompts
    lines = source.splitlines(keepends=True)
    spans = []
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in PROMPT_FIELDS):
            text = getattr(pack, PROMPT_FIELDS[node.targets[0].id])
            for old, new in RENAMED.items():
                text = text.replace(old, new)
            spans.append((node.lineno - 1, node.end_lineno, f"{node.targets[0].id} = {_literal(text)}\n"))
    for start, end, replacement in sorted(spans, reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines)


def seed_source(name: str, prompts: PromptPack | str = GENERIC) -> str:
    """Chapter 7's seed `name` ("value_learning", ..., or the control "no_experience") with `prompts`' texts."""
    if name not in (*SEEDS, CONTROL):
        raise ValueError(f"unknown seed {name!r}; choose from {[*SEEDS, CONTROL]}")
    return with_prompts((SEEDS_DIR / f"{name}.py").read_text(encoding="utf-8"), prompts)


def source_id(source: str) -> str:
    """A short, stable id for a mechanism's source."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


class SourceLearner(Mechanism):
    """A learner from mechanism source, run through helix's metered lab (chapter 7).

    `prompts` words the contract the mechanism reads as `lab.contract`; its writer prompts are its own
    constants (`seed_source` puts a pack's texts there). `budget` limits writer calls, practice calls and
    time: a mechanism that runs out stops, and the loop logs the attempt as failed, with what it spent.
    """

    def __init__(self, source: str, name: str = "mechanism", prompts: PromptPack | str = GENERIC,
                 budget: MechanismBudget | None = None) -> None:
        super().__init__(PACKS[prompts] if isinstance(prompts, str) else prompts)
        check_mechanism(source)  # refuse unsafe source when it is declared, not when it first learns
        self.source = source
        self.name = name
        self.budget = budget

    @classmethod
    def seed(cls, name: str, prompts: PromptPack | str = GENERIC, budget: MechanismBudget | None = None
             ) -> SourceLearner:
        """Chapter 7's seed `name` as a learner."""
        return cls(seed_source(name, prompts), name, prompts, budget)

    @property
    def id(self) -> str:
        return f"{self.name}:{source_id(self.source)}"

    async def learn(self, episodes: list[Episode], act: Any, writer: Writer, contract: Contract) -> LearnedMemory:
        lab = Lab(episodes, act, contract, writer, budget=self.budget)
        lab.contract = contract_text(contract, self.prompts)
        async with _one_at_a_time():
            return await run_mechanism(self.source, lab, name=self.name)
