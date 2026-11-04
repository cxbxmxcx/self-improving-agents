"""The online metacognition loop's parts: recorder, controller, detector, gate.

The offline search found a learning mechanism; this module runs it while the
assistant serves requests. The recorder writes every run to episodic memory.
The controller picks a reasoning setting by lookup, from the floors listing
6.2 measured: the cheapest setting that clears the bar with the memory it
has. The detector watches runs that used memory and quarantines the memory
when they start failing. The learner, the mechanism the search kept, writes
a new function from recent runs, and the gate admits it only when it clears
the same bar on held-back cases.

None of these parts changes the agent loop. They sit in the harness around
it and decide two things per request: which memory the agent may recall and
which reasoning setting it runs at.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chapters.ch05.personal.encoder import embed
from chapters.ch06.assistant.learning import USER_SCOPE, act
from helix.eval.effort import ADEQUATE, EFFORTS, EffortSweep
from helix.memory.base import MemoryContext, MemoryEntry, MemoryQuery
from helix.memory.episodic import EpisodicMemory
from helix.memory.learning_mechanisms import Episode


class Recorder:
    """Every run, written to episodic memory in the order it happened."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.tier = EpisodicMemory(path, embedding_fn=embed)
        self.count = 0

    async def record(self, episode: Episode, *, effort: str, used_memory: bool) -> None:
        await self.tier.write(MemoryEntry(
            id=f"{episode.task_id}-{self.count:03d}", scope_key=USER_SCOPE,
            content={**episode.to_content(), "order": self.count, "effort": effort, "used_memory": used_memory},
            metadata={"created_by": "agent", "kind": "episode"}))
        self.count += 1

    async def since(self, order: int) -> list[Episode]:
        """Runs recorded at or after a point, oldest first, as a learner reads them."""
        entries = await self.tier.read(MemoryQuery(text="", k=10_000), MemoryContext(user_id="maya"))
        recent = sorted((e for e in entries if e.content.get("order", 0) >= order),
                        key=lambda e: e.content["order"])
        return [Episode.from_content(e.content) for e in recent]

    def close(self) -> None:
        self.tier.close()


@dataclass
class Controller:
    """Reasoning by lookup: the setting each memory state was measured to need."""

    table: dict[str, str]  # "memory" or "no_memory" -> reasoning setting

    def effort(self, memory_trusted: bool) -> str:
        return self.table["memory" if memory_trusted else "no_memory"]

    @classmethod
    def from_floors(cls, path: Path, *, bar: float = ADEQUATE) -> Controller:
        """Build the table from listing 6.2's sweeps.

        Each arm gets the cheapest setting that clears the bar. An arm that
        never clears it gets the setting with the best record, since spending
        less would only fail more often.
        """
        sweeps = {d["arm"]: EffortSweep.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))}

        def setting(arm: str) -> str:
            sweep = sweeps[arm]
            measured = [e for e in EFFORTS if e in sweep.cells]
            return sweep.floor(bar) or max(measured, key=lambda e: (sweep.cells[e].rate, EFFORTS.index(e)))

        return cls({"memory": setting("bootstrapped_memory"), "no_memory": setting("no_memory")})


@dataclass
class Detector:
    """Notice when a memory that passed the gate stops working.

    It watches only runs that used the memory. One rejection can be noise;
    `limit` rejections inside the last `window` runs mean the memory no
    longer matches the world it serves.
    """

    window: int = 4
    limit: int = 2
    recent: deque = field(default_factory=deque)

    def observe(self, accepted: bool) -> bool:
        self.recent.append(accepted)
        while len(self.recent) > self.window:
            self.recent.popleft()
        return sum(not ok for ok in self.recent) >= self.limit

    def reset(self) -> None:
        self.recent.clear()


@dataclass
class Gate:
    """Admit a memory only when it clears the bar on cases it never learned from."""

    cases: list[dict[str, Any]]
    bar: float = ADEQUATE

    async def check(self, source: str) -> tuple[int, int, bool]:
        accepted = 0
        for task in self.cases:
            reward, _, _ = await act(source, task["id"])
            accepted += int(reward)
        total = len(self.cases)
        return accepted, total, accepted / total >= self.bar - 1e-9
