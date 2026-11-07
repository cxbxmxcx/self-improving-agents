"""Per-user metacognition: one learned procedure set per person.

Chapter 6's conventions are personal: Dana Park's note about buffers is one
user's world, not every user's. `PerUser` keeps a separate Metacognition
(its own store, trust states, detector windows, and learning) for each user,
chosen per run from `RunConfig(memory_context=MemoryContext(user_id=...))`.

    meta = PerUser(lambda user: Metacognition([...], verifier=check, store=f"runs/meta/{user}"))
    agent = Agent(..., metacognition=meta)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class PerUser:
    def __init__(self, factory: Callable[[str], Any], *, default_user: str = "anonymous") -> None:
        self.factory = factory
        self.default_user = default_user
        self.by_user: dict[str, Any] = {}

    def for_run(self, config: Any) -> Any:
        context = getattr(config, "memory_context", None)
        user = getattr(context, "user_id", None) or self.default_user
        if user not in self.by_user:
            self.by_user[user] = self.factory(user)
        return self.by_user[user]

    async def wait_for_learning(self) -> None:
        for meta in list(self.by_user.values()):
            await meta.wait_for_learning()
