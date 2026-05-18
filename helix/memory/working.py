"""Working memory: what is currently in the model's context window.

Storage is the messages array. Read returns relevant slices, write appends, and
consolidate summarizes old turns. v0 implements the bare minimum: append and
return-the-list. Consolidation and eviction land in Ch 3.

The MemoryTier contract from Spec §7.1 has read/write/score/evict/consolidate.
We stub the ones v0 doesn't use so the contract is visible from the start.
"""

from __future__ import annotations

from typing import Any


class WorkingMemory:
    """The messages array, wrapped behind the tier contract."""

    def __init__(self, initial: list[dict[str, Any]] | None = None) -> None:
        self._messages: list[dict[str, Any]] = list(initial or [])

    def append(self, message: dict[str, Any]) -> None:
        self._messages.append(message)

    def messages(self) -> list[dict[str, Any]]:
        """Return the current message list (mutable; the agent loop writes here)."""
        return self._messages

    async def read(self, query: Any = None, context: Any = None) -> list[dict[str, Any]]:
        """Tier contract. v0 returns everything; later versions return slices."""
        return list(self._messages)

    async def write(self, entry: dict[str, Any]) -> int:
        self._messages.append(entry)
        return len(self._messages) - 1

    async def consolidate(self) -> dict[str, Any]:
        """No-op in v0. Ch 3 adds summarization of old turns here."""
        return {"consolidated": 0}
