"""Working memory: what is currently in the model's context window.

Storage is the messages array within a single agent.run() invocation. Read
returns the message list, write appends. Working memory is per-run, not
persisted across sessions; episodic memory handles cross-session retrieval.

The MemoryTier contract is implemented for uniformity, but working memory
is unique in that it has no scope (it's the live messages array) and its
read/write semantics are message-shaped rather than entry-shaped. The
agent loop holds a WorkingMemory directly via .messages()/.append() rather
than going through the tier protocol; the tier protocol methods exist for
introspection (e.g. the dashboard can iterate working memory like any tier).
"""

from __future__ import annotations

import uuid
from typing import Any

from helix.memory.base import (
    ConsolidationReport,
    EntryId,
    EvictionPolicy,
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    Scope,
    ScopeKey,
)


class WorkingMemory:
    """The messages array, wrapped behind the tier contract.

    Unlike the persistent tiers, working memory lives for one agent.run()
    and is discarded on session end (the trajectory is written to episodic
    memory at SESSION_END; working memory itself is throw-away state).
    """

    tier_name: str = "working"

    def __init__(self, initial: list[dict[str, Any]] | None = None) -> None:
        self._messages: list[dict[str, Any]] = list(initial or [])

    # ---------------- direct, message-shaped API used by the agent loop ----------------

    def append(self, message: dict[str, Any]) -> None:
        self._messages.append(message)

    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    # ---------------- MemoryTier contract ----------------

    async def read(
        self,
        query: MemoryQuery,
        context: MemoryContext,
    ) -> list[MemoryEntry]:
        """Return every message as a MemoryEntry. Working memory has no scope."""
        return [
            MemoryEntry(
                id=str(i),
                scope_key=ScopeKey(Scope.SESSION, context.session_id or "ephemeral"),
                content=msg,
            )
            for i, msg in enumerate(self._messages)
        ]

    async def write(self, entry: MemoryEntry) -> EntryId:
        """Append the entry's content (treated as a message dict)."""
        self._messages.append(entry.content)
        eid = entry.id or str(uuid.uuid4())
        return eid

    async def score(self, entry: MemoryEntry) -> float:
        return 1.0  # Working memory has no scoring

    async def evict(self, policy: EvictionPolicy) -> int:
        """Truncate the message buffer if max_entries is set."""
        if policy.max_entries is None:
            return 0
        excess = len(self._messages) - policy.max_entries
        if excess <= 0:
            return 0
        del self._messages[: excess]
        return excess

    async def consolidate(self) -> ConsolidationReport:
        """No-op in v0. Chapter 3 adds summarization of old turns here."""
        return ConsolidationReport(
            tier=self.tier_name,
            entries_before=len(self._messages),
            entries_after=len(self._messages),
            notes="working memory consolidation not yet implemented",
        )
