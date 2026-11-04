"""Memory subsystem.

Four tiers behind a uniform contract: working, episodic, semantic, procedural.
SPEC section 7. Each tier exposes the same operations and differs in storage,
indexing, and consolidation policy.
"""

from helix.memory.base import (
    ConsolidationReport,
    EntryId,
    EvictionPolicy,
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    MemoryTier,
    Scope,
    ScopeKey,
)
from helix.memory.consolidation import (
    ConsolidationPolicy,
    consolidate_episodic,
    schedule_consolidation,
    stop_consolidation,
)
from helix.memory.episodic import EpisodicMemory
from helix.memory.hooks import MemoryHooks
from helix.memory.operators import Operator, add, downvote, edit, evict_below, upvote
from helix.memory.procedural import ProceduralMemory
from helix.memory.semantic import SemanticMemory
from helix.memory.working import WorkingMemory
from helix.memory.learning import MemoryLearner

__all__ = [
    "ConsolidationReport",
    "EntryId",
    "EvictionPolicy",
    "MemoryContext",
    "MemoryEntry",
    "MemoryQuery",
    "MemoryTier",
    "Scope",
    "ScopeKey",
    "EpisodicMemory",
    "ProceduralMemory",
    "SemanticMemory",
    "WorkingMemory",
    "MemoryHooks",
    "MemoryLearner",
    "Operator",
    "add",
    "upvote",
    "downvote",
    "edit",
    "evict_below",
    "ConsolidationPolicy",
    "consolidate_episodic",
    "schedule_consolidation",
    "stop_consolidation",
]
