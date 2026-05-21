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
from helix.memory.episodic import EpisodicMemory
from helix.memory.procedural import ProceduralMemory
from helix.memory.semantic import SemanticMemory
from helix.memory.working import WorkingMemory

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
]
