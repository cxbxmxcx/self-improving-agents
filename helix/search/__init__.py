"""Search implementations.

Each module here is one Search family. They all conform to the Search Protocol
defined in helix.search.base. Hill-climb, SPO, GEPA, and DGM ship with
Chapters 2 to 4; MemRL arrives in Chapter 5 and the HyperAgents-style
meta-procedure in Chapter 8.
"""

from helix.search.base import (
    Search,
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.search.composed import ComposedSearch
from helix.search.group_relative import GroupRelativeSearch, InsightSampler, SplitSampler
from helix.search.memrl import MemRLSearch, ieu
from helix.search.strategy_chain import StrategyChain

__all__ = [
    "Search",
    "SearchBudget",
    "SearchCostModel",
    "SearchKind",
    "Variant",
    "ComposedSearch",
    "StrategyChain",
    "MemRLSearch",
    "ieu",
    "GroupRelativeSearch",
    "InsightSampler",
    "SplitSampler",
]
