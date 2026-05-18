"""Search implementations.

Each module here is one Search family. They all conform to the Search Protocol
defined in helix.search.base. Hill-climb, SPO, and GEPA ship in Stratum 2;
MIPROv2, AFlow, evolutionary, MemRL, HyperAgents land later strata.
"""

from helix.search.base import (
    Search,
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.search.strategy_chain import StrategyChain

__all__ = [
    "Search",
    "SearchBudget",
    "SearchCostModel",
    "SearchKind",
    "Variant",
    "StrategyChain",
]
