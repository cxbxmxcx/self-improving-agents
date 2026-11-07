"""Dollars per million tokens, pinned so reported costs do not move with LiteLLM releases.

Each entry is (input, output, cached input) as the provider publishes it.
Models missing here are priced from LiteLLM's table by helix's cost helpers.
"""

from __future__ import annotations

from helix.signals.cost import call_dollars, run_cost

PRICES: dict[str, tuple[float, ...]] = {
    # The chapter 6 study's model and prices.
    "gpt-5.4-mini": (0.75, 4.50, 0.075),
}

__all__ = ["PRICES", "call_dollars", "run_cost"]
