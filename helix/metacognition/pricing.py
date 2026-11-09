"""Dollars per million tokens, pinned so reported costs do not move with LiteLLM releases.

Each entry is (input, output, cached input) as the provider publishes it.
Models missing here are priced from LiteLLM's table by helix's cost helpers.
"""

from __future__ import annotations

from helix.signals.cost import call_dollars, run_cost

PRICES: dict[str, tuple[float, ...]] = {
    # The chapter 6 study's model and prices.
    "gpt-5.4-mini": (0.75, 4.50, 0.075),
    # D4's Claude models, at Anthropic's first-party prices. Haiku 5.5's rise to $0.50 / $2.50 for
    # prompts over 100K tokens, which no study sends; its cache-read price is taken as a tenth of
    # input, as Sonnet's is.
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}

__all__ = ["PRICES", "call_dollars", "run_cost"]
