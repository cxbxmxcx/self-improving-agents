"""LLM call wrapper that respects the token-aware rate budget.

Replaces `await litellm.acompletion(...)` at every Helix call site. Behavior:

  1. Estimate input tokens from the messages.
  2. await the rate budget for that (provider, model) before issuing.
  3. Call litellm.acompletion with sensible retry/timeout defaults.
  4. Charge the bucket with actual usage from the response.
  5. Return the response.

This is the single chokepoint through which every LLM call in Helix flows.
SearchBudget enforcement, prompt caching markers, and rate-limit handling
all live here or just upstream.
"""

from __future__ import annotations

from typing import Any

import litellm

from helix.rate_budget import (
    estimate_input_tokens,
    get_budget,
    infer_provider,
)


DEFAULT_NUM_RETRIES = 5
DEFAULT_TIMEOUT = 120


async def acompletion(
    *,
    model: str,
    messages: list[dict],
    num_retries: int = DEFAULT_NUM_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> Any:
    """Drop-in replacement for litellm.acompletion with rate-budget protection.

    Every keyword argument passes through to LiteLLM unchanged except for
    model/messages/num_retries/timeout, which we set defaults on.
    """
    provider = infer_provider(model)
    budget = get_budget(provider, model)

    estimated = estimate_input_tokens(messages)
    await budget.acquire(estimated_tokens=estimated)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            num_retries=num_retries,
            timeout=timeout,
            **kwargs,
        )
    finally:
        # Always reconcile, even on exception, so a failed call still appears
        # in the window (LiteLLM may have already consumed some tokens).
        pass

    # Reconcile with actual usage if the response carries it.
    usage = getattr(response, "usage", None)
    actual = None
    if usage is not None:
        # Anthropic returns prompt_tokens (input). OpenAI uses prompt_tokens too.
        actual = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    if actual is not None and actual > 0:
        budget.charge(actual_tokens=int(actual))

    return response
