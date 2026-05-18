"""Provider-aware prompt caching helpers.

Anthropic supports ephemeral prompt caching via cache_control markers on
content blocks. When the same system prompt or rubric is sent 20 times in a
row (e.g. the reference artifact across 20 eval questions, or the judge rubric
across 20 judge calls), marking the static prefix as cacheable saves ~70-90%
on input tokens for the second-and-later calls.

LiteLLM forwards cache_control markers transparently for Anthropic. Other
providers ignore them (Anthropic-shaped markers are no-ops on OpenAI, etc.),
so applying them is provider-agnostic in practice — but we still gate on
model name so the message structure stays clean for non-Anthropic providers.

Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

from typing import Any


def is_anthropic_model(model: str) -> bool:
    """Return True if the model string routes to Anthropic via LiteLLM."""
    if not model:
        return False
    m = model.lower()
    return m.startswith("claude") or m.startswith("anthropic/")


def cacheable_system_message(
    content: str, model: str, ttl: str = "5m"
) -> dict[str, Any]:
    """Build a system message that marks `content` for ephemeral caching.

    For Anthropic models, returns the content-blocks structure with a
    cache_control marker so LiteLLM forwards the cache hint. For other
    providers, returns the simple string-form system message unchanged.

    ttl: "5m" (default, Anthropic's standard ephemeral TTL) or "1h" (extended
    cache, where supported).
    """
    if not is_anthropic_model(model):
        return {"role": "system", "content": content}

    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl and ttl != "5m":
        cache_control["ttl"] = ttl

    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": content,
                "cache_control": cache_control,
            }
        ],
    }


def cacheable_user_prefix(
    prefix_content: str,
    tail_content: str,
    model: str,
    ttl: str = "5m",
) -> dict[str, Any]:
    """Build a user message whose prefix is cached and whose tail varies.

    Useful for the judge: the rubric instruction is the cacheable prefix; the
    question + candidate answer + reference answer is the variable tail. The
    rubric never changes across the 20 judge calls in a round, so a single
    cache write earns a hit on every subsequent call.
    """
    if not is_anthropic_model(model):
        return {"role": "user", "content": prefix_content + "\n\n" + tail_content}

    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl and ttl != "5m":
        cache_control["ttl"] = ttl

    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prefix_content,
                "cache_control": cache_control,
            },
            {
                "type": "text",
                "text": tail_content,
            },
        ],
    }
