"""Token-aware rate budget tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from helix.observability.bus import EventBus
from helix.observability.events import RateLimitWaited
from helix.rate_budget import (
    TokenBucket,
    estimate_input_tokens,
    get_budget,
    infer_provider,
    reset_budgets_for_tests,
)


@pytest.mark.asyncio
async def test_bucket_allows_call_under_cap():
    bucket = TokenBucket(tokens_per_minute=10_000, requests_per_minute=10)
    waited = await bucket.acquire(estimated_tokens=1_000)
    assert waited == 0.0
    assert bucket.status()["tokens_used_window"] == 1_000


@pytest.mark.asyncio
async def test_bucket_blocks_at_token_cap():
    bucket = TokenBucket(tokens_per_minute=1_000, requests_per_minute=100)
    await bucket.acquire(estimated_tokens=900)
    # Shorten the window so the test doesn't take 60s real time.
    bucket.WINDOW_SEC = 0.5

    t0 = time.monotonic()
    # This second call needs 200 tokens; cap is 1000, used 900 in window.
    # Must wait until the first reservation expires.
    waited = await bucket.acquire(estimated_tokens=200)
    elapsed = time.monotonic() - t0
    assert waited > 0.0
    assert elapsed >= 0.4  # roughly waited for window expiry


@pytest.mark.asyncio
async def test_bucket_blocks_at_rpm_cap():
    bucket = TokenBucket(tokens_per_minute=1_000_000, requests_per_minute=2)
    bucket.WINDOW_SEC = 0.5

    await bucket.acquire(estimated_tokens=100)
    await bucket.acquire(estimated_tokens=100)

    t0 = time.monotonic()
    waited = await bucket.acquire(estimated_tokens=100)
    elapsed = time.monotonic() - t0
    assert waited > 0.0
    assert elapsed >= 0.4


def test_bucket_charge_updates_last_reservation():
    bucket = TokenBucket(tokens_per_minute=10_000, requests_per_minute=10)

    async def go():
        await bucket.acquire(estimated_tokens=1_000)
        bucket.charge(actual_tokens=1_500)
        assert bucket.status()["tokens_used_window"] == 1_500

    asyncio.run(go())


@pytest.mark.asyncio
async def test_bucket_emits_rate_limit_waited_event():
    """When the bucket blocks, it should publish a RateLimitWaited event."""
    bus = EventBus()
    # Replace the global bus singleton with ours for this test.
    from helix.observability import bus as bus_module
    original = bus_module._default_bus
    bus_module._default_bus = bus
    try:
        received: list[RateLimitWaited] = []
        bus.subscribe("rate_limit_waited", lambda e: received.append(e))

        bucket = TokenBucket(
            tokens_per_minute=1_000,
            requests_per_minute=100,
            provider="test",
            model="test-model",
        )
        bucket.WINDOW_SEC = 0.3

        await bucket.acquire(estimated_tokens=900)
        await bucket.acquire(estimated_tokens=200)  # will block

        assert len(received) >= 1
        assert received[0].provider == "test"
        assert received[0].model == "test-model"
        assert received[0].wait_sec > 0
    finally:
        bus_module._default_bus = original


def test_estimate_input_tokens_from_string_messages():
    messages = [
        {"role": "system", "content": "x" * 400},   # ~100 tokens
        {"role": "user", "content": "y" * 800},     # ~200 tokens
    ]
    est = estimate_input_tokens(messages)
    assert 280 <= est <= 320


def test_estimate_input_tokens_from_content_blocks():
    """Anthropic cacheable content uses a list of blocks, not a string."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "x" * 400}]},
        {"role": "user", "content": "y" * 800},
    ]
    est = estimate_input_tokens(messages)
    assert 280 <= est <= 320


def test_infer_provider_recognizes_known_prefixes():
    assert infer_provider("claude-haiku-4-5") == "anthropic"
    assert infer_provider("anthropic/claude-3-opus") == "anthropic"
    assert infer_provider("gpt-4o-mini") == "openai"
    assert infer_provider("gemini-2.0-flash") == "google"
    assert infer_provider("unknown-model") == "unknown"


@pytest.mark.asyncio
async def test_get_budget_returns_singleton_per_model():
    reset_budgets_for_tests()
    b1 = get_budget("anthropic", "claude-haiku-4-5")
    b2 = get_budget("anthropic", "claude-haiku-4-5")
    b3 = get_budget("anthropic", "claude-sonnet-4-6")
    assert b1 is b2
    assert b1 is not b3
