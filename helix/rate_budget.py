"""Token-aware rate budget.

Sliding-window token bucket per (provider, model). Callers ask the bucket
for permission before issuing an LLM call; the bucket sleeps the caller
until granting it would not exceed the cap over the last 60 seconds.

This is the real fix for provider rate-limit errors. LiteLLM's num_retries
backs off briefly on 429s, but when the failure mode is structural
("every call for the next minute will exceed your tier limit"), brief
backoffs just retry into the wall. The token bucket self-throttles before
the wall is hit.

Usage:
    from helix.rate_budget import get_budget
    budget = get_budget("anthropic", "claude-haiku-4-5")
    await budget.acquire(estimated_tokens=8000)
    response = await litellm.acompletion(...)
    budget.charge(actual_tokens=response.usage.total_tokens)

The platform's helix.llm_call.acompletion helper does this automatically;
direct litellm.acompletion callers don't get the protection.

Process-wide singletons via get_budget(); tests construct their own
TokenBucket instances for isolation.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Per-tier default caps. Conservative defaults for Anthropic tier 1, which
# is what most readers will be on. Override by constructing your own bucket.
# ---------------------------------------------------------------------------

DEFAULT_TOKENS_PER_MINUTE = {
    "anthropic": 400_000,   # tier 1 nominal cap is 450K; leave 50K headroom
    "openai": 800_000,
    "google": 1_000_000,
}

DEFAULT_RPM = {
    "anthropic": 50,
    "openai": 500,
    "google": 200,
}


@dataclass
class _Reservation:
    tokens: int
    at: float


class TokenBucket:
    """Sliding-60s budget over input tokens and requests-per-minute.

    Maintains two deques of (tokens, timestamp) and (1, timestamp). On
    acquire, pops entries older than 60s, then checks whether adding the
    new reservation would exceed the cap. If yes, sleeps until enough
    expires; if no, records and returns.

    `charge(actual)` updates the most recent reservation to the actual
    token count returned by the API. If actual exceeds the estimate, the
    overage is recorded so subsequent calls see the true load.
    """

    WINDOW_SEC = 60.0

    def __init__(
        self,
        tokens_per_minute: int = 400_000,
        requests_per_minute: int = 50,
        provider: str = "unknown",
        model: str = "unknown",
    ) -> None:
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        self.provider = provider
        self.model = model
        self._reservations: deque[_Reservation] = deque()
        self._lock = asyncio.Lock()
        self._waiters_blocked = 0  # diagnostic counter

    # ---------------- internals ----------------

    def _purge_expired(self, now: float) -> None:
        cutoff = now - self.WINDOW_SEC
        while self._reservations and self._reservations[0].at < cutoff:
            self._reservations.popleft()

    def _tokens_in_window(self, now: float) -> int:
        self._purge_expired(now)
        return sum(r.tokens for r in self._reservations)

    def _requests_in_window(self, now: float) -> int:
        self._purge_expired(now)
        return len(self._reservations)

    # ---------------- public API ----------------

    async def acquire(self, estimated_tokens: int) -> float:
        """Block until adding `estimated_tokens` won't exceed the cap.

        Returns the number of seconds the caller had to wait. The reservation
        is recorded inside; the caller settles up with charge() after the LLM
        call returns its actual usage.
        """
        if estimated_tokens > self.tokens_per_minute:
            # Single call exceeds the entire per-minute budget; can't help.
            estimated_tokens = self.tokens_per_minute  # record at cap

        wait_total = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                tokens_used = self._tokens_in_window(now)
                requests_used = self._requests_in_window(now)
                if (
                    tokens_used + estimated_tokens <= self.tokens_per_minute
                    and requests_used + 1 <= self.requests_per_minute
                ):
                    # Record the reservation and return.
                    self._reservations.append(_Reservation(tokens=estimated_tokens, at=now))
                    return wait_total

                # Wait until the oldest reservation expires, plus a small
                # cushion to avoid thrashing on the boundary.
                if not self._reservations:
                    # Nothing to wait on but we still failed the check; the
                    # estimate must be huge. Sleep a second and re-check.
                    sleep_for = 1.0
                else:
                    oldest = self._reservations[0].at
                    sleep_for = max(0.1, (oldest + self.WINDOW_SEC) - now + 0.05)
                self._waiters_blocked += 1
                # Optional: emit a RateLimitWaited event (best-effort, never crash).
                try:
                    from helix.observability.bus import get_bus
                    from helix.observability.events import RateLimitWaited
                    await get_bus().publish(
                        RateLimitWaited(
                            provider=self.provider,
                            model=self.model,
                            wait_sec=sleep_for,
                            reason=(
                                "tokens_cap" if tokens_used + estimated_tokens > self.tokens_per_minute
                                else "rpm_cap"
                            ),
                        )
                    )
                except Exception:
                    pass
                await asyncio.sleep(sleep_for)
                wait_total += sleep_for

    def charge(self, actual_tokens: int) -> None:
        """Reconcile the most recent reservation with actual usage.

        Called after the LLM response returns. If actual usage differs from
        the estimate, replace the last reservation's count so the window
        reflects truth.
        """
        if not self._reservations:
            return
        self._reservations[-1].tokens = actual_tokens

    def status(self) -> dict:
        """Diagnostic snapshot of the bucket's current load."""
        now = time.monotonic()
        return {
            "provider": self.provider,
            "model": self.model,
            "tokens_used_window": self._tokens_in_window(now),
            "tokens_cap": self.tokens_per_minute,
            "requests_used_window": self._requests_in_window(now),
            "requests_cap": self.requests_per_minute,
            "reservations": len(self._reservations),
            "waiters_blocked_total": self._waiters_blocked,
        }


# ---------------------------------------------------------------------------
# Per-(provider, model) singletons
# ---------------------------------------------------------------------------

_buckets: dict[tuple[str, str], TokenBucket] = {}


def get_budget(provider: str, model: str) -> TokenBucket:
    """Return the process-wide TokenBucket for this (provider, model)."""
    key = (provider, model)
    if key not in _buckets:
        _buckets[key] = TokenBucket(
            tokens_per_minute=DEFAULT_TOKENS_PER_MINUTE.get(provider, 400_000),
            requests_per_minute=DEFAULT_RPM.get(provider, 50),
            provider=provider,
            model=model,
        )
    return _buckets[key]


def reset_budgets_for_tests() -> None:
    """Drop all singletons; tests can isolate."""
    _buckets.clear()


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

def infer_provider(model: str) -> str:
    """Best-effort provider name from a LiteLLM-compatible model string."""
    if not model:
        return "unknown"
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic/"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("openai/") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("gemini") or m.startswith("google/"):
        return "google"
    if m.startswith("cohere/"):
        return "cohere"
    if m.startswith("groq/"):
        return "groq"
    if m.startswith("mistral"):
        return "mistral"
    return "unknown"


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def estimate_input_tokens(messages: list[dict]) -> int:
    """Rough estimate of input tokens for a messages payload.

    Uses a 4-chars-per-token heuristic across all string content. Not exact
    (real tokenization differs by provider), but good enough for budgeting:
    if the estimate is too low, charge() corrects it on the next call.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total_chars += len(block["text"])
    return max(1, total_chars // 4)
