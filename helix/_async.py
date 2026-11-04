"""Async helpers shared across Helix modules."""

from __future__ import annotations

import inspect
from typing import Any


async def resolve(value: Any) -> Any:
    """Await a callback's result only when it is awaitable, so applications
    may hand Helix either plain or async callbacks."""
    return await value if inspect.isawaitable(value) else value
