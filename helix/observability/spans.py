"""OpenTelemetry span helpers.

Thin wrapper that emits OTel spans for major events. Falls back to a no-op
tracer if no exporter is configured. The framework does not ship its own
dashboard; readers plug in Phoenix, Langfuse, Logfire, LangSmith, Arize, or
Braintrust as OTel consumers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from opentelemetry import trace

_TRACER = trace.get_tracer("helix")


@asynccontextmanager
async def span(name: str, **attributes: Any):
    """Open an OTel span. Attributes appear as span attrs in any consumer."""
    with _TRACER.start_as_current_span(name) as current:
        for key, value in attributes.items():
            try:
                current.set_attribute(key, value)
            except Exception:
                current.set_attribute(key, str(value))
        yield current
