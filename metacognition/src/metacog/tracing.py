"""Group runs into a trace, shaped like the OpenAI Agents SDK's `trace()`.

Runs inside `with trace(...)` share its trace id; a run outside one gets its
own from its RunConfig. Each run also opens an OpenTelemetry span through
helix, so any OTel consumer sees the run with helix's own spans nested in it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    workflow_name: str
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex}")
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False


_current: ContextVar[Trace | None] = ContextVar("metacog_trace", default=None)


@contextmanager
def trace(workflow_name: str, trace_id: str | None = None, group_id: str | None = None,
          metadata: dict[str, Any] | None = None, disabled: bool = False) -> Iterator[Trace]:
    current = Trace(workflow_name, trace_id or f"trace_{uuid.uuid4().hex}", group_id, dict(metadata or {}), disabled)
    token = _current.set(current)
    try:
        yield current
    finally:
        _current.reset(token)


def get_current_trace() -> Trace | None:
    return _current.get()


def enable_console_export() -> None:
    """Print every span (metacog's runs and helix's own) to stdout, for a first look without a backend.

    For a real backend, configure an OpenTelemetry TracerProvider with your exporter (OTLP, Phoenix,
    Langfuse, ...) before running agents; metacog and helix only emit spans.
    """
    from opentelemetry import trace as otel
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    otel.set_tracer_provider(provider)
