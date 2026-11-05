"""function_tool: helix-identical schemas, context injection, sync and async, failures."""

from __future__ import annotations

import pytest
from helix.tools import tool as helix_tool
from pydantic import ValidationError

from metacog import RunContextWrapper, function_tool


async def book_time(title: str, day: str, start: str, minutes: int, attendees: list[str] | None = None) -> dict:
    return {}


@function_tool(description_override="Create an event.")
async def book_time_with_context(ctx: RunContextWrapper[dict], title: str, day: str, start: str, minutes: int,
                                 attendees: list[str] | None = None) -> dict:
    return {"user": ctx.context["user"], "title": title, "attendees": attendees}


def test_schema_matches_helix_and_hides_the_context():
    ours = function_tool(book_time, description_override="Create an event.")
    theirs = helix_tool(description="Create an event.")(book_time)
    assert ours.to_openai_schema() == theirs.to_openai_schema()
    # Same schema with a context parameter, apart from the function's own name.
    with_ctx = book_time_with_context.params_json_schema
    assert with_ctx["properties"] == theirs.args_model.model_json_schema()["properties"]
    assert "ctx" not in with_ctx["properties"] and book_time_with_context.takes_context


async def test_context_reaches_the_function_and_arguments_are_validated():
    ctx = RunContextWrapper(context={"user": "maya"})
    out = await book_time_with_context.invoke(ctx, title="Sync", day="wed", start="10:00", minutes="30")
    assert out == {"user": "maya", "title": "Sync", "attendees": None}
    with pytest.raises(ValidationError):
        await book_time_with_context.invoke(ctx, title="Sync", day="wed", start="10:00")  # minutes missing


async def test_sync_functions_and_docstring_descriptions():
    @function_tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add.description == "Add two numbers." and add.name == "add"
    assert await add.invoke(None, a=2, b=3) == 5


async def test_failure_error_function_replaces_the_exception():
    @function_tool(failure_error_function=lambda ctx, exc: f"could not: {exc}")
    def fails() -> str:
        raise RuntimeError("calendar offline")

    @function_tool
    def raises() -> str:
        raise RuntimeError("calendar offline")

    assert await fails.invoke(None) == "could not: calendar offline"
    with pytest.raises(RuntimeError):
        await raises.invoke(None)  # helix's loop turns this into {"error": ...} for the model
