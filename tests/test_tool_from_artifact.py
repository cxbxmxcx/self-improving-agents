"""Tests for ToolFromArtifact (SPEC §16.2.2)."""

from __future__ import annotations

import pytest

from helix.artifact import ArtifactKind, genesis, Subtype
from helix.tools import ToolFromArtifact


def _make_pair(
    code: str = (
        "async def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
    ),
    description: str = "Add two integers and return the sum.",
    code_id: str = "code.tool.add",
    desc_id: str = "prompt.tool.add.description",
):
    code_art = genesis(id=code_id, kind=ArtifactKind.CODE, content=code)
    desc_art = genesis(id=desc_id, kind=Subtype.TOOL_DESCRIPTION, content=description)
    return code_art, desc_art


def test_tool_from_artifact_infers_name_from_single_async_function():
    code_art, desc_art = _make_pair()
    tool = ToolFromArtifact(code_artifact=code_art, description_artifact=desc_art)
    assert tool.name == "add"
    assert tool.description.startswith("Add two integers")


def test_tool_from_artifact_uses_explicit_tool_name():
    code_art, desc_art = _make_pair()
    tool = ToolFromArtifact(
        code_artifact=code_art,
        description_artifact=desc_art,
        tool_name="sum_two",
        expected_callable="add",
    )
    assert tool.name == "sum_two"


@pytest.mark.asyncio
async def test_tool_from_artifact_call_executes_function():
    code_art, desc_art = _make_pair()
    tool = ToolFromArtifact(code_artifact=code_art, description_artifact=desc_art)
    result = await tool.call(a=2, b=3)
    assert result == 5


def test_tool_from_artifact_to_openai_schema_includes_args():
    code_art, desc_art = _make_pair()
    tool = ToolFromArtifact(code_artifact=code_art, description_artifact=desc_art)
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert "a" in schema["function"]["parameters"]["properties"]
    assert "b" in schema["function"]["parameters"]["properties"]


def test_tool_from_artifact_records_both_refs():
    code_art, desc_art = _make_pair()
    tool = ToolFromArtifact(code_artifact=code_art, description_artifact=desc_art)
    refs = tool.artifact_refs()
    assert ("code.tool.add", 1) in refs
    assert ("prompt.tool.add.description", 1) in refs


def test_tool_from_artifact_rejects_non_code_for_implementation():
    bad = genesis(id="bad", kind=Subtype.PROMPT, content="async def add(): pass")
    desc = genesis(id="d", kind=Subtype.TOOL_DESCRIPTION, content="x")
    with pytest.raises(ValueError, match="ArtifactKind.CODE"):
        ToolFromArtifact(code_artifact=bad, description_artifact=desc)


def test_tool_from_artifact_rejects_non_tool_description_for_description():
    code = genesis(
        id="c",
        kind=ArtifactKind.CODE,
        content="async def add(a: int, b: int) -> int:\n    return a + b\n",
    )
    bad = genesis(id="b", kind=Subtype.PROMPT, content="x")
    with pytest.raises(ValueError, match="TOOL_DESCRIPTION"):
        ToolFromArtifact(code_artifact=code, description_artifact=bad)


def test_tool_from_artifact_wraps_sync_callable_in_async():
    code = (
        "def multiply(a: int, b: int) -> int:\n"
        "    return a * b\n"
    )
    code_art = genesis(id="code.mul", kind=ArtifactKind.CODE, content=code)
    desc_art = genesis(
        id="prompt.mul.description",
        kind=Subtype.TOOL_DESCRIPTION,
        content="Multiply two integers.",
    )
    # Need expected_callable since the heuristic looks for async functions
    # first; with no async function in scope it falls back to any callable.
    tool = ToolFromArtifact(
        code_artifact=code_art,
        description_artifact=desc_art,
        expected_callable="multiply",
    )

    import asyncio
    assert asyncio.run(tool.call(a=3, b=4)) == 12


def test_tool_from_artifact_ambiguous_callable_raises():
    code = (
        "async def first(x: int) -> int:\n    return x\n"
        "async def second(x: int) -> int:\n    return x\n"
    )
    code_art = genesis(id="c.amb", kind=ArtifactKind.CODE, content=code)
    desc_art = genesis(
        id="d.amb",
        kind=Subtype.TOOL_DESCRIPTION,
        content="ambiguous",
    )
    with pytest.raises(ValueError, match="cannot infer tool name"):
        ToolFromArtifact(code_artifact=code_art, description_artifact=desc_art)
