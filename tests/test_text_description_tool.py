"""Tests for TextDescriptionTool (SPEC §16.2.2, Ch 3)."""

from __future__ import annotations

import pytest

from helix.artifact import ArtifactKind, genesis
from helix.tools import TextDescriptionTool


def _desc(content: str = "Search the corpus for relevant passages.",
          id: str = "tool_description.retrieve") -> object:
    return genesis(id=id, kind=ArtifactKind.TOOL_DESCRIPTION, content=content)


async def _retrieve(query: str, k: int = 5) -> list[dict]:
    return [{"q": query, "k": k}]


def test_construct_reads_description_from_artifact():
    desc = _desc("Find documents.")
    tool = TextDescriptionTool(name="retrieve", fn=_retrieve, description_artifact=desc)
    assert tool.name == "retrieve"
    assert tool.description == "Find documents."


def test_args_model_from_fn_type_hints():
    tool = TextDescriptionTool(name="retrieve", fn=_retrieve, description_artifact=_desc())
    schema = tool.to_openai_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "query" in props
    assert "k" in props


@pytest.mark.asyncio
async def test_call_executes_implementation():
    tool = TextDescriptionTool(name="retrieve", fn=_retrieve, description_artifact=_desc())
    result = await tool.call(query="transformers", k=3)
    assert result == [{"q": "transformers", "k": 3}]


def test_swap_description_updates_description():
    desc_v1 = _desc("v1 description")
    tool = TextDescriptionTool(name="retrieve", fn=_retrieve, description_artifact=desc_v1)
    assert tool.description == "v1 description"

    desc_v2 = desc_v1.mutate(new_content="v2 improved description", created_by="spo")
    tool.swap_description(desc_v2)
    assert tool.description == "v2 improved description"
    # The schema reflects the new description too.
    assert tool.to_openai_schema()["function"]["description"] == "v2 improved description"


def test_artifact_refs_returns_description_ref():
    desc = _desc(id="tool_description.retrieve")
    tool = TextDescriptionTool(name="retrieve", fn=_retrieve, description_artifact=desc)
    assert tool.artifact_refs() == [("tool_description.retrieve", 1)]


def test_rejects_non_tool_description_artifact():
    bad = genesis(id="p", kind=ArtifactKind.PROMPT, content="not a description")
    with pytest.raises(ValueError, match="TOOL_DESCRIPTION"):
        TextDescriptionTool(name="retrieve", fn=_retrieve, description_artifact=bad)


def test_swap_rejects_non_tool_description_artifact():
    tool = TextDescriptionTool(name="retrieve", fn=_retrieve, description_artifact=_desc())
    bad = genesis(id="p", kind=ArtifactKind.PROMPT, content="not a description")
    with pytest.raises(ValueError, match="TOOL_DESCRIPTION"):
        tool.swap_description(bad)


def test_sync_implementation_wrapped_in_async():
    def sync_retrieve(query: str) -> str:
        return f"got {query}"

    tool = TextDescriptionTool(name="retrieve", fn=sync_retrieve, description_artifact=_desc())
    import asyncio
    assert asyncio.run(tool.call(query="x")) == "got x"
