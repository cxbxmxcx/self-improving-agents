"""Lifecycle hooks, shaped like the OpenAI Agents SDK's RunHooks and AgentHooks.

Subclass and override what you need. The Runner maps each method onto a helix
hook point: agent start and end on SESSION_START and SESSION_END, LLM start
and end on PRE_MODEL and POST_MODEL, tool start and end on PRE_TOOL and
POST_TOOL. `input_items` are chat messages, not Responses API items.
"""

from __future__ import annotations

from typing import Any, Generic

from helix.metacognition.run_context import RunContextWrapper, TContext


class RunHooks(Generic[TContext]):
    """Callbacks for every agent in a run."""

    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Any) -> None:
        pass

    async def on_agent_end(self, context: RunContextWrapper[TContext], agent: Any, output: Any) -> None:
        pass

    async def on_llm_start(self, context: RunContextWrapper[TContext], agent: Any, system_prompt: str | None,
                           input_items: list[dict[str, Any]]) -> None:
        pass

    async def on_llm_end(self, context: RunContextWrapper[TContext], agent: Any, response: Any) -> None:
        pass

    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Any, tool: Any) -> None:
        pass

    async def on_tool_end(self, context: RunContextWrapper[TContext], agent: Any, tool: Any, result: Any) -> None:
        pass

    async def on_handoff(self, context: RunContextWrapper[TContext], from_agent: Any, to_agent: Any) -> None:
        pass


class AgentHooks(Generic[TContext]):
    """Callbacks for one agent, set as `Agent(hooks=...)`."""

    async def on_start(self, context: RunContextWrapper[TContext], agent: Any) -> None:
        pass

    async def on_end(self, context: RunContextWrapper[TContext], agent: Any, output: Any) -> None:
        pass

    async def on_llm_start(self, context: RunContextWrapper[TContext], agent: Any, system_prompt: str | None,
                           input_items: list[dict[str, Any]]) -> None:
        pass

    async def on_llm_end(self, context: RunContextWrapper[TContext], agent: Any, response: Any) -> None:
        pass

    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Any, tool: Any) -> None:
        pass

    async def on_tool_end(self, context: RunContextWrapper[TContext], agent: Any, tool: Any, result: Any) -> None:
        pass

    async def on_handoff(self, context: RunContextWrapper[TContext], agent: Any, source: Any) -> None:
        pass
