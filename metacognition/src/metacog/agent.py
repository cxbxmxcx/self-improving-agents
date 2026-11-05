"""The Agent: a configuration, shaped like the OpenAI Agents SDK's Agent dataclass.

An Agent holds no run state. The Runner compiles it onto a fresh helix Agent
for each run, because what a run can call (its tools, bound to its context)
belongs to that run.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Generic

from metacog.guardrail import InputGuardrail, OutputGuardrail
from metacog.lifecycle import AgentHooks
from metacog.model_settings import ModelSettings
from metacog.run_context import RunContextWrapper, TContext
from metacog.tool import FunctionTool, function_tool


@dataclass
class Agent(Generic[TContext]):
    name: str
    # A string, or a function of (context, agent) returning one, sync or async.
    instructions: str | Callable[..., Any] | None = None
    handoff_description: str | None = None
    tools: list[Any] = field(default_factory=list)
    model: str | None = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    input_guardrails: list[InputGuardrail] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail] = field(default_factory=list)
    output_type: type[Any] | None = None
    hooks: AgentHooks[TContext] | None = None
    # A metacog.meta.Metacognition: learned procedures and reasoning by lookup (metacog's own field).
    metacognition: Any = None

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """A copy with the given fields replaced. Lists not replaced are shared, as in the Agents SDK."""
        return replace(self, **kwargs)

    async def get_system_prompt(self, context: RunContextWrapper[TContext]) -> str | None:
        if callable(self.instructions):
            prompt = self.instructions(context, self)
            return await prompt if inspect.isawaitable(prompt) else prompt
        return self.instructions

    def as_tool(self, tool_name: str | None = None, tool_description: str | None = None,
                custom_output_extractor: Callable[[Any], Any] | None = None, max_turns: int | None = None,
                run_config: Any = None) -> FunctionTool:
        """This agent as a tool another agent can call with an `input` string (SPEC §11.9: sub-agents
        are tools). The sub-run shares the caller's context and recording."""
        agent = self

        async def run_agent(context: RunContextWrapper[Any], input: str) -> Any:
            from metacog.run import DEFAULT_MAX_TURNS, Runner

            result = await Runner.run(agent, input, context=context.context, max_turns=max_turns or DEFAULT_MAX_TURNS,
                                      run_config=run_config)
            if custom_output_extractor is not None:
                extracted = custom_output_extractor(result)
                return await extracted if inspect.isawaitable(extracted) else extracted
            return str(result.final_output)

        name = tool_name or re.sub(r"[^a-zA-Z0-9_-]+", "_", self.name).strip("_").lower()
        return function_tool(run_agent, name_override=name,
                             description_override=tool_description or self.handoff_description or "")
