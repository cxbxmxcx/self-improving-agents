"""Run an agent: compile it onto the helix loop, run it, and read the result back.

Each run builds a fresh helix Agent from the metacog Agent, binds the tools to
this run's context, and registers hooks for everything the Agents SDK shape
adds. The helix loop itself is never changed (SPEC §6):

- conversation history: PRE_MODEL, inserted after the system message on the first call
- input guardrails: PRE_MODEL, first call, refusing when one trips
- model settings and output_type: a PRE_MODEL RequestPatch on every call
- usage and lifecycle hooks: SESSION_START, PRE/POST_MODEL, PRE/POST_TOOL, SESSION_END
- output parsing and output guardrails: PRE_OUTPUT

helix never loops back after PRE_OUTPUT, so output that fails `output_type`
raises ModelBehaviorError rather than asking the model again.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from helix.agent import Agent as HelixAgent
from helix.artifact import Subtype, genesis
from helix.hooks import HookPoint, Refusal, RequestPatch
from helix.memory.base import MemoryContext
from helix.metacognition.exceptions import (
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    UserError,
)
from helix.metacognition.items import (
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    _arguments,
    items_from,
    messages_from,
    model_response,
    raw_responses,
)
from helix.metacognition.model_settings import DEFAULT_MODEL, ModelSettings
from helix.metacognition.pricing import PRICES, run_cost
from helix.metacognition.recording import Recording, restore_seconds, served_seconds
from helix.metacognition.recording import install as install_router
from helix.metacognition.result import RunResult
from helix.metacognition.run_context import RunContextWrapper
from helix.metacognition.tool import FunctionTool
from helix.observability import span
from helix.scaffold.common import META_KEY
from helix.trajectory import Outcome, StepKind, Trajectory
from pydantic import TypeAdapter

from helix.metacognition.agent import Agent
from helix.metacognition.guardrail import InputGuardrail, OutputGuardrail
from helix.metacognition.stream import RunItemStreamEvent, RunResultStreaming
from helix.metacognition.tracing import Trace, get_current_trace

DEFAULT_MAX_TURNS = 10


@dataclass
class RunConfig:
    """Settings for one run, layered over the agent's. The last five fields are metacog's own."""

    model: str | None = None
    model_settings: ModelSettings | None = None
    input_guardrails: list[InputGuardrail] | None = None
    output_guardrails: list[OutputGuardrail] | None = None
    workflow_name: str = "Agent workflow"
    trace_id: str | None = None
    group_id: str | None = None
    trace_metadata: dict[str, Any] | None = None
    tracing_disabled: bool = False
    # Serve this run's model calls from a recording (replay, record, or live).
    recording: Recording | None = None
    # helix's per-run cap on tool calls, beside max_turns.
    max_tool_calls: int = 20
    # Dollars per million tokens; models missing here are priced from LiteLLM's table.
    prices: dict[str, tuple[float, ...]] | None = None
    # helix memory scopes (session, user, organization) for tiers that read them.
    memory_context: MemoryContext | None = None
    # Metacognition: which case this run is (practice rebuilds it from case_data), and how the
    # agent's metacognition takes part: "online" (control, record, learn), "frozen" (recall only;
    # nothing recorded; the run's own settings decide effort), "off", or "practice" (internal).
    case_id: str | None = None
    case_data: Any = None
    metacognition_mode: str = "online"
    practice_candidate: tuple[str, str] | None = None
    # Run each function tool through `executor(tool, ctx, arguments)` instead: a replayed world.
    tool_executor: Any = None
    effort: str | None = None  # this run's reasoning, over the controller's lookup
    memory: str | None = None  # what this run's agent reads first


@dataclass
class _RunState:
    input_checked: bool = False
    input_results: list[Any] = field(default_factory=list)
    output_results: list[Any] = field(default_factory=list)
    tripped_input: Any = None
    tripped_output: Any = None
    output_error: Exception | None = None
    parsed: Any = None
    parsed_set: bool = False


def _first_call(trajectory: Trajectory) -> bool:
    return not any(s.kind == StepKind.MODEL_CALL for s in trajectory.steps)


def _split_input(input: str | list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """(history, the request): a string is the request; a list ends with it."""
    if isinstance(input, str):
        return [], input
    messages = [dict(m) for m in input]
    if not messages or messages[-1].get("role") != "user":
        raise UserError("an input list must end with a user message")
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return messages[:-1], str(content)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "agent"


def _helix_tool(tool: Any, ctx: RunContextWrapper[Any], executor: Any = None) -> Any:
    if isinstance(tool, FunctionTool):
        return tool.bind(ctx, executor)
    if hasattr(tool, "to_openai_schema") and hasattr(tool, "call"):
        return tool  # a helix Tool, used as it is
    raise UserError(f"not a tool: {tool!r}; decorate the function with @function_tool")


# ----------------------------------------------------------------- output type

def _output_schema(output_type: type[Any]) -> tuple[dict[str, Any], bool]:
    """The JSON schema for `output_type`, wrapped in an object when it is not one already."""
    schema = TypeAdapter(output_type).json_schema()
    if schema.get("type") == "object":
        return schema, False
    return {"type": "object", "properties": {"response": schema}, "required": ["response"]}, True


def _response_format(output_type: type[Any]) -> dict[str, Any]:
    schema, _ = _output_schema(output_type)
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", getattr(output_type, "__name__", "final_output"))[:64]
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": False}}


def _parse_output(output_type: type[Any], text: str) -> Any:
    _, wrapped = _output_schema(output_type)
    data = json.loads(text)
    return TypeAdapter(output_type).validate_python(data["response"] if wrapped else data)


# ----------------------------------------------------------------- hooks

def _history_hook(history: list[dict[str, Any]]):
    async def insert(messages: list[dict[str, Any]], trajectory: Trajectory, **_: Any) -> None:
        if _first_call(trajectory) and "metacog_history" not in trajectory.metadata:
            trajectory.metadata["metacog_history"] = len(history)
            at = 1 if messages and messages[0].get("role") == "system" else 0
            messages[at:at] = [dict(m) for m in history]

    return insert


def _input_guardrail_hook(guardrails: list[InputGuardrail], ctx: RunContextWrapper[Any], agent: Agent,
                          input: Any, state: _RunState):
    async def check(trajectory: Trajectory, **_: Any) -> Refusal | None:
        if state.input_checked or not _first_call(trajectory):
            return None
        state.input_checked = True
        for guardrail in guardrails:
            result = await guardrail.run(ctx, agent, input)
            state.input_results.append(result)
            if result.output.tripwire_triggered:
                state.tripped_input = result
                return Refusal(reason=f"input guardrail {guardrail.get_name()} triggered",
                               source=f"input_guardrail:{guardrail.get_name()}")
        return None

    return check


def _settings_hook(params: dict[str, Any]):
    async def patch(**_: Any) -> RequestPatch:
        return RequestPatch(params=dict(params), source="model_settings")

    return patch


_READ_ONLY = ("This request is not one to act on. You can read and answer, and you can't take any action here. State "
              "only what your tools show; if they don't show something, say you can't tell. If the request asks for "
              "something to be done, say that a person needs to handle it.")


def _effort_hook(params: dict[str, Any]):
    async def patch(**_: Any) -> RequestPatch:
        return RequestPatch(params=dict(params), source="metacognition:controller")

    return patch


def _usage_hook(ctx: RunContextWrapper[Any]):
    async def count(trajectory: Trajectory, **_: Any) -> None:
        ctx.usage.add_call(trajectory.steps[-1].payload.get("usage"))

    return count


def _output_hook(guardrails: list[OutputGuardrail], ctx: RunContextWrapper[Any], agent: Agent,
                 output_type: type[Any] | None, state: _RunState):
    async def finish(response: str, trajectory: Trajectory, **_: Any) -> Refusal | None:
        output: Any = response
        if output_type is not None:
            try:
                output = _parse_output(output_type, response)
            except (ValueError, KeyError, TypeError) as exc:  # bad JSON (a ValueError) or a schema mismatch
                state.output_error = ModelBehaviorError(
                    f"final output does not match output_type {getattr(output_type, '__name__', output_type)}: {exc}")
                return None
            # Without an output_type the run returns what the loop returns, so
            # a refusal or patch from another PRE_OUTPUT hook is respected.
            state.parsed, state.parsed_set = output, True
        for guardrail in guardrails:
            result = await guardrail.run(ctx, agent, output)
            state.output_results.append(result)
            if result.output.tripwire_triggered:
                state.tripped_output = result
                return Refusal(reason=f"output guardrail {guardrail.get_name()} triggered",
                               source=f"output_guardrail:{guardrail.get_name()}")
        return None

    return finish


def _register_lifecycle(registry: Any, run_hooks: Any, agent: Agent, ctx: RunContextWrapper[Any],
                        tools: dict[str, Any]) -> None:
    agent_hooks = agent.hooks

    async def on_start(**_: Any) -> None:
        if run_hooks is not None:
            await run_hooks.on_agent_start(ctx, agent)
        if agent_hooks is not None:
            await agent_hooks.on_start(ctx, agent)

    async def on_llm_start(messages: list[dict[str, Any]], **_: Any) -> None:
        has_system = bool(messages) and messages[0].get("role") == "system"
        system = messages[0].get("content") if has_system else None
        items = [dict(m) for m in (messages[1:] if has_system else messages)]
        if run_hooks is not None:
            await run_hooks.on_llm_start(ctx, agent, system, items)
        if agent_hooks is not None:
            await agent_hooks.on_llm_start(ctx, agent, system, items)

    async def on_llm_end(trajectory: Trajectory, **_: Any) -> None:
        response = model_response(trajectory.steps[-1])
        if run_hooks is not None:
            await run_hooks.on_llm_end(ctx, agent, response)
        if agent_hooks is not None:
            await agent_hooks.on_llm_end(ctx, agent, response)

    async def on_tool_start(tool_name: str, **_: Any) -> None:
        tool = tools.get(tool_name, tool_name)
        if run_hooks is not None:
            await run_hooks.on_tool_start(ctx, agent, tool)
        if agent_hooks is not None:
            await agent_hooks.on_tool_start(ctx, agent, tool)

    async def on_tool_end(tool_name: str, result: Any, **_: Any) -> None:
        tool = tools.get(tool_name, tool_name)
        if run_hooks is not None:
            await run_hooks.on_tool_end(ctx, agent, tool, result)
        if agent_hooks is not None:
            await agent_hooks.on_tool_end(ctx, agent, tool, result)

    async def on_end(trajectory: Trajectory, **_: Any) -> None:
        if run_hooks is not None:
            await run_hooks.on_agent_end(ctx, agent, trajectory.final_output)
        if agent_hooks is not None:
            await agent_hooks.on_end(ctx, agent, trajectory.final_output)

    registry.register(HookPoint.SESSION_START, on_start)
    registry.register(HookPoint.PRE_MODEL, on_llm_start)
    registry.register(HookPoint.POST_MODEL, on_llm_end)
    registry.register(HookPoint.PRE_TOOL, on_tool_start)
    registry.register(HookPoint.POST_TOOL, on_tool_end)
    registry.register(HookPoint.SESSION_END, on_end)


# ----------------------------------------------------------------- the run

def _register_stream(registry: Any, agent: Agent, stream: Any) -> None:
    """Push step-level items onto a streaming result as the loop produces them."""
    pending: list[str] = []

    async def model_done(trajectory: Trajectory, **_: Any) -> None:
        entry = trajectory.steps[-1].payload.get("response") or {}
        if entry.get("content"):
            stream.queue.put_nowait(RunItemStreamEvent("message_output_created",
                                                       MessageOutputItem(agent, entry["content"])))
        pending[:] = []
        for tc in entry.get("tool_calls") or []:
            fn = tc.get("function", {})
            pending.append(tc.get("id", ""))
            stream.queue.put_nowait(RunItemStreamEvent("tool_called", ToolCallItem(
                agent, fn.get("name", ""), _arguments(fn.get("arguments")), tc.get("id", ""))))

    async def tool_done(result: Any, **_: Any) -> None:
        call_id = pending.pop(0) if pending else None
        stream.queue.put_nowait(RunItemStreamEvent("tool_output", ToolCallOutputItem(agent, call_id, result)))

    registry.register(HookPoint.POST_MODEL, model_done)
    registry.register(HookPoint.POST_TOOL, tool_done)


async def _run(agent: Agent, input: str | list[dict[str, Any]], *, context: Any, max_turns: int, hooks: Any,
               config: RunConfig, session: Any, stream: Any = None) -> RunResult:
    if not isinstance(agent, Agent):
        raise UserError(f"Runner.run needs a metacog Agent, not {type(agent).__name__}")
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=context)
    model = config.model or agent.model or DEFAULT_MODEL
    settings = agent.model_settings.resolve(config.model_settings)
    instructions = await agent.get_system_prompt(ctx)
    history, request = _split_input(input)
    if session is not None:
        history = list(await session.get_items()) + history
    state = _RunState()
    output_type = None if agent.output_type in (None, str) else agent.output_type
    memory_context = config.memory_context or MemoryContext()
    # Metacognition's two effects on a run: recalled procedure tools (with one line
    # of context each, written by helix's MemoryHooks) and a reasoning setting.
    meta = agent.metacognition if config.metacognition_mode != "off" else None
    if meta is not None and hasattr(meta, "for_run"):
        meta = meta.for_run(config)  # PerUser: this user's metacognition
    plan = await meta.before_run(agent, ctx, request, config, memory_context, model=model) if meta is not None \
        else None

    # The decision tier: a trusted program gets the first try, and an action it takes ends the run with no model.
    if (plan is not None and plan.program is not None and stream is None and session is None and output_type is None
            and config.tool_executor is None
            and not (agent.input_guardrails or agent.output_guardrails or config.input_guardrails
                     or config.output_guardrails)):
        served = await meta.programs.serve(plan, agent, ctx, request, asyncio.get_running_loop())
        if served is not None:
            await meta.after_run(plan, ctx, served, trajectory=served.trajectory)
            served.metacognition = plan.report
            return served

    if plan is not None and plan.memory:
        instructions = f"{instructions}\n\n{plan.memory}"
    read_only = plan is not None and plan.read_only
    if read_only:
        instructions = f"{instructions}\n\n{_READ_ONLY}"

    helix_agent = HelixAgent(
        system_prompt=genesis(f"prompt.{_slug(agent.name)}", Subtype.PROMPT, instructions or ""),
        tools=[_helix_tool(t, ctx, config.tool_executor) for t in agent.tools if not read_only or t.read_only]
        + (plan.tools if plan is not None else []),
        model=model,
        max_iterations=max_turns,
        max_tool_calls=config.max_tool_calls,
        memory_tiers={"procedural": plan.tier} if plan is not None and plan.tier is not None else None,
    )
    if plan is not None and helix_agent.memory_hooks is not None:
        helix_agent.memory_hooks.read_only = True
        helix_agent.memory_hooks.skill_k = plan.skill_k
    registry = helix_agent.hooks
    if history:
        registry.register(HookPoint.PRE_MODEL, _history_hook(history))
    input_guardrails = [*agent.input_guardrails, *(config.input_guardrails or [])]
    if input_guardrails:
        registry.register(HookPoint.PRE_MODEL, _input_guardrail_hook(input_guardrails, ctx, agent, input, state))
    params = settings.to_params()
    if output_type is not None:
        params["response_format"] = _response_format(output_type)
    if params:
        registry.register(HookPoint.PRE_MODEL, _settings_hook(params))
    if plan is not None and plan.effort is not None:
        # Registered after the settings patch, so the controller's lookup wins (SPEC §6.2).
        registry.register(HookPoint.PRE_MODEL, _effort_hook(meta.effort_params(model, plan.effort)))
    registry.register(HookPoint.POST_MODEL, _usage_hook(ctx))
    if hooks is not None or agent.hooks is not None:
        named = {getattr(t, "name", ""): t for t in agent.tools}
        _register_lifecycle(registry, hooks, agent, ctx, named)
    output_guardrails = [*agent.output_guardrails, *(config.output_guardrails or [])]
    registry.register(HookPoint.PRE_OUTPUT, _output_hook(output_guardrails, ctx, agent, output_type, state))
    if stream is not None:
        _register_stream(registry, agent, stream)

    run_trace = get_current_trace() or Trace(config.workflow_name, group_id=config.group_id,
                                             metadata=dict(config.trace_metadata or {}),
                                             disabled=config.tracing_disabled)
    if config.trace_id and get_current_trace() is None:
        run_trace.trace_id = config.trace_id
    install_router()  # Claude models go through Anthropic's SDK even when no recording is active
    with served_seconds() as seconds:
        if run_trace.disabled:
            output, trajectory = await helix_agent.run(request, memory_context)
        else:
            async with span("helix.metacognition.run", agent=agent.name, workflow=run_trace.workflow_name,
                            trace_id=run_trace.trace_id):
                output, trajectory = await helix_agent.run(request, memory_context)
    restore_seconds(trajectory, seconds)
    if plan is not None and plan.decider_calls:  # Jev's calls from a program that handed off
        trajectory.metadata.setdefault(META_KEY, {}).setdefault("decider_calls", []).extend(plan.decider_calls)
    trajectory.metadata["trace"] = {"workflow_name": run_trace.workflow_name, "trace_id": run_trace.trace_id,
                                    "group_id": run_trace.group_id}

    if state.tripped_input is not None:
        raise InputGuardrailTripwireTriggered(state.tripped_input)
    if trajectory.outcome == Outcome.TIMED_OUT:
        if plan is not None:  # a run that never finished is a failed run, and the detector should see it
            await meta.after_run(plan, ctx, None, trajectory=trajectory, failed=f"max turns ({max_turns}) exceeded")
        raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded", trajectory)
    if state.output_error is not None:
        raise state.output_error
    if state.tripped_output is not None:
        raise OutputGuardrailTripwireTriggered(state.tripped_output)

    result = RunResult(
        input=input, new_items=items_from(trajectory, agent), raw_responses=raw_responses(trajectory),
        final_output=state.parsed if state.parsed_set else output, last_agent=agent, context_wrapper=ctx,
        trajectory=trajectory, cost=run_cost(trajectory, config.prices or PRICES),
        input_guardrail_results=state.input_results, output_guardrail_results=state.output_results,
        trace_id=run_trace.trace_id,
    )
    if plan is not None:
        await meta.after_run(plan, ctx, result, trajectory=trajectory)
        result.metacognition = plan.report
    if session is not None:
        given = [{"role": "user", "content": input}] if isinstance(input, str) else [dict(m) for m in input]
        await session.add_items(given + messages_from(trajectory))
    return result


_sync_loop: asyncio.AbstractEventLoop | None = None


class Runner:
    @classmethod
    async def run(cls, starting_agent: Agent, input: str | list[dict[str, Any]], *, context: Any = None,
                  max_turns: int = DEFAULT_MAX_TURNS, hooks: Any = None, run_config: RunConfig | None = None,
                  session: Any = None) -> RunResult:
        return await cls._run(starting_agent, input, context=context, max_turns=max_turns, hooks=hooks,
                              run_config=run_config, session=session)

    @classmethod
    async def _run(cls, starting_agent: Agent, input: str | list[dict[str, Any]], *, context: Any, max_turns: int,
                   hooks: Any, run_config: RunConfig | None, session: Any, stream: Any = None) -> RunResult:
        config = run_config or RunConfig()
        if config.recording is not None:
            with config.recording.active():
                return await _run(starting_agent, input, context=context, max_turns=max_turns, hooks=hooks,
                                  config=config, session=session, stream=stream)
        return await _run(starting_agent, input, context=context, max_turns=max_turns, hooks=hooks,
                          config=config, session=session, stream=stream)

    @classmethod
    def run_streamed(cls, starting_agent: Agent, input: str | list[dict[str, Any]], *, context: Any = None,
                     max_turns: int = DEFAULT_MAX_TURNS, hooks: Any = None, run_config: RunConfig | None = None,
                     session: Any = None) -> RunResultStreaming:
        """Start a run and return at once; iterate `stream_events()` for its items. Needs a running loop."""
        streaming = RunResultStreaming(starting_agent, input)

        async def drive() -> None:
            try:
                result = await cls._run(starting_agent, input, context=context, max_turns=max_turns, hooks=hooks,
                                        run_config=run_config, session=session, stream=streaming)
            except BaseException as exc:  # noqa: BLE001 - handed to the reader of the stream
                streaming.finish(error=exc)
                if not isinstance(exc, Exception):
                    raise
            else:
                streaming.finish(result)

        streaming.task = asyncio.get_running_loop().create_task(drive())
        return streaming

    @classmethod
    def run_sync(cls, starting_agent: Agent, input: str | list[dict[str, Any]], **kwargs: Any) -> RunResult:
        """Run outside an event loop. One loop is kept across calls, so LiteLLM's clients stay valid."""
        global _sync_loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise UserError("run_sync cannot be called inside a running event loop; use `await Runner.run(...)`")
        if _sync_loop is None or _sync_loop.is_closed():
            _sync_loop = asyncio.new_event_loop()
        return _sync_loop.run_until_complete(cls.run(starting_agent, input, **kwargs))
