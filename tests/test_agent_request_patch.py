"""Loop plumbing for chapter 6: request patches, model-call accounting, and
cancellation reasons. SPEC §6.2 grants each of these rights; none adds a
branch to the agent loop."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from helix.agent import Agent
from helix.artifact import ArtifactKind, Subtype, genesis
from helix.hooks import HookPoint, RequestPatch
from helix.tools import tool
from helix.trajectory import StepKind


def _reply(content: str = "", *, call: str | None = None, usage=None):
    calls = None if call is None else [SimpleNamespace(
        id="call-1", type="function", function=SimpleNamespace(name=call, arguments="{}"))]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
                           usage=usage)


def _agent(**kwargs) -> Agent:
    return Agent(system_prompt=genesis("p.sys", Subtype.PROMPT, "You help with a calendar."),
                 model="fast-model", **kwargs)


def _model_calls(trajectory):
    return [s.payload for s in trajectory.steps if s.kind == StepKind.MODEL_CALL]


async def test_request_patch_switches_the_model_and_params_for_that_call():
    agent = _agent()

    async def route(**_):
        return RequestPatch(model="deliberate-model", params={"reasoning_effort": "high"}, source="router")

    agent.hooks.register(HookPoint.PRE_MODEL, route)
    completion = AsyncMock(return_value=_reply("Booked."))
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Book a writing block.")
    assert completion.await_args.kwargs["model"] == "deliberate-model"
    assert completion.await_args.kwargs["reasoning_effort"] == "high"
    step = _model_calls(trajectory)[0]
    assert (step["model"], step["params"]) == ("deliberate-model", {"reasoning_effort": "high"})


async def test_later_request_patches_override_earlier_ones_and_the_agent_keeps_its_model():
    agent = _agent(temperature=0.0)

    async def first(**_):
        return RequestPatch(model="model-a", params={"reasoning_effort": "low"})

    async def second(**_):
        return RequestPatch(params={"reasoning_effort": "high"})

    agent.hooks.register(HookPoint.PRE_MODEL, first)
    agent.hooks.register(HookPoint.PRE_MODEL, second)
    completion = AsyncMock(return_value=_reply("Done."))
    with patch("helix.agent.helix_acompletion", completion):
        await agent.run("Plan three meetings before Friday.")
    kwargs = completion.await_args.kwargs
    assert (kwargs["model"], kwargs["reasoning_effort"], kwargs["temperature"]) == ("model-a", "high", 0.0)
    assert agent.model == "fast-model"


async def test_model_call_steps_record_tokens_and_seconds():
    agent = _agent()
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=5)
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Done.", usage=usage))):
        _, trajectory = await agent.run("Book a writing block.")
    step = _model_calls(trajectory)[0]
    assert step["model"] == "fast-model"
    assert step["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    assert step["seconds"] >= 0.0
    json.dumps(step)  # the trajectory must stay serializable for the archive


async def test_model_call_steps_leave_usage_empty_when_the_provider_reports_none():
    agent = _agent()
    replies = [_reply("Done.", usage=None), _reply("Done.", usage=MagicMock())]
    for reply in replies:
        with patch("helix.agent.helix_acompletion", AsyncMock(return_value=reply)):
            _, trajectory = await agent.run("Book a writing block.")
        assert _model_calls(trajectory)[0]["usage"] is None


def _assistant_with_send_tool():
    sent: list[str] = []

    @tool(description="Send the drafted reply for the user.")
    async def send_reply() -> dict:
        sent.append("sent")
        return {"sent": True}

    return _agent(tools=[send_reply]), sent


async def test_cancelled_tool_call_returns_the_monitors_reason():
    agent, sent = _assistant_with_send_tool()

    async def monitor(tool_name, **_):
        return ("cancel", "needs the user's confirmation") if tool_name == "send_reply" else None

    agent.hooks.register(HookPoint.PRE_TOOL, monitor)
    completion = AsyncMock(side_effect=[_reply(call="send_reply"), _reply("I need your confirmation first.")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Reply to Dana that Thursday works.")
    assert sent == []
    results = [s.payload["result"] for s in trajectory.steps if s.kind == StepKind.TOOL_RESULT]
    assert results == [{"cancelled": True, "reason": "needs the user's confirmation"}]
    tool_message = [m for m in completion.await_args_list[1].kwargs["messages"] if m["role"] == "tool"][0]
    assert "needs the user's confirmation" in tool_message["content"]


async def test_bare_cancel_still_returns_a_plain_cancellation():
    agent, sent = _assistant_with_send_tool()

    async def monitor(**_):
        return "cancel"

    agent.hooks.register(HookPoint.PRE_TOOL, monitor)
    completion = AsyncMock(side_effect=[_reply(call="send_reply"), _reply("Not sent.")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Reply to Dana.")
    results = [s.payload["result"] for s in trajectory.steps if s.kind == StepKind.TOOL_RESULT]
    assert (sent, results) == ([], [{"cancelled": True}])


def test_reflector_is_a_text_subtype_at_layer_one():
    reflector = genesis("refl", Subtype.REFLECTOR, "Name the one change that would have avoided the rejection.")
    assert (reflector.kind, reflector.subtype, reflector.layer) == (ArtifactKind.TEXT, Subtype.REFLECTOR, 1)
