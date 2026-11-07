"""The Agents SDK surface, compiled onto helix, against a scripted model."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from metacog import (
    Agent,
    AgentHooks,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    Reasoning,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    Runner,
    SQLiteSession,
    function_tool,
    input_guardrail,
    output_guardrail,
    trace,
)
from metacog.testing import ScriptedModel, call, text


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


def weather_agent(**kwargs) -> Agent:
    return Agent(name="Assistant", instructions="You are helpful.", tools=[get_weather], model="gpt-5.4-mini",
                 **kwargs)


async def test_quickstart_shape():
    model = ScriptedModel([call("get_weather", {"city": "Tokyo"}, prompt_tokens=50, completion_tokens=5),
                           text("It is sunny in Tokyo.", prompt_tokens=80, completion_tokens=8)])
    with model.active():
        result = await Runner.run(weather_agent(), "What's the weather in Tokyo?")
    assert result.final_output == "It is sunny in Tokyo."
    assert [i.type for i in result.new_items] == ["tool_call_item", "tool_call_output_item", "message_output_item"]
    assert result.new_items[1].output == "Sunny in Tokyo"
    first, second = model.requests
    assert first["messages"] == [{"role": "system", "content": "You are helpful."},
                                 {"role": "user", "content": "What's the weather in Tokyo?"}]
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call_get_weather",
                                      "content": json.dumps("Sunny in Tokyo")}
    assert (result.usage.requests, result.usage.input_tokens, result.usage.output_tokens) == (2, 130, 13)
    assert result.cost.dollars > 0 and result.trace_id


def test_run_sync_outside_a_loop():
    with ScriptedModel([text("Hello.")]).active():
        assert Runner.run_sync(weather_agent(), "Hi").final_output == "Hello."


async def test_run_settings_override_agent_settings():
    agent = weather_agent(model_settings=ModelSettings(reasoning=Reasoning(effort="high"), temperature=0.2))
    model = ScriptedModel([text("ok")])
    with model.active():
        await Runner.run(agent, "Hi", run_config=RunConfig(model_settings=ModelSettings(reasoning=Reasoning("low"))))
    request = model.requests[0]
    assert (request["reasoning_effort"], request["temperature"]) == ("low", 0.2)


async def test_tools_receive_the_run_context():
    seen = []

    @function_tool
    def whoami(ctx: RunContextWrapper[dict]) -> str:
        """Name the user."""
        seen.append(ctx.context)
        return ctx.context["user"]

    agent = Agent(name="A", instructions="Help.", tools=[whoami])
    with ScriptedModel([call("whoami"), text("You are Maya.")]).active():
        result = await Runner.run(agent, "Who am I?", context={"user": "Maya"})
    assert seen == [{"user": "Maya"}] and result.new_items[1].output == "Maya"


async def test_tool_errors_reach_the_model():
    @function_tool
    def broken() -> str:
        """Always fails."""
        raise RuntimeError("calendar offline")

    model = ScriptedModel([call("broken"), text("Sorry.")])
    with model.active():
        await Runner.run(Agent(name="A", instructions="Help.", tools=[broken]), "Go")
    assert json.loads(model.requests[1]["messages"][-1]["content"]) == {"error": "calendar offline"}


async def test_input_guardrail_stops_before_the_model():
    @input_guardrail
    def no_homework(ctx, agent, input) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info="homework", tripwire_triggered="homework" in input)

    model = ScriptedModel([text("never sent")])
    with model.active(), pytest.raises(InputGuardrailTripwireTriggered) as caught:
        await Runner.run(weather_agent(input_guardrails=[no_homework]), "Do my homework")
    assert model.requests == [] and caught.value.guardrail_result.output.output_info == "homework"


async def test_output_guardrail_trips_on_the_final_output():
    @output_guardrail
    async def no_rain(ctx, agent, output) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered="rain" in output)

    with ScriptedModel([text("Heavy rain.")]).active(), pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(weather_agent(output_guardrails=[no_rain]), "Weather?")


class Forecast(BaseModel):
    city: str
    sunny: bool


async def test_output_type_requests_a_schema_and_parses_it():
    model = ScriptedModel([text('{"city": "Tokyo", "sunny": true}')])
    with model.active():
        result = await Runner.run(weather_agent(output_type=Forecast), "Weather in Tokyo?")
    assert result.final_output == Forecast(city="Tokyo", sunny=True)
    assert model.requests[0]["response_format"]["json_schema"]["name"] == "Forecast"
    with ScriptedModel([text("not json")]).active(), pytest.raises(ModelBehaviorError):
        await Runner.run(weather_agent(output_type=Forecast), "Weather?")


async def test_non_object_output_types_are_wrapped():
    with ScriptedModel([text('{"response": ["a", "b"]}')]).active():
        result = await Runner.run(weather_agent(output_type=list[str]), "Two letters")
    assert result.final_output == ["a", "b"]


async def test_max_turns():
    with ScriptedModel([call("get_weather", {"city": "Oslo"})] * 3).active(), pytest.raises(MaxTurnsExceeded):
        await Runner.run(weather_agent(), "Loop", max_turns=3)


async def test_sessions_carry_the_conversation():
    session = SQLiteSession("maya")
    model = ScriptedModel([text("Hi Maya."), text("You said your name is Maya.")])
    with model.active():
        await Runner.run(weather_agent(), "I'm Maya.", session=session)
        await Runner.run(weather_agent(), "What's my name?", session=session)
    assert [m["role"] for m in model.requests[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert len(await session.get_items()) == 4


async def test_input_lists_and_to_input_list_round_trip():
    model = ScriptedModel([call("get_weather", {"city": "Rome"}), text("Sunny in Rome."), text("Yes.")])
    with model.active():
        first = await Runner.run(weather_agent(), "Weather in Rome?")
        follow_up = first.to_input_list() + [{"role": "user", "content": "Is that good?"}]
        await Runner.run(weather_agent(), follow_up)
    assert [m["role"] for m in model.requests[2]["messages"]] == ["system", "user", "assistant", "tool",
                                                                  "assistant", "user"]


async def test_lifecycle_hooks_fire_in_order():
    events = []

    class Log(RunHooks):
        async def on_agent_start(self, context, agent):
            events.append("agent_start")

        async def on_llm_start(self, context, agent, system_prompt, input_items):
            events.append(f"llm_start:{system_prompt}")

        async def on_tool_start(self, context, agent, tool):
            events.append(f"tool_start:{tool.name}")

        async def on_tool_end(self, context, agent, tool, result):
            events.append(f"tool_end:{result}")

        async def on_agent_end(self, context, agent, output):
            events.append(f"agent_end:{output}")

    class AgentLog(AgentHooks):
        async def on_llm_end(self, context, agent, response):
            events.append(f"llm_end:{response.content or response.tool_calls[0]['name']}")

    with ScriptedModel([call("get_weather", {"city": "Oslo"}), text("Sunny.")]).active():
        await Runner.run(weather_agent(hooks=AgentLog()), "Oslo?", hooks=Log())
    assert events == ["agent_start", "llm_start:You are helpful.", "llm_end:get_weather", "tool_start:get_weather",
                      "tool_end:Sunny in Oslo", "llm_start:You are helpful.", "llm_end:Sunny.", "agent_end:Sunny."]


async def test_agents_as_tools_share_the_model_and_context():
    translator = Agent(name="Translator", instructions="Translate to Spanish.", handoff_description="Translates text.")
    orchestrator = Agent(name="Orchestrator", instructions="Use your tools.",
                         tools=[translator.as_tool(tool_name="translate")])
    model = ScriptedModel([call("translate", {"input": "hello"}), text("hola"), text("In Spanish: hola")])
    with model.active():
        result = await Runner.run(orchestrator, "Say hello in Spanish")
    assert result.final_output == "In Spanish: hola"
    assert model.requests[1]["messages"][0]["content"] == "Translate to Spanish."
    assert orchestrator.tools[0].description == "Translates text."


async def test_dynamic_instructions_and_traces():
    async def instructions(ctx, agent):
        return f"Help {ctx.context['user']}."

    model = ScriptedModel([text("a"), text("b")])
    with model.active(), trace("Weather workflow") as t:
        first = await Runner.run(Agent(name="A", instructions=instructions), "x", context={"user": "Maya"})
        second = await Runner.run(Agent(name="A", instructions=instructions), "y", context={"user": "Maya"})
    assert model.requests[0]["messages"][0]["content"] == "Help Maya."
    assert first.trace_id == second.trace_id == t.trace_id
