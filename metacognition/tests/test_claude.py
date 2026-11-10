"""helix.metacognition.claude: Claude models through Anthropic's SDK, behind helix's OpenAI-style call (no API key)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from helix.metacognition import claude, recording
from helix.metacognition.ladders import ladder_for
from helix.metacognition.pricing import PRICES
from helix.metacognition.recording import Recording

TOOL = {"type": "function", "function": {"name": "read_trip", "description": "Read a trip.",
                                         "parameters": {"type": "object", "properties": {"trip_id": {"type": "string"}},
                                                        "required": ["trip_id"]}}}


def block(kind: str, **fields):
    return SimpleNamespace(type=kind, **fields)


def reply(*content, thinking: int = 0, stop: str = "end_turn"):
    usage = SimpleNamespace(input_tokens=900, output_tokens=120, cache_read_input_tokens=100,
                            cache_creation_input_tokens=0,
                            output_tokens_details=SimpleNamespace(thinking_tokens=thinking))
    return SimpleNamespace(content=list(content), usage=usage, stop_reason=stop)


class FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.message


class FakeClient:
    """Anthropic's async client as far as the adapter uses it: messages.stream(...)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.messages = self

    def stream(self, **params):
        self.requests.append(params)
        return FakeStream(self.replies.pop(0))


@pytest.fixture
def fake(monkeypatch):
    def install(*replies):
        client = FakeClient(replies)
        monkeypatch.setattr(claude, "_client", lambda: client)
        claude._turns.clear()
        return client

    return install


def test_each_model_turns_thinking_off_its_own_way():
    assert claude.thinking_params("claude-haiku-5-5", "none") == {"thinking": {"type": "disabled"},
                                                                  "output_config": {"effort": "low"}}
    assert claude.thinking_params("claude-sonnet-5-5", "none") == {"thinking": {"type": "between_tools"},
                                                                   "output_config": {"effort": "low"}}
    assert claude.thinking_params("anthropic/claude-sonnet-5-5", "high") == {"thinking": {"type": "adaptive"},
                                                                             "output_config": {"effort": "high"}}
    assert claude.thinking_params("claude-haiku-5-5", None) == {}
    with pytest.raises(ValueError, match="can't turn thinking off"):
        claude.thinking_params("claude-opus-5-5", "none")
    # The ladders follow: a model that can't turn thinking off starts at low.
    assert ladder_for("claude-sonnet-5-5").rungs[0] == "none"
    assert ladder_for("claude-opus-5-5").rungs[0] == "low"
    assert ladder_for("claude-haiku-5-5").params("none") == {"reasoning_effort": "none"}


async def test_a_tool_loop_sends_each_turn_back_as_the_api_sent_it(fake):
    thought = block("thinking", thinking="", signature="sig-1")
    client = fake(reply(thought, block("text", text="Reading the trip."),
                        block("tool_use", id="toolu_1", name="read_trip", input={"trip_id": "E201"}),
                        thinking=30, stop="tool_use"),
                  reply(block("text", text="Claimed 412.50 USD.")))
    system = {"role": "system", "content": "You file expense claims."}
    user = {"role": "user", "content": "File trip E201."}
    first = await claude.acompletion(model="claude-haiku-5-5", messages=[system, user], tools=[TOOL],
                                     reasoning_effort="low", num_retries=5, timeout=120)
    sent = client.requests[0]
    assert sent["system"] == "You file expense claims." and sent["messages"] == [user]
    assert sent["tools"] == [{"name": "read_trip", "description": "Read a trip.",
                              "input_schema": TOOL["function"]["parameters"]}]
    assert (sent["thinking"], sent["output_config"], sent["max_tokens"]) == ({"type": "adaptive"}, {"effort": "low"},
                                                                             claude.MAX_TOKENS)
    message = first.choices[0].message
    assert message.content == "Reading the trip." and first.choices[0].finish_reason == "tool_use"
    assert [(c.id, c.function.name, json.loads(c.function.arguments)) for c in message.tool_calls] == [
        ("toolu_1", "read_trip", {"trip_id": "E201"})]
    usage = first.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.completion_tokens_details.reasoning_tokens,
            usage.prompt_tokens_details.cached_tokens) == (1000, 120, 30, 100)

    # helix sends the turn back in OpenAI form; the API gets the original blocks, thinking included.
    assistant = {"role": "assistant", "content": "Reading the trip.",
                 "tool_calls": [{"id": "toolu_1", "type": "function",
                                 "function": {"name": "read_trip", "arguments": '{"trip_id": "E201"}'}}]}
    result = {"role": "tool", "tool_call_id": "toolu_1", "content": '{"trip": "E201"}'}
    second = await claude.acompletion(model="claude-haiku-5-5", messages=[system, user, assistant, result],
                                      tools=[TOOL], reasoning_effort="low")
    turns = client.requests[1]["messages"]
    assert turns[1]["role"] == "assistant" and turns[1]["content"][0] is thought
    assert turns[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                                     "content": '{"trip": "E201"}'}]}
    assert second.choices[0].message.content == "Claimed 412.50 USD." and second.choices[0].message.tool_calls is None


async def test_the_results_of_one_turn_go_back_in_one_user_message(fake):
    client = fake(reply(block("text", text="done")))
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "read_trip", "arguments": '{"trip_id": "E1"}'}},
        {"id": "b", "type": "function", "function": {"name": "read_trip", "arguments": '{"trip_id": "E2"}'}}]}
    await claude.acompletion(model="claude-sonnet-5-5", reasoning_effort="none", tools=[TOOL], tool_choice="required",
                             messages=[{"role": "user", "content": "Both trips."}, assistant,
                                       {"role": "tool", "tool_call_id": "a", "content": "one"},
                                       {"role": "tool", "tool_call_id": "b", "content": "two"}])
    sent = client.requests[0]
    # A turn this adapter didn't produce is rebuilt from its OpenAI form.
    assert [b["type"] for b in sent["messages"][1]["content"]] == ["tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in sent["messages"][2]["content"]] == ["a", "b"]
    assert sent["tool_choice"] == {"type": "auto"}  # forced choices are rejected by Sonnet 5.5
    assert sent["thinking"] == {"type": "between_tools"}


async def test_structured_output_is_refused_rather_than_dropped(fake):
    fake()
    with pytest.raises(NotImplementedError, match="output_config.format"):
        await claude.acompletion(model="claude-haiku-5-5", messages=[{"role": "user", "content": "x"}],
                                 response_format={"type": "json_object"})


async def test_the_router_sends_claude_models_to_the_sdk_and_others_to_litellm(monkeypatch):
    calls = []

    async def fake_claude(**kwargs):
        calls.append(("claude", kwargs["model"]))
        return "from claude"

    async def fake_litellm(**kwargs):
        calls.append(("litellm", kwargs["model"]))
        return "from litellm"

    monkeypatch.setattr(claude, "acompletion", fake_claude)
    monkeypatch.setattr(recording, "_original", fake_litellm)
    assert await recording._dispatch(model="claude-haiku-5-5", messages=[]) == "from claude"
    assert await recording._dispatch(model="anthropic/claude-sonnet-5-5", messages=[]) == "from claude"
    assert await recording._dispatch(model="gpt-5.4-mini", messages=[]) == "from litellm"
    assert calls == [("claude", "claude-haiku-5-5"), ("claude", "anthropic/claude-sonnet-5-5"),
                     ("litellm", "gpt-5.4-mini")]


async def test_a_live_claude_call_is_recorded_with_its_thinking_and_priced(tmp_path, monkeypatch):
    async def fake_claude(**kwargs):
        return claude.to_response(reply(block("text", text="ok"), thinking=40), kwargs["model"])

    monkeypatch.setattr(claude, "acompletion", fake_claude)
    rec = Recording(tmp_path / "claude.jsonl", mode="live")
    response = await rec(model="claude-haiku-5-5", messages=[{"role": "user", "content": "hi"}],
                         reasoning_effort="low")
    row = rec.live_rows[0]
    assert response.choices[0].message.content == "ok"
    assert (row["model"], row["effort"], row["finish_reason"]) == ("claude-haiku-5-5", "low", "end_turn")
    assert (row["usage"]["prompt_tokens"], row["usage"]["completion_tokens"], row["usage"]["reasoning_tokens"],
            row["usage"]["cached_tokens"]) == (1000, 120, 40, 100)
    # Haiku 5.5 at $0.10 / $0.50 per million tokens, cache reads at $0.01.
    assert rec.spent(PRICES) == pytest.approx((900 * 0.10 + 100 * 0.01 + 120 * 0.50) / 1e6)
    replayed = Recording(tmp_path / "claude.jsonl")
    again = await replayed(model="claude-haiku-5-5", messages=[{"role": "user", "content": "hi"}],
                           reasoning_effort="low")
    assert again.choices[0].message.content == "ok"


def test_the_builder_model_is_priced_and_keeps_thinking_on():
    from helix.metacognition.ladders import ladder_for
    from helix.metacognition.model_settings import BUILDER_MODEL
    from helix.metacognition.pricing import PRICES, call_dollars

    assert BUILDER_MODEL == "claude-fable-5-2" and PRICES[BUILDER_MODEL] == (8.00, 40.00, 0.20)
    assert call_dollars(BUILDER_MODEL, {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}, PRICES) == 48.0
    assert "none" not in ladder_for(BUILDER_MODEL).rungs
    assert claude.thinking_params(BUILDER_MODEL, "medium") == {"thinking": {"type": "adaptive"},
                                                              "output_config": {"effort": "medium"}}


def test_a_declined_request_comes_back_empty_and_is_kept():
    from types import SimpleNamespace as NS

    usage = NS(input_tokens=10, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    message = NS(content=[], stop_reason="refusal", stop_details=NS(category="cyber", explanation="declined"), usage=usage)
    before = len(claude.refusals)
    reply = claude.to_response(message, "claude-fable-5-2")
    assert reply.choices[0].message.content == "" and reply.choices[0].finish_reason == "refusal"
    assert claude.refusals[before:] == [{"model": "claude-fable-5-2", "category": "cyber", "explanation": "declined"}]
