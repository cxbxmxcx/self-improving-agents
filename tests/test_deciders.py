"""Decision models: one set of typed questions, any provider, the same answers (chapter 6)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from helix.deciders import (
    Answer,
    Choice,
    LLMDecider,
    OpenAIDecisionsDecider,
    Probability,
    RecordingDecider,
    ReplayDecider,
    Score,
    TypeSafeDecider,
    decider_from_name,
    question_from_dict,
    question_key,
    question_to_dict,
    score_position,
)

QUESTIONS = {
    "consequence": Choice("What does this call change for other people?",
                          {"none": "Only the user's own calendar", "sends": "A message sent for the user"}),
    "ambiguous": Probability("The request names a person matching more than one contact"),
    "progress": Score("How far along is the task?", ["Not started", "Partly done", "Done"]),
}


def transport(reply: dict, sent: list, statuses: list[int] | None = None) -> httpx.MockTransport:
    """A fake provider that records each request and answers after any error statuses."""
    def handle(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), "auth": request.headers["authorization"],
                     "body": json.loads(request.content)})
        if statuses:
            return httpx.Response(statuses.pop(0), json={"error": "busy"})
        return httpx.Response(200, json=reply)
    return httpx.MockTransport(handle)


JEV_REPLY = {
    "model": "jev-1.13.0",
    "usage": {"input_tokens": 400, "output_tokens": 30},
    "answers": {
        "consequence": {"type": "choice", "choice": "sends", "confidence": 0.91,
                        "probabilities": {"none": 0.05, "sends": 0.95}},
        "ambiguous": {"type": "noul", "noul": 0.82},
        "progress": {"type": "score", "score": 1.5, "confidence": 0.6,
                     "legend": {"0": "Not started", "1": "Partly done", "2": "Done"},
                     "probabilities": {"1": 0.5, "2": 0.5}},
    },
}


async def test_jev_adapter_maps_choice_score_and_probability_to_the_systemone_wire_format():
    sent: list = []
    decider = TypeSafeDecider(api_key="test-key", transport=transport(JEV_REPLY, sent))
    answers = await decider.ask("Send the reply to Dana", QUESTIONS)
    request = sent[0]
    assert request["url"] == "https://api.typesafe.ai/v1/systemone"
    assert request["auth"] == "Bearer test-key"
    body = request["body"]
    assert body["model"] == "jev-latest"
    assert body["questions"]["ambiguous"] == {"type": "noul", "instructions": QUESTIONS["ambiguous"].instructions}
    assert body["questions"]["progress"]["criteria"] == ["Not started", "Partly done", "Done"]
    assert (answers["consequence"].value, answers["consequence"].confidence) == ("sends", 0.91)
    assert (answers["ambiguous"].kind, answers["ambiguous"].value) == ("probability", 0.82)
    assert answers["progress"].value == 1.5
    assert answers["progress"].position == pytest.approx(0.75)


async def test_openai_decisions_adapter_sends_a_named_question_list_and_returns_the_same_answers():
    sent: list = []
    reply = {"model": "gpt-6-luna", "usage": {"input_tokens": 250}, "answers": [
        {"type": "choice", "name": "consequence", "choice": "sends", "confidence": 0.9,
         "probabilities": [{"value": "none", "probability": 0.1}, {"value": "sends", "probability": 0.9}]},
        {"type": "predicate", "name": "ambiguous", "probability": 0.8},
        {"type": "score", "name": "progress", "score": 1.0, "confidence": 0.7,
         "probabilities": [{"value": "Partly done", "probability": 1.0}]},
    ]}
    decider = OpenAIDecisionsDecider(api_key="test-key", transport=transport(reply, sent))
    answers = await decider.ask({"request": "Send the reply to Dana"}, QUESTIONS)
    body = sent[0]["body"]
    assert sent[0]["url"] == "https://api.openai.com/v1/decisions"
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == '{"request": "Send the reply to Dana"}'
    assert [q["type"] for q in body["questions"]] == ["choice", "predicate", "score"]
    assert body["questions"][2]["levels"] == [{"label": "Not started"}, {"label": "Partly done"}, {"label": "Done"}]
    assert answers["consequence"].probabilities == {"none": 0.1, "sends": 0.9}
    assert (answers["ambiguous"].kind, answers["ambiguous"].value) == ("probability", 0.8)
    assert answers["progress"].position == pytest.approx(0.5)
    assert decider.name == "openai/gpt-6-luna"


async def test_hosted_deciders_retry_busy_responses_with_backoff():
    sent: list = []
    reply = {"model": "jev-1.13.0", "answers": {"ambiguous": {"type": "noul", "noul": 0.3}}}
    decider = TypeSafeDecider(api_key="k", transport=transport(reply, sent, statuses=[429, 529]))
    with mock.patch("helix.deciders.base.asyncio.sleep") as sleep:
        answers = await decider.ask("x", {"ambiguous": QUESTIONS["ambiguous"]})
    assert len(sent) == 3
    assert sleep.await_count == 2
    assert answers["ambiguous"].value == 0.3


async def test_usage_counts_calls_tokens_dollars_and_the_answering_version():
    sent: list = []
    decider = TypeSafeDecider(api_key="k", transport=transport(JEV_REPLY, sent))
    await decider.ask("one", QUESTIONS)
    await decider.ask("two", QUESTIONS)
    usage = decider.usage()
    assert (usage.calls, usage.input_tokens, usage.version) == (2, 800, "jev-1.13.0")
    assert usage.dollars == pytest.approx(800 * 42 / 1e9)
    assert usage.seconds >= 0.0
    assert usage.as_cost().tokens == 800


def test_missing_api_key_names_the_variable_to_set():
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY or JEV_API_KEY"):
            TypeSafeDecider()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIDecisionsDecider()


async def test_jev_key_may_be_stored_as_jev_api_key():
    sent: list = []
    with mock.patch.dict("os.environ", {"JEV_API_KEY": "jev-key"}, clear=True):
        decider = TypeSafeDecider(transport=transport(JEV_REPLY, sent))
    await decider.ask("x", QUESTIONS)
    assert sent[0]["auth"] == "Bearer jev-key"


def test_score_position_reads_probabilities_by_label_or_index_and_either_base():
    levels = ["Low", "Middle", "High"]
    assert score_position(0, levels, {"High": 1.0}) == pytest.approx(1.0)
    assert score_position(0, levels, {"0": 0.5, "2": 0.5}) == pytest.approx(0.5)
    assert score_position(0, levels, {"3": 1.0}, index_hint=["1", "2", "3"]) == pytest.approx(1.0)
    assert score_position(1.0, levels, {}) == pytest.approx(0.5)
    assert score_position(0.8, levels, {}, raw_scale="unit") == pytest.approx(0.8)
    assert score_position(1.0, ["Only"], {}) is None


def _llm_reply(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=120, completion_tokens=20))


async def test_llm_decider_parses_typed_answers_from_a_json_reply():
    content = "```json\n" + json.dumps({"answers": {
        "consequence": {"probabilities": {"none": 0.2, "sends": 0.8}},
        "ambiguous": {"probability": 0.7},
        "progress": {"probabilities": {"0": 0.0, "1": 0.0, "Done": 1.0}},
    }}) + "\n```"
    decider = LLMDecider("openai/gpt-5.4-mini", params={"reasoning_effort": "minimal"})
    with mock.patch("helix.deciders.llm.acompletion", new=mock.AsyncMock(return_value=_llm_reply(content))) as call:
        answers = await decider.ask("Send the reply to Dana", QUESTIONS)
    assert call.await_args.kwargs["reasoning_effort"] == "minimal"
    assert (answers["consequence"].value, answers["consequence"].confidence) == ("sends", pytest.approx(0.8))
    assert answers["ambiguous"].value == pytest.approx(0.7)
    assert answers["progress"].position == pytest.approx(1.0)
    assert (decider.usage().calls, decider.usage().input_tokens) == (1, 120)


async def test_llm_decider_falls_back_to_uniform_probabilities_when_the_reply_has_none():
    decider = LLMDecider("openai/gpt-5.4-mini")
    with mock.patch("helix.deciders.llm.acompletion", new=mock.AsyncMock(return_value=_llm_reply("no idea"))):
        answers = await decider.ask("x", QUESTIONS)
    assert answers["consequence"].probabilities == {"none": 0.5, "sends": 0.5}
    assert answers["ambiguous"].value == 0.5
    assert answers["progress"].position == pytest.approx(0.5)


async def test_recording_decider_writes_answers_that_replay_serves_unchanged(tmp_path):
    sent: list = []
    path = tmp_path / "answers.jsonl"
    recorder = RecordingDecider(TypeSafeDecider(api_key="k", transport=transport(JEV_REPLY, sent)), path)
    live = await recorder.ask("Send the reply to Dana", QUESTIONS)
    replay = ReplayDecider(path)
    replayed = await replay.ask("Send the reply to Dana", QUESTIONS)
    assert replayed == live
    assert replay.name == "replay/jev-latest"
    assert len(replay) == 1


async def test_replay_decider_reports_recorded_cost_and_seconds(tmp_path):
    path = tmp_path / "answers.jsonl"
    record = {"key": question_key("s", {"ambiguous": QUESTIONS["ambiguous"]}), "decider": "jev-latest",
              "version": "jev-1.13.0", "state": "s", "questions": {}, "seconds": 0.31,
              "answers": {"ambiguous": Answer("probability", 0.4).to_dict()},
              "input_tokens": 410, "dollars": 0.0000172}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    replay = ReplayDecider(path)
    await replay.ask("s", {"ambiguous": QUESTIONS["ambiguous"]})
    usage = replay.usage()
    assert (usage.calls, usage.input_tokens, usage.version) == (1, 410, "jev-1.13.0")
    assert usage.seconds == pytest.approx(0.31)


async def test_replay_decider_fails_on_a_miss_in_strict_mode(tmp_path):
    replay = ReplayDecider(tmp_path / "empty.jsonl")
    with pytest.raises(KeyError, match="no recorded answer"):
        await replay.ask("unseen", QUESTIONS)


class _AskOnlyDecider:
    """A third-party decision model that offers `ask` but no usage report."""

    name = "third-party"

    async def ask(self, state, questions):
        return {"ambiguous": Answer("probability", 0.9)}


async def test_replay_decider_answers_a_miss_with_its_fallback(tmp_path):
    replay = ReplayDecider(tmp_path / "empty.jsonl", fallback=_AskOnlyDecider())
    answers, call = await replay.ask_with_usage("unseen", {"ambiguous": QUESTIONS["ambiguous"]})
    assert answers["ambiguous"].value == 0.9
    assert call.calls == 1 and call.version == "third-party"


async def test_recording_decider_records_each_calls_own_usage(tmp_path):
    inner = LLMDecider("openai/gpt-5.4-mini")
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"answers": {"ambiguous": {"probability": 0.8}}}'))],
        usage=SimpleNamespace(prompt_tokens=40),
    )
    path = tmp_path / "rec.jsonl"
    recorder = RecordingDecider(inner, path)
    with mock.patch("helix.deciders.llm.acompletion", mock.AsyncMock(return_value=reply)):
        await recorder.ask("first", {"ambiguous": QUESTIONS["ambiguous"]})
        _, second = await recorder.ask_with_usage("second", {"ambiguous": QUESTIONS["ambiguous"]})
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["input_tokens"] for r in records] == [40, 40]
    assert second.calls == 1 and inner.usage().calls == 2


def test_decider_from_name_selects_the_adapter_by_prefix(tmp_path):
    with mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": "k", "OPENAI_API_KEY": "k"}):
        assert isinstance(decider_from_name("jev-latest"), TypeSafeDecider)
        assert isinstance(decider_from_name("openai/gpt-6-luna"), OpenAIDecisionsDecider)
    assert isinstance(decider_from_name("llm/openai/gpt-5.4-mini"), LLMDecider)
    assert isinstance(decider_from_name(f"replay:{tmp_path / 'a.jsonl'}"), ReplayDecider)
    with pytest.raises(ValueError, match="unknown decision model"):
        decider_from_name("gpt-4o")


def test_questions_survive_a_round_trip_through_dicts():
    for question in QUESTIONS.values():
        assert question_from_dict(question_to_dict(question)) == question
    with_criteria = Probability("Urgent", {"true": "needs action today", "false": "can wait"})
    assert question_from_dict(question_to_dict(with_criteria)) == with_criteria


class _DriftingDecider:
    """A live decision model whose answer to the same question drifts."""

    name = "drifting"

    def __init__(self) -> None:
        self.calls = 0

    async def ask(self, state, questions):
        self.calls += 1
        return {"ambiguous": Answer("probability", 0.5 + self.calls / 100)}


async def test_replay_with_a_fallback_answers_a_repeated_miss_the_same_way(tmp_path):
    live = _DriftingDecider()
    replay = ReplayDecider(tmp_path / "empty.jsonl", fallback=live)
    first = await replay.ask("unseen", {"ambiguous": QUESTIONS["ambiguous"]})
    again = await replay.ask("unseen", {"ambiguous": QUESTIONS["ambiguous"]})
    assert first["ambiguous"].value == again["ambiguous"].value == 0.51
    assert live.calls == 1
