"""The record and replay layer: keys, modes, and per-context routing."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from metacog import recording
from metacog.recording import MissingReply, Recording, call_key, restore_seconds, served_seconds

CALL = {
    "model": "gpt-5.4-mini",
    "messages": [{"role": "system", "content": "You help."}, {"role": "user", "content": "Hi"}],
    "tools": None,
    "reasoning_effort": "none",
}


def fake_response(content: str = "hello", *, tool_call: bool = False) -> SimpleNamespace:
    calls = [SimpleNamespace(id="call_1", function=SimpleNamespace(name="look", arguments='{"q": "x"}'))]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls if tool_call else None))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
    )


@pytest.fixture
def live(monkeypatch):
    """Stand in for the real model: count calls and answer with a fixed reply."""
    recording.install()
    calls: list[dict] = []

    async def model(**kwargs):
        calls.append(kwargs)
        return fake_response(f"reply {len(calls)}", tool_call=True)

    monkeypatch.setattr(recording, "_original", model)
    return calls


async def model_call(**kwargs):
    """A call made the way helix makes it, through helix.llm_call."""
    import helix.llm_call

    return await helix.llm_call.acompletion(**kwargs)


def test_keys_match_the_book_recordings():
    from chapters.ch06.assistant.recording import call_key as book_key

    assert call_key(CALL) == book_key(CALL)
    assert call_key(CALL) != call_key({**CALL, "reasoning_effort": "high"})
    assert call_key(CALL) == call_key({**CALL, "timeout": 30})  # transport settings do not change a reply


async def test_record_calls_once_then_serves_the_saved_reply(tmp_path, live):
    rec = Recording(tmp_path / "r.jsonl", mode="record")
    with rec.active():
        first = await model_call(**CALL)
        second = await model_call(**CALL)
    assert len(live) == 1 and rec.live_calls == 1
    assert first.choices[0].message.content == second.choices[0].message.content == "reply 1"
    assert first.choices[0].message.tool_calls[0].function.name == "look"
    row = json.loads((tmp_path / "r.jsonl").read_text(encoding="utf-8"))
    assert row["key"] == call_key(CALL) and row["usage"]["reasoning_tokens"] == 7


async def test_replay_serves_without_calling_and_refuses_unknown_calls(tmp_path, live):
    with Recording(tmp_path / "r.jsonl", mode="record").active():
        await model_call(**CALL)
    rec = Recording(tmp_path / "r.jsonl", mode="replay")
    with rec.active():
        reply = await model_call(**CALL)
        with pytest.raises(MissingReply):
            await model_call(**{**CALL, "reasoning_effort": "high"})
    assert reply.choices[0].message.content == "reply 1"
    assert len(live) == 1  # only the recording call reached the model


async def test_live_mode_calls_every_time(tmp_path, live):
    with Recording(tmp_path / "r.jsonl", mode="live").active():
        await model_call(**CALL)
        await model_call(**CALL)
    assert len(live) == 2


async def test_distinct_repeats_give_each_repeat_its_own_reply(tmp_path, live):
    with Recording(tmp_path / "r.jsonl", mode="record", distinct_repeats=True).active():
        replies = [await model_call(**CALL) for _ in range(3)]
    assert [r.choices[0].message.content for r in replies] == ["reply 1", "reply 2", "reply 3"]
    with Recording(tmp_path / "r.jsonl", mode="replay", distinct_repeats=True).active():
        replayed = [await model_call(**CALL) for _ in range(3)]
    assert [r.choices[0].message.content for r in replayed] == ["reply 1", "reply 2", "reply 3"]


async def test_concurrent_contexts_use_their_own_recordings(tmp_path):
    for name, content in (("a.jsonl", "from a"), ("b.jsonl", "from b")):
        row = {"key": call_key(CALL), "model": CALL["model"], "seconds": 1.5, "content": content,
               "tool_calls": [], "usage": {}}
        (tmp_path / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

    async def answer(name: str) -> str:
        with Recording(tmp_path / name).active():
            await asyncio.sleep(0)  # let the other task interleave
            return (await model_call(**CALL)).choices[0].message.content

    assert await asyncio.gather(answer("a.jsonl"), answer("b.jsonl")) == ["from a", "from b"]


async def test_without_a_recording_calls_reach_the_model(live):
    reply = await model_call(**CALL)
    assert reply.choices[0].message.content == "reply 1" and len(live) == 1


async def test_served_seconds_restore_recorded_latency(tmp_path):
    from helix.trajectory import StepKind, Trajectory

    row = {"key": call_key(CALL), "model": CALL["model"], "seconds": 2.25, "content": "ok",
           "tool_calls": [], "usage": {}}
    (tmp_path / "r.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    trajectory = Trajectory(task="t")
    trajectory.append(StepKind.MODEL_CALL, {"seconds": 0.001})
    with Recording(tmp_path / "r.jsonl").active(), served_seconds() as seconds:
        await model_call(**CALL)
    restore_seconds(trajectory, seconds)
    assert seconds == [2.25] and trajectory.steps[0].payload["seconds"] == 2.25
