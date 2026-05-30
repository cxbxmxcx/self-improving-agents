"""Tests for the Guardrail wrapper and refusal short-circuit. SPEC §16.2.1, §6.2."""

from __future__ import annotations

import pytest

from helix.artifact import ArtifactKind, genesis, Subtype
from helix.guardrails import (
    Guardrail,
    GuardrailFailure,
    GuardrailVerdict,
    InputGuardrailPayload,
    OutputGuardrailPayload,
    compile_code_artifact,
)
from helix.hooks import HookPoint, Refusal


# ---------------- compile_code_artifact ----------------


def test_compile_code_artifact_extracts_async_check():
    art = genesis(
        id="g.pass",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    fn = compile_code_artifact(art)
    assert fn is not None
    assert callable(fn)


def test_compile_code_artifact_requires_async_check():
    art = genesis(
        id="g.sync",
        kind=ArtifactKind.CODE,
        content=(
            "def check(payload):\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    with pytest.raises(ValueError, match="must be async"):
        compile_code_artifact(art)


def test_compile_code_artifact_rejects_non_code_kind():
    art = genesis(id="g.prompt", kind=Subtype.PROMPT, content="some text")
    with pytest.raises(ValueError, match="ArtifactKind.CODE"):
        compile_code_artifact(art)


def test_compile_code_artifact_requires_named_function():
    art = genesis(
        id="g.empty",
        kind=ArtifactKind.CODE,
        content="x = 1\n",
    )
    with pytest.raises(ValueError, match="does not define"):
        compile_code_artifact(art)


# ---------------- Guardrail.evaluate ----------------


@pytest.mark.asyncio
async def test_input_guardrail_allows_clean_input():
    art = genesis(
        id="g.allow",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    return GuardrailVerdict(allow=True, reason='ok')\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input")
    v = await g.evaluate(InputGuardrailPayload(question="hello"))
    assert v.allow is True


@pytest.mark.asyncio
async def test_input_guardrail_refuses_pii():
    art = genesis(
        id="g.pii",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    if '@' in payload.question:\n"
            "        return GuardrailVerdict(allow=False, reason='contains_email')\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input")
    v = await g.evaluate(InputGuardrailPayload(question="email me at a@b.com"))
    assert v.allow is False
    assert v.reason == "contains_email"


@pytest.mark.asyncio
async def test_input_guardrail_patches_payload():
    art = genesis(
        id="g.scrub",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    cleaned = payload.question.replace('SECRET', '[redacted]')\n"
            "    if cleaned != payload.question:\n"
            "        return GuardrailVerdict(\n"
            "            allow=True, reason='scrubbed',\n"
            "            patched_payload=InputGuardrailPayload(question=cleaned, context=payload.context),\n"
            "        )\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input")
    v = await g.evaluate(InputGuardrailPayload(question="my SECRET key"))
    assert v.allow is True
    assert v.patched_payload is not None
    assert "[redacted]" in v.patched_payload.question


@pytest.mark.asyncio
async def test_output_guardrail_refuses_long_answer():
    art = genesis(
        id="g.length",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    if len(payload.answer) > 100:\n"
            "        return GuardrailVerdict(allow=False, reason='too_long')\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    g = Guardrail(artifact=art, phase="output")
    v = await g.evaluate(OutputGuardrailPayload(answer="x" * 200))
    assert v.allow is False
    assert v.reason == "too_long"


@pytest.mark.asyncio
async def test_guardrail_fail_open_swallows_exceptions():
    art = genesis(
        id="g.crash",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    raise RuntimeError('oops')\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input", fail_open=True)
    v = await g.evaluate(InputGuardrailPayload(question="x"))
    assert v.allow is True
    assert "guardrail_error_open" in v.reason


@pytest.mark.asyncio
async def test_guardrail_fail_closed_raises():
    art = genesis(
        id="g.crash2",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    raise RuntimeError('oops')\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input", fail_open=False)
    with pytest.raises(GuardrailFailure):
        await g.evaluate(InputGuardrailPayload(question="x"))


def test_guardrail_invalid_phase_raises():
    art = genesis(
        id="g.ok",
        kind=ArtifactKind.CODE,
        content="async def check(payload):\n    return GuardrailVerdict(allow=True)\n",
    )
    with pytest.raises(ValueError, match="phase"):
        Guardrail(artifact=art, phase="bogus")


def test_guardrail_hook_point_maps_correctly():
    art = genesis(
        id="g.ok",
        kind=ArtifactKind.CODE,
        content="async def check(payload):\n    return GuardrailVerdict(allow=True)\n",
    )
    assert Guardrail(artifact=art, phase="input").hook_point == HookPoint.PRE_MODEL
    assert Guardrail(artifact=art, phase="output").hook_point == HookPoint.PRE_OUTPUT


# ---------------- hook handler and refusal sentinel ----------------


@pytest.mark.asyncio
async def test_input_handler_returns_refusal_on_block():
    art = genesis(
        id="g.block",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    return GuardrailVerdict(allow=False, reason='blocked')\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input")
    handler = g.make_hook_handler()
    from helix.trajectory import Trajectory
    traj = Trajectory(task="bad input")
    result = await handler(messages=[], trajectory=traj)
    assert isinstance(result, Refusal)
    assert result.reason == "blocked"
    assert result.source.startswith("guardrail:g.block:v")


@pytest.mark.asyncio
async def test_input_handler_records_artifact_ref_on_trajectory():
    art = genesis(
        id="g.lineage",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input")
    handler = g.make_hook_handler()
    from helix.trajectory import Trajectory
    traj = Trajectory(task="hello")
    await handler(messages=[], trajectory=traj)
    # The handler should have recorded the guardrail's artifact ref at step 0.
    used = traj.artifacts_used.get(0, [])
    assert ("g.lineage", 1) in [tuple(r) for r in used]


@pytest.mark.asyncio
async def test_input_handler_returns_patched_payload():
    art = genesis(
        id="g.patch",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    return GuardrailVerdict(\n"
            "        allow=True,\n"
            "        patched_payload=InputGuardrailPayload(question='cleaned', context={}),\n"
            "    )\n"
        ),
    )
    g = Guardrail(artifact=art, phase="input")
    handler = g.make_hook_handler()
    from helix.trajectory import Trajectory
    traj = Trajectory(task="dirty")
    result = await handler(messages=[], trajectory=traj)
    assert isinstance(result, tuple)
    assert result[0] == "patched"
    assert result[1].question == "cleaned"


@pytest.mark.asyncio
async def test_output_handler_constructs_payload_from_response_string():
    art = genesis(
        id="g.out",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    if 'forbidden' in payload.answer:\n"
            "        return GuardrailVerdict(allow=False, reason='contains_forbidden')\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    g = Guardrail(artifact=art, phase="output")
    handler = g.make_hook_handler()
    from helix.trajectory import Trajectory
    traj = Trajectory(task="ask")
    bad = await handler(response="this is forbidden text", trajectory=traj)
    assert isinstance(bad, Refusal)
    good = await handler(response="this is fine", trajectory=traj)
    assert good is None
