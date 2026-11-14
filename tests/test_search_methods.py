"""GEPA and DGM share one evaluator contract, and the archive's quality-diversity sampling explores."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from helix.archive import SQLiteArchive
from helix.archive.archive import SamplingStrategy
from helix.artifact import ArtifactKind, genesis
from helix.search import dgm as dgm_module
from helix.search import gepa as gepa_module
from helix.search.base import SearchBudget, Variant
from helix.search.methods import METHODS, build_search
from helix.signal import GapMeasurement


def reply(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(total_tokens=10))


def measure(score: float, **metadata) -> GapMeasurement:
    return GapMeasurement(score=score, signal_id="s", signal_version=1, triggered=True, raw_value=score,
                          feedback=f"scored {score}", metadata=metadata)


@pytest.mark.parametrize("method", METHODS)
async def test_either_method_measures_every_candidate_through_the_same_evaluator(method, monkeypatch):
    prompts: list[str] = []

    async def proposer(model, messages, **kwargs):
        prompts.append(messages[0]["content"])
        return reply(f"candidate {len(prompts)}")

    monkeypatch.setattr(gepa_module, "helix_acompletion", proposer)
    monkeypatch.setattr(dgm_module, "helix_acompletion", proposer)
    scored: list[str] = []

    async def evaluate(candidate, questions=None):
        scored.append(candidate.content)
        return measure(0.1 * len(scored))

    search = build_search(method, proposer_model="m", evaluator=evaluate, mutation_prompt="mutate",
                          crossover_prompt="cross", population=2, generations=2, rng=random.Random(3))
    archive = SQLiteArchive(":memory:")
    seed = genesis("mechanism", ArtifactKind.CODE, "the seed", created_by="test")
    async for _ in search.propose(seed, None, archive, SearchBudget(max_candidates=4)):
        pass
    assert scored[0] == "the seed" and len(scored) == 5  # the seed, then each candidate once
    assert len(set(scored)) == 5  # identical prompts would have collapsed these into one design
    assert len(set(prompts)) == len(prompts)
    best = await archive.best(k=1, artifact_id=seed.id)
    assert best and best[0].measurement.score is not None
    archive.close()


async def test_dgm_rejected_proposals_go_back_with_the_reason_and_are_dropped_if_never_valid(monkeypatch):
    asked: list[str] = []

    async def proposer(model, messages, **kwargs):
        asked.append(messages[1]["content"])
        return reply("bad")

    monkeypatch.setattr(dgm_module, "helix_acompletion", proposer)

    async def evaluate(candidate, questions=None):
        return measure(0.5)

    search = build_search("dgm", proposer_model="m", evaluator=evaluate, mutation_prompt="mutate",
                          validate=lambda source: "it is bad", population=1, generations=1, rng=random.Random(1))
    archive = SQLiteArchive(":memory:")
    seed = genesis("mechanism", ArtifactKind.CODE, "the seed", created_by="test")
    assert [v async for v in search.propose(seed, None, archive, SearchBudget(max_candidates=1))] == []
    assert len(asked) == 3 and "was rejected: it is bad" in asked[-1]
    assert len(search.discarded) == 3
    archive.close()


async def test_a_search_on_a_kept_archive_resumes_without_measuring_the_seed_again(monkeypatch, tmp_path):
    async def proposer(model, messages, **kwargs):
        return reply(f"candidate {messages[0]['content'][-80:]}")

    monkeypatch.setattr(dgm_module, "helix_acompletion", proposer)
    scored: list[str] = []

    async def evaluate(candidate, questions=None):
        scored.append(candidate.content)
        return measure(0.5)

    seed = genesis("mechanism", ArtifactKind.CODE, "the seed", created_by="test")
    path = str(tmp_path / "archive.sqlite")
    for _ in range(2):
        search = build_search("dgm", proposer_model="m", evaluator=evaluate, mutation_prompt="mutate",
                              population=1, generations=1, rng=random.Random(1))
        archive = SQLiteArchive(path)
        async for _variant in search.propose(seed, None, archive, SearchBudget(max_candidates=1)):
            pass
        archive.close()
    assert scored.count("the seed") == 1


async def seeded_archive(first_score: float, *others: tuple[float, list[str] | None]) -> tuple[SQLiteArchive, list]:
    archive = SQLiteArchive(":memory:")
    seed = genesis("mechanism", ArtifactKind.CODE, "v0", created_by="test")
    await archive.record(Variant(artifact=seed, parent=seed, search_method="t"), measure(first_score))
    variants = [seed]
    for n, (score, behavior) in enumerate(others, 1):
        child = seed.mutate(new_content=f"v{n}", created_by="test", version=n + 1)
        await archive.record(Variant(artifact=child, parent=seed, search_method="t"),
                             measure(score, **({"behavior": behavior} if behavior is not None else {})))
        variants.append(child)
    return archive, variants


async def test_quality_diversity_does_not_always_return_the_top_scorer():
    archive, variants = await seeded_archive(0.4, (0.9, None), (0.85, None), (0.8, None))
    rng = random.Random(5)
    drawn = {(await archive.sample(SamplingStrategy.QUALITY_DIVERSITY, artifact_id="mechanism", rng=rng)).artifact.version
             for _ in range(60)}
    assert len(drawn) >= 3
    archive.close()


async def test_quality_diversity_discounts_variants_that_behave_alike():
    archive, _ = await seeded_archive(0.0, (0.8, ["a"]), (0.8, ["a"]), (0.8, ["a"]), (0.8, ["b"]))
    rng = random.Random(9)
    counts: dict[int, int] = {}
    for _ in range(600):
        version = (await archive.sample(SamplingStrategy.QUALITY_DIVERSITY, artifact_id="mechanism", rng=rng)).artifact.version
        counts[version] = counts.get(version, 0) + 1
    assert counts[5] > 1.5 * max(counts.get(2, 0), counts.get(3, 0), counts.get(4, 0))  # the lone behavior is favored
    archive.close()
