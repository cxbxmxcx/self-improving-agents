"""The support inbox, the production-scale test's desk with real messages, on a short soak (no key, no spend).

The datasets are not in the repository. These tests run where they have been downloaded (see examples/inbox/README.md)
and are skipped elsewhere.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from examples.inbox import policy, scripted, world
from examples.inbox.stream import DATA, Stream, plan

pytestmark = pytest.mark.skipif(not (DATA / "banking77" / "train.csv").exists(),
                                reason="Banking77 and CLINC150 have not been downloaded")
SHORT = ["--per-day", "20", "--days", "10", "--held", "8"]


def test_every_topic_has_one_team_in_every_version_of_the_policy():
    assert len(policy.TOPICS) == len(set(policy.TOPICS)) == 77
    for version in policy.POLICIES.values():
        assert set(version.teams) == set(policy.TOPICS)
        assert version.route("exchange_rate", "metal") == {"ticket": False, "team": None, "urgent": False}
        assert version.route("lost_or_stolen_card", "metal")["urgent"]  # urgent tickets keep their team
    assert policy.FIRST.route("transaction_charged_twice", "premium")["team"] == "Premium care"
    assert policy.SECOND.route("transaction_charged_twice", "premium")["team"] == "Disputes"
    assert policy.THIRD.route("card_arrival", "premium")["team"] == "Cards"


def test_the_reference_function_reads_the_policy_and_a_literal_one_goes_stale():
    for key in policy.POLICIES:
        for topic in policy.TOPICS[::5]:
            assert world.check_function(world.REFERENCE, topic, "C00013", key)[0]
    learned_first = scripted.literal_function("2026-1")
    stale = [t for t in policy.DISPUTED if not world.check_function(learned_first, t, "C00001", "2026-2")[0]]
    assert stale == list(policy.DISPUTED)  # every disputed topic moved team, and the first version's function missed it


def test_the_stream_holds_real_messages_and_its_five_changes():
    stream = Stream()
    days = {day.number: day for day in stream.days}
    assert {n: d.change for n, d in days.items() if d.change} == {
        5: "disputed payments move to a new Disputes team", 8: "CLINC150's writers arrive",
        11: "the policy service is down for one message in ten", 14: "top-ups launch",
        17: "premium care narrows to the metal plan"}
    served = [r for day in stream.days for r in stream.served(day)]
    texts = [r["text"] for r in served] + [r["text"] for r in stream.exam()] + [r["text"] for r in stream.start()]
    assert len(set(texts)) == len(texts) == 4240  # 4,000 served, the 200 of the exam and the 40 of the start
    kinds = Counter(r["kind"] for r in served)
    assert kinds["out"] == 400 and kinds["clinc"] == 260
    assert not any(r["topic"] in policy.TOP_UPS for r in served if r["day"] < 14)
    assert sum(r["outage"] for r in served) == sum(r["outage"] for r in stream.served(days[11]))
    later = [r for r in served if r["day"] >= 5 and r["expect"] == "act"]
    assert 0.3 < sum(r["sensitive"] for r in later) / len(later) < 0.6


@pytest.fixture
def soak(tmp_path: Path) -> list[str]:
    return ["--soak", *SHORT, "--output", str(tmp_path)]


async def test_a_soak_learns_the_starting_function_and_serves_the_short_stream(soak: list[str], tmp_path: Path):
    from examples.inbox import run
    from examples.scale import desk_report

    start = await run.run_start(run.parse(["start", *soak]))
    assert start["learned"] is not None and start["stopped"] == ""
    full = await run.run_arm("full", run.parse(["full", *soak]))
    loop = await run.run_arm("loop", run.parse(["loop", *soak]))
    alone = await run.run_arm("alone", run.parse(["alone", *soak]))
    await run.run_searches(run.parse(["searches", *soak, "--searches", "2"]))
    assert full["stopped"] == loop["stopped"] == alone["stopped"] == ""
    assert any("installed" in event for event in full["nights"][0]["events"])
    stream = [row for row in full["rows"] if row["phase"] == "stream"]
    assert sum(row["outcome"] == "program" for row in stream) >= 0.3 * len(stream)
    relearned = [e for e in full["learning"] if e["name"] == "route_ticket" and "installed" in e["outcome"]]
    assert relearned  # a policy change made the literal function stale, and the loop relearned it
    out = desk_report.report(tmp_path)
    assert [c["number"] for c in out["criteria"]] == list(range(1, 10))
    assert [change["day"] for change in out["changes"]] == [2, 4, 6, 7, 8]
    assert len(plan(10)) == 10
