"""Group-relative selection: GRPO's advantage as a Search. SPEC §4.2, §4.2.1.

`propose` yields a group of N candidates from a sampler and `select` returns
the one with the highest advantage over the group mean. The reinforcement
is promotion of that artifact, never a gradient step; the arithmetic lives
in helix.improvement.group_round so the round and the search agree.

The sampler is the pluggable part. Chapter 5 ships InsightSampler, which
asks a model for N candidate lessons from one trajectory and mints each as
a MEMORY_ENTRY artifact, and SplitSampler, a deterministic splitter that
tests and demos use to build a group without a model.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from helix.artifact import Artifact, ArtifactKind, genesis
from helix.memory.archive_bridge import entry_payload, memory_artifact_content
from helix.memory.base import MemoryEntry, Scope, ScopeKey, memory_artifact_id
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import MutationProposed, SearchCompleted, SearchStarted
from helix.search.base import SearchBudget, SearchCostModel, SearchKind, Variant
from helix.signal import Signal
from helix.trajectory import Trajectory

if TYPE_CHECKING:
    from helix.archive import Archive


SamplerFn = Callable[[Artifact, int, dict[str, Any]], Awaitable[list[Artifact]]]


def _advantage_of(variant: Variant) -> float:
    """Rank by advantage when a group round filled it, else by raw score."""
    m = variant.measurement
    if m is None:
        return float("-inf")
    adv = m.metadata.get("advantage")
    if adv is not None:
        return float(adv)
    return float(m.score) if m.score is not None else float("-inf")


class GroupRelativeSearch:
    """N candidates per round, selected by advantage over the group mean."""

    def __init__(
        self,
        sampler: SamplerFn | Any,
        group_size: int = 4,
        select: str = "positive",
        context: dict[str, Any] | None = None,
        bus: EventBus | None = None,
    ) -> None:
        if select not in ("positive", "top_k"):
            raise ValueError(f"select must be 'positive' or 'top_k', got {select!r}")
        self.sampler = sampler
        self.group_size = group_size
        self.select_rule = select
        # Context reaches the sampler untouched: the trajectory to mine for
        # insights, the scope to file them under, anything else it needs.
        self.context = dict(context or {})
        self.bus = bus or get_bus()

    @property
    def kind(self) -> SearchKind:
        return SearchKind.GROUP_RELATIVE

    @property
    def cost_model(self) -> SearchCostModel:
        return SearchCostModel(
            proposes_per_round=self.group_size,
            rounds_typical=1,
            signal_calls_per_round=self.group_size,
        )

    async def _sample(self, parent: Artifact, n: int, context: dict[str, Any]) -> list[Artifact]:
        fn = getattr(self.sampler, "sample", self.sampler)
        result = fn(parent, n, context)
        if inspect.isawaitable(result):
            result = await result
        return list(result)

    async def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: Archive,
        budget: SearchBudget,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Variant]:
        ctx = {**self.context, **(context or {})}
        await self.bus.publish(
            SearchStarted(
                search_kind=self.kind.value,
                seed_artifact_id=seed.id,
                seed_version=seed.version,
                budget={"max_candidates": budget.max_candidates, "group_size": self.group_size},
            )
        )
        yielded = 0
        try:
            sampled = await self._sample(seed, self.group_size, ctx)
            # Siblings that share the seed's id are re-minted against the
            # archive's next free version so no two of them collide.
            next_free = await archive.next_version(seed.id) if hasattr(archive, "next_version") else seed.version + 1
            base_version = max(next_free, seed.version + 1)
            for i, art in enumerate(sampled[: self.group_size]):
                if budget.exhausted():
                    break
                if art.id == seed.id and art.parent_id == seed.ref:
                    art = seed.mutate(
                        art.content,
                        created_by=art.created_by,
                        metadata=dict(art.metadata),
                        version=base_version + i,
                    )
                excerpt = art.content if isinstance(art.content, str) else str(entry_payload(art) or art.content)
                await self.bus.publish(
                    MutationProposed(
                        search_kind=self.kind.value,
                        parent_artifact_id=seed.id,
                        parent_version=seed.version,
                        candidate_version=art.version,
                        proposer_model=str(getattr(self.sampler, "model", type(self.sampler).__name__)),
                        rationale=f"group candidate {i + 1}/{self.group_size}",
                        candidate_excerpt=excerpt[:200],
                    )
                )
                budget.saw_candidate()
                yielded += 1
                yield Variant(
                    artifact=art,
                    parent=seed,
                    search_method=self.kind.value,
                    metadata={"group_index": i, "group_size": self.group_size},
                )
        finally:
            await self.bus.publish(
                SearchCompleted(
                    search_kind=self.kind.value,
                    winner_artifact_id=seed.id,
                    winner_version=seed.version,
                    candidates_evaluated=yielded,
                )
            )

    async def select(
        self,
        candidates: list[Variant],
        signal: Signal,
        archive: Archive,
    ) -> Artifact:
        measured = [c for c in candidates if c.measurement is not None]
        if measured:
            return max(measured, key=_advantage_of).artifact
        if candidates:
            return candidates[0].artifact
        raise ValueError("GroupRelativeSearch.select: no candidates")


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

INSIGHT_PROMPT = """You are distilling lessons from one agent run so a future run can do better.

Read the run below and write {n} distinct, short, reusable lessons (one sentence
each). Each lesson must be actionable on a similar future task. Number them
1 through {n}, one per line, and output nothing else.

Task: {task}

Run:
{body}

Final answer: {final_output}
"""


def _trajectory_text(trajectory: Trajectory, limit: int = 6000) -> str:
    lines = []
    for step in trajectory.steps:
        lines.append(f"[{step.kind.value}] {step.payload}")
    return "\n".join(lines)[:limit]


def parse_numbered_lines(text: str, n: int) -> list[str]:
    """Pull up to n lesson strings out of a numbered or bulleted list."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        if stripped and stripped not in out:
            out.append(stripped)
        if len(out) >= n:
            break
    return out


class InsightSampler:
    """Extract N candidate insights from one trajectory as memory artifacts.

    Each insight becomes its own genesis MEMORY_ENTRY (ExpeL ADD) with
    utility 0.0; the group round then scores them and the positive ones
    survive. The trajectory arrives in `context["trajectory"]`, or the
    parent artifact's entry payload is used when none is given.
    """

    def __init__(
        self,
        llm: Callable[[str], Awaitable[str]],
        n: int = 4,
        tier: str = "episodic",
        prompt: str = INSIGHT_PROMPT,
        model: str = "injected",
    ) -> None:
        self.llm = llm
        self.n = n
        self.tier = tier
        self.prompt = prompt
        self.model = model

    async def sample(self, parent: Artifact, n: int, context: dict[str, Any]) -> list[Artifact]:
        n = n or self.n
        trajectory: Trajectory | None = context.get("trajectory")
        if trajectory is not None:
            task, body, final = trajectory.task, _trajectory_text(trajectory), trajectory.final_output
            source_id = trajectory.id
        else:
            payload = entry_payload(parent)
            task = str(payload.get("user_message") or payload.get("task") or "")
            body = str(payload.get("summary") or payload)
            final = payload.get("final_output")
            source_id = str(payload.get("trajectory_id") or parent.id)
        scope_str = context.get("scope")
        if scope_str is None and isinstance(parent.content, dict):
            scope_str = parent.content.get("scope")
        scope_key = ScopeKey.from_str(scope_str) if scope_str else ScopeKey(Scope.GLOBAL, "*")

        text = await self.llm(self.prompt.format(n=n, task=task, body=body, final_output=final))
        insights = parse_numbered_lines(text, n)
        return [
            self.make_artifact(insight, source_id, scope_key, i)
            for i, insight in enumerate(insights)
        ]

    def make_artifact(self, insight: str, source_id: str, scope_key: ScopeKey, index: int) -> Artifact:
        digest = hashlib.sha256(f"{source_id}:{insight}".encode("utf-8")).hexdigest()[:10]
        entry = MemoryEntry(
            id=f"insight-{digest}",
            scope_key=scope_key,
            content={"insight": insight, "source_trajectory_id": source_id},
        )
        return genesis(
            id=memory_artifact_id(self.tier, entry.id),
            kind=ArtifactKind.MEMORY_ENTRY,
            content=memory_artifact_content(self.tier, entry, utility=0.0),
            created_by="group_relative",
            metadata={
                "tier": self.tier,
                "scope": scope_key.to_str(),
                "entry_id": entry.id,
                "operator": "add",
                "group_index": index,
                "source_trajectory_id": source_id,
            },
        )


class SplitSampler:
    """Deterministic sampler: split the seed's text into N sibling variants.

    Text artifacts are split into N contiguous chunks of sentences; memory
    entry artifacts split their `insight` field the same way and keep the
    rest of the content dict. No model is involved, which is what makes it
    the right sampler for tests and offline demos.
    """

    model = "deterministic"

    def __init__(self, n: int | None = None) -> None:
        self.n = n

    async def sample(self, parent: Artifact, n: int, context: dict[str, Any]) -> list[Artifact]:
        n = self.n or n
        if isinstance(parent.content, str):
            text = parent.content
            build = lambda chunk: chunk  # noqa: E731
        elif isinstance(parent.content, dict):
            payload = entry_payload(parent)
            text = str(payload.get("insight") or payload.get("final_output") or payload)

            def build(chunk: str) -> dict[str, Any]:
                return {
                    **parent.content,
                    "entry": {**payload, "insight": chunk},
                }
        else:
            raise TypeError("SplitSampler supports text and memory-entry artifacts")

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        if not sentences:
            sentences = [text]
        chunks = _chunk(sentences, n)
        return [
            parent.mutate(
                build(" ".join(chunk)),
                created_by="split_sampler",
                metadata={**dict(parent.metadata), "group_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]


def _chunk(items: list[str], n: int) -> list[list[str]]:
    """Split into at most n non-empty contiguous groups, as even as possible."""
    n = max(1, min(n, len(items)))
    size, extra = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        out.append(items[start:end])
        start = end
    return out
