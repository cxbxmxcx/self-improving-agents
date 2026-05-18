"""EvalQuestion and EvalSet types plus the JSON loader.

The fixed-corpus eval set is the Ch 2 default. Each question carries a band
(difficulty), tags, an expected v0 failure mode, and a reference answer. The
EvalSet is band-aware so a search method can request a stratified sample
without re-loading from disk.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    band: int
    question: str
    reference_answer: str
    expected_failure_mode_v0: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    category: str = ""  # v2 schema; v1 leaves empty

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


@dataclass
class EvalSet:
    """A typed collection of EvalQuestion. Band-aware accessors included."""

    questions: list[EvalQuestion]
    description: str = ""
    bands: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def by_band(self, band: int) -> list[EvalQuestion]:
        return [q for q in self.questions if q.band == band]

    def by_id(self, qid: str) -> EvalQuestion | None:
        return next((q for q in self.questions if q.id == qid), None)

    def sample(self, n: int, stratified: bool = True, seed: int | None = None) -> list[EvalQuestion]:
        """Sample n questions. If stratified, sample proportionally per band."""
        rng = random.Random(seed)
        if not stratified:
            return rng.sample(self.questions, k=min(n, len(self.questions)))
        bands = sorted({q.band for q in self.questions})
        per_band = max(1, n // len(bands))
        out: list[EvalQuestion] = []
        for b in bands:
            pool = self.by_band(b)
            out.extend(rng.sample(pool, k=min(per_band, len(pool))))
        return out[:n]


def load_eval_set(path: str | Path) -> EvalSet:
    """Load eval_questions.json (v1 or v2 schema) into a typed EvalSet.

    v1 schema has a `band` integer per question (1-4) and a top-level `bands`
    map describing each.

    v2 schema has a `category` string per question and a top-level `categories`
    map; band defaults to 3 (multi-hop) if not provided since v2 questions are
    deliberately harder.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    questions = [
        EvalQuestion(
            id=q["id"],
            band=int(q["band"]) if "band" in q else 3,
            question=q["question"],
            reference_answer=q["reference_answer"],
            expected_failure_mode_v0=q.get("expected_failure_mode_v0", ""),
            tags=tuple(q.get("tags", [])),
            category=q.get("category", ""),
        )
        for q in data["questions"]
    ]
    bands = {int(k): v for k, v in data.get("bands", {}).items()}
    return EvalSet(
        questions=questions,
        description=data.get("description", ""),
        bands=bands,
    )
