"""Where the listings' decisions and proposals come from: a recording, or a
live model that adds to it.

Every listing reads the same recording first. In record mode, a question the
recording lacks goes to the live model and its answer joins the recording,
so each question has one recorded answer across all four listings. Replay
mode never calls a model, and a missing answer fails loudly instead of
reusing a stale one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from helix.deciders import LLMDecider, RecordingDecider, ReplayDecider, decider_from_name

RECORDED = Path(__file__).parent / "recorded"
MODES = ("replay", "record", "live")


def decision_model(mode: str, *, folder: Path = RECORDED, name: str = "jev-latest",
                   file: str = "decisions.jsonl") -> ReplayDecider:
    """The recording, with a live decision model behind it unless replaying.

    An LLM answers without reasoning, as a decision model should.
    """
    path = folder / file
    if mode == "replay":
        return ReplayDecider(path)
    live = (LLMDecider(name.split("/", 1)[1], params={"reasoning_effort": "none"}) if name.startswith("llm/")
            else decider_from_name(name))
    return ReplayDecider(path, fallback=RecordingDecider(live, path), name=live.name)


def _key(kwargs: dict[str, Any]) -> str:
    canonical = json.dumps({"model": kwargs.get("model"), "messages": kwargs.get("messages")}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def recorded_proposals(path: Path, *, live: bool) -> Iterator[None]:
    """Serve the search proposer's replies from a recording; when live, ask
    the proposer about prompts the recording lacks and keep its replies."""
    target = "helix.search.gepa.helix_acompletion"
    replies: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                replies.setdefault(row["key"], row["content"])

    async def proposer(**kwargs: Any):
        key = _key(kwargs)
        if key not in replies:
            if not live:
                raise KeyError("no recorded proposal for this prompt; rerun the search with --record or --live")
            from helix.llm_call import acompletion

            response = await acompletion(**kwargs)
            replies[key] = response.choices[0].message.content
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "model": kwargs.get("model"), "content": replies[key]}) + "\n")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies[key]))], usage=None)

    with patch(target, side_effect=proposer):
        yield
