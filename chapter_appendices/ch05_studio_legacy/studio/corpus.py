"""Small local runbook search and a fixed lexical encoder, with no downloads."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import re

CORPUS_DIR = Path(__file__).with_name("corpus")
ENCODER_ID = "studio-token-hash-1024-v1"
DIMENSIONS = 1024
STOPWORDS = set("a an and are as at be before by can do for from has have how i if in is it its me my of on or our so that the their this to was we what when which with you your".split())


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def embed(texts: list[str]) -> list[list[float]]:
    """Normalize hashed token counts; this is lexical matching, not semantics.

    A stable digest keeps vectors identical across processes. The encoder is
    fixed at construction and must be reused when reopening the memory store.
    """
    vectors = []
    for text in texts:
        vector = [0.0] * DIMENSIONS
        for word, count in Counter(tokens(text)).items():
            slot = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big") % DIMENSIONS
            vector[slot] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


class StudioCorpus:
    """Search only approved Markdown runbooks, never ticket outcome fixtures."""

    def __init__(self, directory: Path = CORPUS_DIR):
        self.passages = []
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].removeprefix("# ")
            for section in re.split(r"\n## ", text)[1:]:
                heading, _, body = section.partition("\n")
                self.passages.append({"source": path.name, "section": heading, "text": title + "\n" + body.strip()})
        if not self.passages:
            raise ValueError(f"No studio runbooks in {directory}")
        self.vectors = embed([p["text"] + " " + p["section"] for p in self.passages])

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not tokens(query):
            return []
        vector = embed([query])[0]
        scored = [(sum(a*b for a, b in zip(vector, other)), passage)
                  for passage, other in zip(self.passages, self.vectors)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [dict(passage, score=round(score, 6)) for score, passage in scored[:max(0, min(k, 10))] if score > 0]
