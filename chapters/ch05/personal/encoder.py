"""A fixed lexical encoder for scheduling experiences, with no downloads."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import re

ENCODER_ID = "calendar-token-hash-1024-v1"
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

