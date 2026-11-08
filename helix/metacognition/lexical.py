"""A fixed lexical encoder: hashed token counts, with no model download.

This is the encoder the book's chapter 5 and 6 harnesses use for small
offline stores, kept identical so recall rankings, and therefore the requests
an agent sends, match the book's recordings. It matches words, not meanings:
"meeting" and "appointment" share nothing. Reopen a store with the encoder
that wrote it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

ENCODER_ID = "token-hash-1024-v1"
DIMENSIONS = 1024
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can", "do", "for", "from", "has", "have", "how",
    "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so", "that", "the", "their", "this",
    "to", "was", "we", "what", "when", "which", "with", "you", "your",
}


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def embed(texts: list[str]) -> list[list[float]]:
    """Normalized hashed token counts, stable across processes."""
    vectors = []
    for text in texts:
        vector = [0.0] * DIMENSIONS
        for word, count in Counter(tokens(text)).items():
            slot = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big") % DIMENSIONS
            vector[slot] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors
