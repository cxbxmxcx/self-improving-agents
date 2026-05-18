"""Paragraph chunker.

Splits text into paragraph chunks targeting 200-400 tokens, with a hard ceiling
matching the embedding model's context (512 tokens for bge-base-en-v1.5).

Strategy:
1. Split on blank lines into raw paragraphs.
2. Merge consecutive short paragraphs until we're near the target lower bound.
3. Split any paragraph that exceeds the ceiling by sentence (naive period-split).

This is the v0 chunker. Better chunking (semantic, structural) is a Ch 3 topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_CHAR_RATIO = 4  # rough approximation: 1 token per ~4 chars for English


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    paragraph_idx: int

    def approx_tokens(self) -> int:
        return max(1, len(self.text) // TOKEN_CHAR_RATIO)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // TOKEN_CHAR_RATIO)


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _word_chunks(text: str, target_tokens: int) -> list[str]:
    """Fallback: slice a punctuation-free run-on string into word groups."""
    words = text.split()
    target_words = max(1, target_tokens * TOKEN_CHAR_RATIO // 6)  # approx chars-per-word ~ 6
    out = []
    for i in range(0, len(words), target_words):
        out.append(" ".join(words[i : i + target_words]))
    return out


def chunk_text(
    text: str,
    source: str,
    page: int = 0,
    target_min: int = 200,
    target_max: int = 400,
    hard_ceiling: int = 480,
) -> list[Chunk]:
    """Chunk a block of text into paragraph-sized pieces.

    target_min/target_max set the desired token range. hard_ceiling is below
    the model's 512-token limit to leave room for special tokens.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    merged: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if _approx_tokens(candidate) <= target_max:
            buffer = candidate
            if _approx_tokens(buffer) >= target_min:
                merged.append(buffer)
                buffer = ""
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
            if _approx_tokens(buffer) >= target_min:
                merged.append(buffer)
                buffer = ""
    if buffer:
        merged.append(buffer)

    chunks: list[Chunk] = []
    idx = 0
    for piece in merged:
        if _approx_tokens(piece) <= hard_ceiling:
            chunks.append(Chunk(text=piece, source=source, page=page, paragraph_idx=idx))
            idx += 1
            continue

        sentences = _sentence_split(piece)
        # If sentence splitting collapses back to one giant string (no
        # punctuation), fall back to word-level slicing so the hard ceiling
        # is actually enforced. Without this, run-on text bypasses the limit.
        if len(sentences) <= 1 and _approx_tokens(piece) > hard_ceiling:
            sentences = _word_chunks(piece, hard_ceiling)
        cur = ""
        for sent in sentences:
            candidate = f"{cur} {sent}".strip() if cur else sent
            if _approx_tokens(candidate) > hard_ceiling and cur:
                chunks.append(Chunk(text=cur, source=source, page=page, paragraph_idx=idx))
                idx += 1
                cur = sent
            else:
                cur = candidate
        if cur:
            chunks.append(Chunk(text=cur, source=source, page=page, paragraph_idx=idx))
            idx += 1

    return chunks
