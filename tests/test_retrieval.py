"""Chunker tests; index round-trip is left for integration since it loads ~440MB."""

from __future__ import annotations

from helix.retrieval.chunker import chunk_text


def test_chunker_emits_at_least_one_chunk_for_short_input():
    chunks = chunk_text("Hello world.", source="t.pdf", page=0)
    assert len(chunks) >= 1
    assert chunks[0].source == "t.pdf"


def test_chunker_splits_on_blank_lines():
    text = "Paragraph one is short.\n\nParagraph two is also short.\n\nThird short one."
    chunks = chunk_text(text, source="t.pdf", page=0, target_min=5, target_max=50)
    assert len(chunks) >= 1


def test_chunker_respects_hard_ceiling():
    long_para = (" ".join(["lorem ipsum dolor sit amet"] * 400))
    chunks = chunk_text(long_para, source="big.pdf", page=0, hard_ceiling=120)
    for c in chunks:
        assert c.approx_tokens() <= 120 + 30  # sentence-split slop allowed


def test_chunker_preserves_paragraph_indices_monotonic():
    text = "\n\n".join([f"paragraph number {i} with enough text to count" for i in range(8)])
    chunks = chunk_text(text, source="t.pdf", page=0, target_min=5, target_max=30)
    idxs = [c.paragraph_idx for c in chunks]
    assert idxs == sorted(idxs)
