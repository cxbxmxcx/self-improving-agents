"""The lexical encoder must match the book's, or recall (and replay) diverges."""

from __future__ import annotations

from helix.metacognition.lexical import embed

TEXTS = [
    "Book a 45-minute hiring debrief with Ana Silva and Omar Haddad this week, Wednesday or Thursday.",
    "Find every start time that works for the user and all the listed attendees on every one of the days.",
    "",
    "the of and",
]


def test_matches_the_book_encoder():
    from chapters.ch05.personal.encoder import embed as book_embed

    assert embed(TEXTS) == book_embed(TEXTS)


def test_vectors_are_normalized_or_zero():
    for vector in embed(TEXTS):
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9 or norm == 0.0
