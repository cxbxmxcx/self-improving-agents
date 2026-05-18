"""Retrieval primitives.

LanceDB-backed hybrid (vector + BM25) index plus a paragraph chunker. v0 uses
this to give HelixAgent a single retrieval tool; later chapters put the
embedding model, the chunker, and the retrieval-router prompt under search.
"""
