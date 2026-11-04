"""Retrieval primitives.

LanceDB-backed hybrid (vector + BM25) index plus a paragraph chunker. v0 uses
this to give HelixAgent a single retrieval tool. Chapter 7 puts the retrieve
tool's description under search; the embedding model and chunker stay fixed.
"""
