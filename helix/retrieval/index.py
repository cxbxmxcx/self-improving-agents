"""LanceDB-backed hybrid retrieval.

One open_index() function for build-time and query-time use. The index stores
text plus a 768-dim vector from BAAI/bge-base-en-v1.5 and exposes:

  - add_chunks(): write paragraphs to the index (used by the ingestion script)
  - search(): hybrid (vector + BM25) query, returns top-k chunks

The embedding model loads on first use and stays cached. v0 sticks with a
single model; Ch 3 puts the embedding choice under search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lancedb
import pyarrow as pa

from helix.retrieval.chunker import Chunk

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768

_embedder = None


def _get_embedder():
    """Lazy-load the sentence-transformers model so import-time stays fast."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    model = _get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


@dataclass
class SearchHit:
    text: str
    source: str
    page: int
    score: float


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("paragraph_idx", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        ]
    )


class RetrievalIndex:
    # Per-process LRU cache for search hits. Keyed by (query, k, mode); each
    # entry is a list[SearchHit]. Bounded at 128 entries so a long-running
    # Improver doesn't accumulate unbounded memory. Search hits are immutable
    # once retrieved (the index doesn't change underneath us), so caching by
    # query string is safe.
    _CACHE_MAXSIZE = 128

    def __init__(self, db_path: str | Path, table_name: str = "corpus") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db = lancedb.connect(str(self.db_path))
        self._table = None
        self._search_cache: dict[tuple[str, int, str], list[SearchHit]] = {}
        self._cache_order: list[tuple[str, int, str]] = []

    def _open_or_create(self):
        if self.table_name in self._db.table_names():
            return self._db.open_table(self.table_name)
        return self._db.create_table(self.table_name, schema=_schema(), mode="overwrite")

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vectors = embed(texts)
        rows = [
            {
                "id": f"{c.source}::{c.page}::{c.paragraph_idx}",
                "text": c.text,
                "source": c.source,
                "page": c.page,
                "paragraph_idx": c.paragraph_idx,
                "vector": vec,
            }
            for c, vec in zip(chunks, vectors)
        ]
        table = self._open_or_create()
        table.add(rows)
        self._table = table
        return len(rows)

    def build_fts(self) -> None:
        """Build BM25 full-text index on the text column. Run once after ingest."""
        table = self._open_or_create()
        try:
            table.create_fts_index("text", replace=True)
        except Exception:
            # FTS index may not be supported on every backend; vector search still works.
            pass

    def clear_cache(self) -> None:
        """Drop the search-result LRU cache. Call after re-ingesting the index."""
        self._search_cache.clear()
        self._cache_order.clear()

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Literal["vector", "keyword", "hybrid"] = "hybrid",
    ) -> list[SearchHit]:
        cache_key = (query, k, mode)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            # Move to most-recently-used.
            try:
                self._cache_order.remove(cache_key)
            except ValueError:
                pass
            self._cache_order.append(cache_key)
            return cached

        table = self._db.open_table(self.table_name)
        if mode == "vector":
            qvec = embed([query])[0]
            rows = table.search(qvec).limit(k).to_list()
        elif mode == "keyword":
            rows = table.search(query, query_type="fts").limit(k).to_list()
        else:
            qvec = embed([query])[0]
            try:
                rows = (
                    table.search(query_type="hybrid")
                    .vector(qvec)
                    .text(query)
                    .limit(k)
                    .to_list()
                )
            except Exception:
                rows = table.search(qvec).limit(k).to_list()

        hits: list[SearchHit] = []
        for r in rows:
            score = r.get("_relevance_score") or r.get("_score") or (1.0 - r.get("_distance", 0.0))
            hits.append(
                SearchHit(
                    text=r["text"],
                    source=r["source"],
                    page=r["page"],
                    score=float(score),
                )
            )
        # Cache write, with LRU eviction at the cap.
        self._search_cache[cache_key] = hits
        self._cache_order.append(cache_key)
        while len(self._cache_order) > self._CACHE_MAXSIZE:
            oldest = self._cache_order.pop(0)
            self._search_cache.pop(oldest, None)
        return hits


def open_index(db_path: str | Path, table_name: str = "corpus") -> RetrievalIndex:
    return RetrievalIndex(db_path=db_path, table_name=table_name)
