# Corpus ingestion

The HelixAgent retrieval corpus is built from PDFs you drop into
`ingestion/pdfs/`. The build script extracts text, chunks by paragraph, embeds
with `BAAI/bge-base-en-v1.5`, and writes a LanceDB index to
`data/helix_corpus.lance/`.

## Workflow

```
1. Drop one or more PDFs into ingestion/pdfs/
2. python ingestion/build_index.py
3. The index appears at data/helix_corpus.lance/
4. Chapter scripts read it without rebuilding.
```

## What it does

- **pypdf** extracts text page by page. Scanned PDFs without an OCR layer come
  out empty; run them through OCR first.
- **`helix.retrieval.chunker`** splits text on blank lines, merges short
  paragraphs, and respects the 512-token embedding ceiling.
- **`BAAI/bge-base-en-v1.5`** runs on CPU. First invocation downloads ~440MB
  from Hugging Face into your local cache; subsequent runs are offline.
- **LanceDB** stores text plus 768-dim vectors plus a BM25 full-text index for
  hybrid search.

## Re-running

`build_index.py` is idempotent: it deletes the existing index directory and
rebuilds from scratch. Drop in new PDFs, re-run, commit the new
`data/helix_corpus.lance/` directory.

## What to put in `ingestion/pdfs/`

Anything you want HelixAgent to retrieve from. The PDFs themselves are not
tracked in git; `python ingestion/download_corpus.py` fetches the book's
default paper set, and the built index at `data/helix_corpus.lance/` ships in
the repo so chapter scripts work out of the box without either step. Readers
building their own agents replace the corpus with their own PDFs.

## Sizing

The shipped index should fit in git. Past ~100MB use `git-lfs`. Past a few
hundred MB consider distributing the corpus separately and shipping a tiny
sample instead.
