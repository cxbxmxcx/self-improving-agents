"""PDF ingestion script.

Reads every PDF in ingestion/pdfs/, extracts text with pypdf, chunks via the
helix paragraph chunker, embeds with bge-base-en-v1.5, and writes to a LanceDB
index at data/helix_corpus.lance/.

Usage:
    python ingestion/build_index.py
    python ingestion/build_index.py --pdfs path/to/pdfs --out data/helix_corpus.lance

Idempotent: re-running rebuilds the index from scratch. Drop new PDFs in,
re-run, ship the new index.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow `import helix.*` when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypdf import PdfReader

from helix.retrieval.chunker import Chunk, chunk_text
from helix.retrieval.index import open_index


def extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_num, page_text), ...] for a PDF."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"  [warn] page {i} of {path.name} failed: {exc}")
            text = ""
        if text.strip():
            pages.append((i, text))
    return pages


def build(pdfs_dir: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"Removing existing index at {out_path}")
        shutil.rmtree(out_path)

    pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdfs_dir}. Drop some in and re-run.")
        return

    print(f"Found {len(pdf_files)} PDFs in {pdfs_dir}")

    all_chunks: list[Chunk] = []
    for pdf in pdf_files:
        print(f"Reading {pdf.name}")
        pages = extract_pdf(pdf)
        for page_num, page_text in pages:
            chunks = chunk_text(page_text, source=pdf.name, page=page_num)
            all_chunks.extend(chunks)
        print(f"  {len(pages)} pages, {sum(1 for c in all_chunks if c.source == pdf.name)} chunks so far")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Embedding and writing to {out_path}")

    index = open_index(out_path)
    written = index.add_chunks(all_chunks)
    print(f"Wrote {written} rows. Building BM25 index.")
    index.build_fts()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HelixAgent retrieval corpus.")
    parser.add_argument(
        "--pdfs",
        type=Path,
        default=REPO_ROOT / "ingestion" / "pdfs",
        help="Directory of PDFs to ingest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "helix_corpus.lance",
        help="Path to write the LanceDB index.",
    )
    args = parser.parse_args()
    build(args.pdfs, args.out)


if __name__ == "__main__":
    main()
