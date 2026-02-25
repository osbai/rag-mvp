"""
Unit tests — no API calls, no ML model loading.

Covers:
- Chunker correctness
- ChromaDB + fake embedding smoke test
- Retrieval pipeline (ingest → query)
- Ingestion and chunking of actual docs/ files
"""

from pathlib import Path
from unittest.mock import patch

from src.chunker import chunk_text
from src.main import _load_text
from src.store import add_documents, query_documents

DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_PATH = DOCS_DIR / "logic_test" / "protocol_purple_jellybean.txt"
DIETARY_PDF = DOCS_DIR / "Dietary Guidelines For Americans.pdf"
QUESTION = "What should I eat right before my workout according to your specific method?"


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def test_chunker_short_document_terminates():
    """Document shorter than chunk_size must not loop forever."""
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert chunks == ["This is a short document."]


def test_chunker_preserves_content():
    """All content must be present across chunks (no data lost)."""
    text = ("word " * 300).strip()  # ~1500 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert "word" in " ".join(chunks)


def test_chunker_empty_text():
    """Empty input must return an empty list, not crash."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# ---------------------------------------------------------------------------
# ChromaDB smoke test
# ---------------------------------------------------------------------------

def test_chromadb_smoke():
    """Verify ChromaDB EphemeralClient + WordHash embedding works end-to-end."""
    import chromadb
    from tests.conftest import _WordHashEmbedding

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        name="smoke", embedding_function=_WordHashEmbedding()
    )
    col.upsert(documents=["hello world"], ids=["1"])
    results = col.query(query_texts=["hello"], n_results=1)
    assert results["documents"] == [["hello world"]]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_retrieval_contains_protocol(tmp_path):
    """The purple-jellybean chunk must appear in the top retrieved results."""
    chroma_path = str(tmp_path / "chroma")

    with (
        patch("src.store.CHROMA_PATH", chroma_path),
        patch("src.store.COLLECTION_NAME", "test_logic"),
    ):
        chunks = chunk_text(DOCS_PATH.read_text())
        add_documents(chunks, source=DOCS_PATH.name)

        results = query_documents(QUESTION, top_k=3)
        assert results, "No results returned — collection may be empty"

        combined = " ".join(doc for doc, _ in results).lower()
        assert "purple jellybean" in combined, (
            f"Expected 'purple jellybean' in retrieved chunks.\nGot:\n{combined[:600]}"
        )
        assert "4 minutes" in combined, (
            f"Expected '4 minutes' in retrieved chunks.\nGot:\n{combined[:600]}"
        )


# ---------------------------------------------------------------------------
# Ingestion and chunking of actual docs/
# ---------------------------------------------------------------------------


def test_chunk_txt_doc():
    """Chunking the logic test .txt preserves content and produces >= 1 chunk."""
    chunks = chunk_text(DOCS_PATH.read_text())
    assert len(chunks) >= 1
    combined = " ".join(chunks).lower()
    assert "purple jellybean" in combined


def test_chunk_pdf_doc():
    """Loading and chunking a real PDF does not crash and yields >= 1 chunk."""
    text = _load_text(DIETARY_PDF)
    assert text.strip(), "PDF text extraction returned empty content"
    chunks = chunk_text(text)
    assert len(chunks) >= 1


def test_ingest_txt_doc():
    """Ingesting a .txt file stores chunks with correct count and source metadata."""
    chunks = chunk_text(DOCS_PATH.read_text())
    n = add_documents(chunks, source=DOCS_PATH.name)
    assert n == len(chunks)
    assert n >= 1

    results = query_documents("purple jellybean", top_k=3)
    assert results
    sources = {meta["source"] for _, meta in results}
    assert DOCS_PATH.name in sources


def test_ingest_pdf_doc():
    """Ingesting a PDF stores at least one chunk without errors."""
    text = _load_text(DIETARY_PDF)
    chunks = chunk_text(text)
    n = add_documents(chunks, source=DIETARY_PDF.name)
    assert n >= 1


def test_ingest_all_docs():
    """Every .txt and .pdf in docs/ can be ingested without raising an exception."""
    files = [f for f in DOCS_DIR.rglob("*") if f.suffix.lower() in {".txt", ".pdf"}]
    assert files, "No .txt or .pdf files found in docs/ — add at least one document"

    total = 0
    for file in files:
        text = _load_text(file)
        chunks = chunk_text(text)
        n = add_documents(chunks, source=file.name)
        total += n

    assert total > 0, "No chunks were ingested from any document"
