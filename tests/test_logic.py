"""
Logic test: verifies that the RAG pipeline is grounded by document content
rather than falling back to generic knowledge.

Test document : docs/logic_test/protocol_purple_jellybean.txt
Test question : "What should I eat right before my workout according to your specific method?"
Success       : response mentions "7 purple jellybeans" and "4 minutes"
Failure       : generic fitness advice (e.g. "a small piece of fruit")
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.chunker import chunk_text
from src.store import add_documents, query_documents

DOCS_PATH = (
    Path(__file__).parent.parent
    / "docs/logic_test/protocol_purple_jellybean.txt"
)
QUESTION = "What should I eat right before my workout according to your specific method?"


# ---------------------------------------------------------------------------
# Smoke test — ChromaDB + fake embedding (no API call, no ML model)
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
# Retrieval test (no API call)
# ---------------------------------------------------------------------------

def test_retrieval_contains_protocol(tmp_path):
    """The purple-jellybean chunk must appear in the top retrieved results."""
    chroma_path = str(tmp_path / "chroma")

    with (
        patch("src.store.CHROMA_PATH", chroma_path),
        patch("src.store.COLLECTION_NAME", "test_logic"),
    ):
        text = DOCS_PATH.read_text()
        chunks = chunk_text(text)
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
# Integration test — calls Claude (requires ANTHROPIC_API_KEY)
# Run with: pytest -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_rag_answer_is_grounded(tmp_path):
    """Claude's answer must cite the specific protocol, not generic fitness advice."""
    from src.rag import rag_query

    chroma_path = str(tmp_path / "chroma")

    with (
        patch("src.store.CHROMA_PATH", chroma_path),
        patch("src.store.COLLECTION_NAME", "test_logic"),
    ):
        text = DOCS_PATH.read_text()
        chunks = chunk_text(text)
        add_documents(chunks, source=DOCS_PATH.name)

        answer = "".join(rag_query(QUESTION, top_k=3)).lower()

        assert "purple jellybean" in answer, (
            f"RAG answer did not mention 'purple jellybean'.\n"
            f"Likely hallucination or retrieval failure.\n\nAnswer:\n{answer}"
        )
        assert "4 minutes" in answer, (
            f"RAG answer did not mention '4 minutes'.\n\nAnswer:\n{answer}"
        )
