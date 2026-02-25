"""
Integration tests — require ANTHROPIC_API_KEY, call the Claude API.

Run with:
    pytest tests/test_integration.py -m integration
"""

from pathlib import Path
from unittest.mock import patch

import anthropic
import pytest

from src.chunker import chunk_text
from src.store import add_documents

DOCS_PATH = (
    Path(__file__).parent.parent
    / "docs/logic_test/protocol_purple_jellybean.txt"
)
QUESTION = "What should I eat right before my workout according to your specific method?"


@pytest.mark.integration
def test_rag_answer_is_grounded(tmp_path):
    """Claude's answer must cite the specific protocol, not generic fitness advice."""
    from src.rag import rag_query

    chroma_path = str(tmp_path / "chroma")

    with (
        patch("src.store.CHROMA_PATH", chroma_path),
        patch("src.store.COLLECTION_NAME", "test_integration"),
    ):
        chunks = chunk_text(DOCS_PATH.read_text())
        add_documents(chunks, source=DOCS_PATH.name)

        try:
            answer = "".join(rag_query(QUESTION, top_k=3)).lower()
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                pytest.skip(f"Anthropic API overloaded — retry later ({e})")
            raise

        assert "purple jellybean" in answer, (
            f"RAG answer did not mention 'purple jellybean'.\n"
            f"Likely hallucination or retrieval failure.\n\nAnswer:\n{answer}"
        )
        assert "4 minutes" in answer, (
            f"RAG answer did not mention '4 minutes'.\n\nAnswer:\n{answer}"
        )
