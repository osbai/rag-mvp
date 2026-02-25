"""
Embedding model tests — load the real all-MiniLM-L6-v2 model via ChromaDB.

These tests are intentionally separate from unit tests because they require
~700 MB of RAM and trigger a model download on first run.

Run with:
    pytest tests/test_embedding.py -m embedding -v
"""

import math
from pathlib import Path

import chromadb
import pytest
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DOCS_PATH = (
    Path(__file__).parent.parent
    / "docs/logic_test/protocol_purple_jellybean.txt"
)
DIETARY_PDF = Path(__file__).parent.parent / "docs" / "Dietary Guidelines For Americans.pdf"
QUESTION = "What should I eat right before my workout according to your specific method?"


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


@pytest.mark.embedding
def test_embedding_dimension():
    """DefaultEmbeddingFunction must produce 384-dim vectors (all-MiniLM-L6-v2)."""
    ef = DefaultEmbeddingFunction()
    vectors = ef(["Hello, world!", "Another sentence."])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert any(v != 0.0 for v in vectors[0]), "Embedding vector is all zeros"


@pytest.mark.embedding
def test_embedding_semantic_similarity():
    """Similar sentences must score higher cosine similarity than dissimilar ones."""
    ef = DefaultEmbeddingFunction()
    similar_a = "The cat sat on the mat."
    similar_b = "A cat is resting on a rug."
    unrelated = "The stock market crashed today."

    vecs = ef([similar_a, similar_b, unrelated])

    sim_close = _cosine_sim(vecs[0], vecs[1])
    sim_far = _cosine_sim(vecs[0], vecs[2])

    assert sim_close > sim_far, (
        f"Expected similar sentences to score higher.\n"
        f"  similar pair:   {sim_close:.4f}\n"
        f"  unrelated pair: {sim_far:.4f}"
    )


DISTRACTORS = [
    "Staying hydrated is essential during exercise. Drink water before, during, and after your workout.",
    "A balanced diet rich in protein, healthy fats, and complex carbohydrates supports long-term athletic performance.",
    "Sleep is the most underrated recovery tool. Aim for 7–9 hours per night to maximise muscle repair.",
    "Cardiovascular training improves heart health and increases endurance capacity over time.",
    "Stretching after exercise reduces soreness and helps maintain flexibility as you age.",
]


@pytest.mark.embedding
def test_retrieval_with_real_embeddings():
    """The real embedding model must rank the correct chunk above distractors.

    The protocol document is split into 5 chunks (chunk_size=300) so the model must
    actually rank — not just return the only document. Combined with 5 unrelated
    distractor chunks, the collection has 10 docs total and retrieval top-3 must
    include the chunk containing the specific pre-workout protocol.
    """
    from src.chunker import chunk_text

    chunks = chunk_text(DOCS_PATH.read_text(), chunk_size=300, overlap=50)

    ef = DefaultEmbeddingFunction()
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        name="test_retrieval_real",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_docs = DISTRACTORS + chunks
    col.upsert(
        documents=all_docs,
        ids=[str(i) for i in range(len(all_docs))],
    )

    results = col.query(query_texts=[QUESTION], n_results=3)
    top_docs = " ".join(results["documents"][0]).lower()

    assert "purple jellybean" in top_docs, (
        f"Real embedding retrieval missed 'purple jellybean' in top-3.\nGot:\n{top_docs[:600]}"
    )
    assert "4 minutes" in top_docs, (
        f"Real embedding retrieval missed '4 minutes' in top-3.\nGot:\n{top_docs[:600]}"
    )


@pytest.mark.embedding
def test_retrieval_large_pdf():
    """Retrieval works correctly across the 26 chunks produced from a real PDF.

    The Dietary Guidelines PDF is loaded, chunked, and fully embedded. A query
    about vegetable servings must surface the chunk that states the specific
    recommendation (3 servings per day).
    """
    from src.chunker import chunk_text
    from src.main import _load_text

    chunks = chunk_text(_load_text(DIETARY_PDF))
    assert len(chunks) > 10, f"Expected many chunks from PDF, got {len(chunks)}"

    ef = DefaultEmbeddingFunction()
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        name="test_retrieval_pdf",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    col.upsert(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
    )

    results = col.query(
        query_texts=["How many servings of vegetables and fruits should I eat per day?"],
        n_results=3,
    )
    top_docs = " ".join(results["documents"][0]).lower()

    assert "vegetable" in top_docs, (
        f"Expected 'vegetable' in top-3 results.\nGot:\n{top_docs[:600]}"
    )
    assert "3 servings" in top_docs, (
        f"Expected '3 servings' (the guideline recommendation) in top-3 results.\nGot:\n{top_docs[:600]}"
    )
