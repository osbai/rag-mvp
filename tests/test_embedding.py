"""
Embedding model tests — load the real all-MiniLM-L6-v2 model via ChromaDB.

These tests are intentionally separate from unit tests because they require
~700 MB of RAM and trigger a model download on first run.

Run with:
    pytest tests/test_embedding.py -m embedding -v
"""

import math

import pytest


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


@pytest.mark.embedding
def test_embedding_dimension():
    """DefaultEmbeddingFunction must produce 384-dim vectors (all-MiniLM-L6-v2)."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    vectors = ef(["Hello, world!", "Another sentence."])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert any(v != 0.0 for v in vectors[0]), "Embedding vector is all zeros"


@pytest.mark.embedding
def test_embedding_semantic_similarity():
    """Similar sentences must score higher cosine similarity than dissimilar ones."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

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
