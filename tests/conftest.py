"""
Pytest configuration.

Patches src.store._get_collection() to use a lightweight word-hash embedding
instead of ChromaDB's default (which loads onnxruntime + all-MiniLM-L6-v2,
requiring ~700 MB of RAM and triggering model downloads).
"""

import pytest
import chromadb


class _WordHashEmbedding:
    """Deterministic 128-dim bag-of-words embedding. No ML model required."""

    DIM = 128

    def name(self):
        return "word-hash"

    def is_legacy(self):
        return False

    def __call__(self, input):  # noqa: A002
        return [self._embed(doc) for doc in input]

    def embed_query(self, input):  # noqa: A002
        return [self._embed(text) for text in input]

    def _embed(self, text: str):
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            vec[hash(word) % self.DIM] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


@pytest.fixture(autouse=True)
def lightweight_store(monkeypatch):
    """
    Auto-applied to every test.
    Replaces _get_collection() so tests never touch disk or load an ML model.
    One EphemeralClient per test — shared across all calls within the same test
    so add_documents and query_documents see the same in-memory collection.
    """
    import src.store as store

    client = chromadb.EphemeralClient()

    def _get_collection():
        return client.get_or_create_collection(
            name=store.COLLECTION_NAME,
            embedding_function=_WordHashEmbedding(),
            metadata={"hnsw:space": "cosine"},
        )

    monkeypatch.setattr(store, "_get_collection", _get_collection)
