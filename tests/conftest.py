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


def _make_collection(chroma_path: str, name: str) -> chromadb.Collection:
    client = chromadb.EphemeralClient()  # in-memory, no disk I/O
    return client.get_or_create_collection(
        name=name,
        embedding_function=_WordHashEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture(autouse=True)
def lightweight_store(monkeypatch):
    """
    Auto-applied to every test.
    Replaces _get_collection() so tests never touch disk or load an ML model.
    Each test gets its own isolated in-memory collection.
    """
    import src.store as store

    monkeypatch.setattr(
        store,
        "_get_collection",
        lambda: _make_collection(store.CHROMA_PATH, store.COLLECTION_NAME),
    )
