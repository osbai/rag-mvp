import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_PATH, COLLECTION_NAME, TOP_K


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks: list[str], source: str) -> int:
    """Upsert chunks into the vector store. Returns number of chunks stored."""
    collection = _get_collection()
    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def query_documents(query: str, top_k: int = TOP_K) -> list[tuple[str, dict]]:
    """Return the top-k most relevant (document, metadata) pairs for a query."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
    )
    docs: list[str] = results["documents"][0]
    metas: list[dict] = results["metadatas"][0]
    return list(zip(docs, metas))


def collection_stats() -> dict:
    """Return basic stats about the collection."""
    collection = _get_collection()
    return {"name": COLLECTION_NAME, "count": collection.count()}
