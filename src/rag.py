from typing import Generator

import anthropic

from src.config import ANTHROPIC_API_KEY, MODEL, TOP_K
from src.store import query_documents

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on provided context.
Base your answer strictly on the context below. If the context does not contain
enough information to answer, say so clearly rather than guessing.\
"""


def rag_query(question: str, top_k: int = TOP_K) -> Generator[str, None, None]:
    """
    Retrieve relevant chunks for `question`, then stream Claude's answer.
    Yields text tokens as they arrive.
    """
    results = query_documents(question, top_k)

    if not results:
        yield "No documents found in the knowledge base. Please ingest some documents first."
        return

    context_blocks = [
        f"[Source: {meta['source']} — chunk {meta['chunk']}]\n{doc}"
        for doc, meta in results
    ]
    context = "\n\n---\n\n".join(context_blocks)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            yield text
