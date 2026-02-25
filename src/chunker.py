def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring sentence/paragraph boundaries."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Try to break at a natural boundary if not at end of text
        if end < len(text):
            for separator in ("\n\n", "\n", ". ", " "):
                pos = chunk.rfind(separator)
                if pos > chunk_size // 2:
                    end = start + pos + len(separator)
                    chunk = text[start:end]
                    break

        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap

    return [c for c in chunks if c.strip()]
