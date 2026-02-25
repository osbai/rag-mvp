import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
MODEL: str = "claude-opus-4-6"

CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K: int = int(os.getenv("TOP_K", "5"))
