# RAG MVP

A minimal Retrieval-Augmented Generation (RAG) pipeline using **Claude** (Anthropic) and **ChromaDB**.

Ingest local documents (`.txt` / `.pdf`), then ask questions — relevant chunks are retrieved from the vector store and passed as context to Claude, which streams a grounded answer.

## Stack

| Component | Library |
|-----------|---------|
| LLM | `anthropic` — `claude-opus-4-6` with adaptive thinking |
| Vector store | `chromadb` (persistent, local) |
| Embeddings | ChromaDB default (`all-MiniLM-L6-v2`, runs locally) |
| PDF parsing | `pypdf` |
| CLI | `typer` + `rich` |

## Project structure

```
rag-mvp/
├── src/
│   ├── config.py     # settings loaded from .env
│   ├── chunker.py    # overlapping text splitter
│   ├── store.py      # ChromaDB read/write
│   ├── rag.py        # retrieve → prompt → stream
│   └── main.py       # CLI commands
├── data/             # ChromaDB persists here (git-ignored)
├── tests/
├── .env.example
└── pyproject.toml
```

## Setup

**Requirements:** Python 3.11+

```bash
git clone <repo-url> && cd rag-mvp

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Ingest documents

```bash
# Single file
rag ingest ./docs/report.pdf

# Entire directory (all .txt and .pdf files)
rag ingest ./docs/

# Custom chunking
rag ingest ./docs/ --chunk-size 800 --overlap 150
```

### Query

```bash
rag query "What are the key findings?"
rag query "Summarize the methodology" --top-k 8
```

### Stats

```bash
rag stats
```

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required.** Your Anthropic API key |
| `CHROMA_PATH` | `./data/chroma` | Where ChromaDB stores its data |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |

## How it works

1. **Ingest** — Documents are split into overlapping chunks at natural boundaries (paragraphs → sentences → spaces). Each chunk is embedded by ChromaDB (`all-MiniLM-L6-v2`) and stored locally.
2. **Query** — The question is embedded and the top-K most similar chunks are retrieved via cosine similarity.
3. **Generate** — Retrieved chunks are assembled into a context block and sent to `claude-opus-4-6` with adaptive thinking enabled. The answer streams token-by-token to the terminal.
