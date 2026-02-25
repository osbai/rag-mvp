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
├── docs/             # your input documents (.txt, .pdf) — put files here
├── data/             # ChromaDB vector store, auto-generated (git-ignored)
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

## `docs/` vs `data/`

| Folder | Role | Managed by |
|--------|------|------------|
| `docs/` | **Input** — your raw `.txt` and `.pdf` files | You |
| `data/` | **Output** — ChromaDB's embeddings and index | Auto-generated on `rag ingest` |

`docs/` is the source of truth. `data/` is a derived cache that can be deleted and rebuilt from `docs/` at any time. This is why `data/` is git-ignored and `docs/` is not.

## Testing

Install dev dependencies first:

```bash
pip install -e ".[dev]"
```

Tests are split into three files:

| File | What it covers | API call? | ML model? |
|------|---------------|-----------|-----------|
| `tests/test_unit.py` | Chunker, ChromaDB smoke, retrieval, ingestion | No | No |
| `tests/test_embedding.py` | Real `all-MiniLM-L6-v2` embedding model | No | **Yes** (~700 MB) |
| `tests/test_integration.py` | Full RAG pipeline (Claude answer) | Yes | No |

### Unit tests (no API call, no ML model)

```bash
pytest tests/test_unit.py -v
```

Covers:
- **Chunker** — short documents terminate, content is preserved, empty input is handled
- **ChromaDB smoke** — `EphemeralClient` + lightweight embedding works end-to-end
- **Retrieval** — ingesting the logic test document and querying it returns the right chunks
- **Ingestion** — chunking and ingesting `.txt` and `.pdf` files from `docs/` works without errors

### Embedding model test (loads ML model)

Verifies the real `all-MiniLM-L6-v2` embedding model is working correctly. Loads ~700 MB of RAM and triggers a model download on first run.

```bash
pytest tests/test_embedding.py -m embedding -v
```

Covers:
- **Dimension** — each embedding vector is exactly 384 dimensions
- **Semantic similarity** — similar sentences score higher cosine similarity than unrelated ones
- **Retrieval (small doc)** — protocol document is split into 5 chunks (chunk_size=300) and mixed with 5 distractors; the model must rank the right chunks in the top-3 for the pre-workout query
- **Retrieval (large PDF)** — Dietary Guidelines PDF is chunked into 26 pieces and fully embedded; a vegetable servings query must surface the chunk stating the specific recommendation

### Integration test (calls Claude)

Runs the full RAG pipeline and checks that Claude's answer is grounded in the document rather than falling back to generic knowledge. Requires `ANTHROPIC_API_KEY` to be set. Skips automatically if the API is overloaded.

```bash
pytest tests/test_integration.py -m integration -v
```

### Logic test explained

| | Detail |
|-|--------|
| Document | `docs/logic_test/protocol_purple_jellybean.txt` |
| Question | *"What should I eat right before my workout according to your specific method?"* |
| Expected answer | "You should have 7 purple jellybeans 4 minutes before your set." |
| Failure signal | A generic answer like "A small piece of fruit or some fast-acting carbs is best." |

The test passes only if the retrieved context — and Claude's answer — contain `"purple jellybean"` and `"4 minutes"`, confirming the system is grounded by the document and not hallucinating from general fitness knowledge.

## How it works

1. **Ingest** — Documents are split into overlapping chunks at natural boundaries (paragraphs → sentences → spaces). Each chunk is embedded by ChromaDB (`all-MiniLM-L6-v2`) and stored locally.
2. **Query** — The question is embedded and the top-K most similar chunks are retrieved via cosine similarity.
3. **Generate** — Retrieved chunks are assembled into a context block and sent to `claude-opus-4-6` with adaptive thinking enabled. The answer streams token-by-token to the terminal.
