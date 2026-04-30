# Local RAG – Hybrid Search Q&A System (V2)

A fully local Retrieval-Augmented Generation system that combines **semantic search**
(ChromaDB + BGE-Large) with **keyword search** (BM25), reranks with a **cross-encoder**,
and generates grounded answers via a local LLM (Ollama). No data leaves your machine.

## What's New in V2

| Feature | V1 | V2 |
|---|---|---|
| **Chunking** | Fixed 1000-char window | Sentence-aware recursive splitting (500 chars) |
| **Embeddings** | `all-MiniLM-L6-v2` (384d) | `BAAI/bge-large-en-v1.5` (1024d) + metadata enrichment |
| **BM25 Tokenization** | `str.lower().split()` | Regex word extraction + stopword removal |
| **Reranking** | None | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Query Expansion** | None | Multi-query (3 LLM-generated variants) |
| **Memory** | Single-turn | Conversation memory (last 3 turns) |
| **Citations** | None | LLM cites `[Source: file, Entity: name]` |
| **RRF Weights** | Hardcoded | Configurable via sidebar sliders |
| **Evaluation** | None | Hit Rate, MRR, NDCG on 40 labeled questions |

## How It Works

```
User Query
    ↓
Conversation Memory → Rewrite follow-ups into standalone queries
    ↓
Multi-Query Expansion → 3 query variants via local Ollama
    ↓
Hybrid Search → Semantic (ChromaDB) + BM25 per variant
    ↓
RRF Fusion → Merge & rank with configurable weights
    ↓
Cross-Encoder Reranking → Jointly score (query, chunk) pairs
    ↓
Top-K Contexts → LLM generates answer with source citations
```

### Stage 1: Data Ingestion (`ingest.py`)

Fetches plain-text content from Wikipedia for 46 entities (21 people + 25 places).
Tries English Wikipedia first, falls back to Turkish Wikipedia. Each article is saved
as a `.txt` file in `data/`.

### Stage 2: Database Building (`build_db.py`)

- **Sentence-aware chunking**: Recursively splits text on `\n\n` → `\n` → `. ` → ` `
  boundaries (pure Python, no LangChain). Target: 500-char chunks, 50-char overlap.
- **Metadata-enriched embeddings**: Prepends entity name/type to each chunk before
  embedding to anchor it in vector space (e.g., `"cemal paşa (person): ..."`)
- **BGE-Large embeddings**: 1024-dimensional vectors via `BAAI/bge-large-en-v1.5`
- **Improved BM25**: Regex tokenization with English stopword removal

### Stage 3: Interactive Q&A (`app.py`)

A Streamlit web interface with:
- **Conversation memory** — follow-up questions work naturally
- **Multi-query expansion** — LLM generates search variants for better recall
- **Cross-encoder reranking** — precision layer after RRF fusion
- **Source citations** — LLM cites which document each claim comes from
- **Configurable RRF** — tune semantic/BM25 weights and K via sidebar

### Stage 4: Evaluation (`evaluate.py`)

Quantitative retrieval quality metrics on 40 labeled Q&A pairs:
- **Hit Rate @K** — correct source in Top-K
- **MRR** — Mean Reciprocal Rank
- **NDCG @K** — Normalized Discounted Cumulative Gain
- Per-difficulty breakdown (easy / medium / hard)

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally with `llama3.2`:
  ```bash
  ollama pull llama3.2
  ```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/local-rag.git
cd local-rag

# Install Python dependencies
pip install -r requirements.txt
```

> **Note:** First run will download embedding (~1.3 GB) and reranker (~80 MB) models
> from Hugging Face. Subsequent runs use the cached models.

## Usage

Run the stages in order:

```bash
# Step 1: Fetch Wikipedia articles
python ingest.py

# Step 2: Build the search indices (sentence-aware chunks + enriched embeddings)
python build_db.py

# Step 3: Launch the Streamlit app
streamlit run app.py

# Optional: Run retrieval evaluation
python evaluate.py --top_k 5
python evaluate.py --top_k 5 --no-rerank   # compare without reranking
```

The app opens in your browser. Ask questions, use follow-ups ("tell me more about him"),
and adjust RRF weights in the sidebar.

## Tech Stack

| Component          | Technology                                   |
|--------------------|----------------------------------------------|
| Data Source         | Wikipedia MediaWiki API                      |
| Embeddings         | `BAAI/bge-large-en-v1.5` (1024d)             |
| Reranker           | `cross-encoder/ms-marco-MiniLM-L-6-v2`       |
| Vector Store       | ChromaDB (persistent, cosine)                |
| Keyword Search     | BM25Okapi (rank_bm25)                        |
| Fusion             | Reciprocal Rank Fusion (RRF)                 |
| LLM                | Ollama – `llama3.2` (local)                   |
| UI                 | Streamlit                                    |

## Project Structure

```
local-rag/
├── ingest.py            # Fetches Wikipedia articles to data/
├── build_db.py          # Sentence-aware chunking + enriched embeddings + BM25
├── app.py               # Streamlit hybrid search Q&A with reranking & memory
├── evaluate.py          # Retrieval quality evaluation framework
├── requirements.txt     # Python dependencies
├── eval/
│   └── ground_truth.json  # 40 labeled Q&A pairs
├── data/                # 47 Wikipedia article text files
├── chroma_db/           # ChromaDB persistent storage
└── bm25_index.pkl       # Serialized BM25 index
```

## Configuration

Key parameters are adjustable in source files or via the Streamlit sidebar:

| Parameter                | File / UI        | Default                       | Description                        |
|--------------------------|------------------|-------------------------------|------------------------------------|
| `CHUNK_SIZE`             | build_db.py      | 500 chars                     | Target characters per chunk        |
| `CHUNK_OVERLAP`          | build_db.py      | 50 chars                      | Overlap between chunks             |
| `EMBEDDING_MODEL_NAME`   | build_db.py      | `BAAI/bge-large-en-v1.5`      | Sentence-transformers model        |
| `RERANKER_MODEL_NAME`    | app.py           | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model     |
| Semantic Weight          | Sidebar slider   | 0.9                           | RRF weight for semantic search     |
| BM25 Weight              | Sidebar slider   | 0.1                           | RRF weight for keyword search      |
| RRF K                    | Sidebar slider   | 60                            | RRF smoothing constant             |
| Multi-Query              | Sidebar checkbox | ON                            | Enable/disable query expansion     |
| Top-K                    | Sidebar slider   | 5                             | Contexts sent to LLM               |
| `OLLAMA_MODEL`           | app.py           | `llama3.2`                    | Local LLM model                    |

## Constraints

- **100% Localhost** — No external LLM APIs at runtime
- **No LangChain / LlamaIndex** — Core logic in native Python
- **Anti-Hallucination** — LLM answers only from retrieved context with citations