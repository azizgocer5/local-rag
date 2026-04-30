# Product Requirements Document – Local RAG Hybrid Search System (V2)

## 1. Overview

I'm building a local Retrieval-Augmented Generation (RAG) system that answers user questions
by searching a knowledge base of Wikipedia articles. The system uses a **hybrid search** approach—
combining dense semantic search with sparse keyword search—and generates answers via a locally
hosted LLM. Everything runs on-premise with no cloud API calls, making it suitable for
privacy-sensitive use cases.

V2 introduces a multi-stage retrieval pipeline with cross-encoder reranking, multi-query
expansion, metadata-enriched embeddings, conversation memory, and a quantitative evaluation
framework.

## 2. Problem Statement

Standard keyword search misses semantically related content, while pure vector search can
ignore exact keyword matches. A single-stage retrieval pipeline also struggles with entity
disambiguation—especially when the knowledge base contains alternate name spellings (e.g.,
"Djemal" vs. "Cemal Pasha"). I want a system that combines multiple retrieval strategies,
reranks results with a cross-encoder for precision, and feeds the top contexts to a local
LLM for grounded, hallucination-free answers.

## 3. System Architecture

The project consists of four pipeline stages:

### Stage 1 — Data Ingestion (`ingest.py`)
- Fetch plain-text Wikipedia articles via the MediaWiki API.
- Support both English and Turkish Wikipedia (fallback: EN → TR).
- Cover two entity categories: **People** (21 entities) and **Places** (25 entities).
- Save each article as a UTF-8 `.txt` file in the `data/` directory.
- Filename convention: `{type}_{snake_case_name}.txt` (e.g., `person_albert_einstein.txt`).

### Stage 2 — Database Building (`build_db.py`)
- Read all `.txt` files from `data/`.
- **Sentence-aware chunking**: Recursively split text using a hierarchical separator strategy
  (`\n\n` → `\n` → `. ` → ` `) targeting 500-char chunks with 50-char overlap.
  Implemented in pure Python—no LangChain or LlamaIndex.
- Parse entity metadata (type, name) from the filename.
- **Metadata-enriched embeddings**: Prepend entity name and type to each chunk before
  embedding (e.g., `"cemal paşa (person): === Military trial === ..."`). This anchors
  chunks to their entity in the vector space, solving name-mismatch problems.
- **Semantic Index**: Embed enriched chunks using `BAAI/bge-large-en-v1.5` (1024d) and
  store in a **ChromaDB** persistent collection (`wiki_rag`) with cosine similarity.
  Original (non-enriched) text is stored for display and BM25.
- **Keyword Index**: Tokenize all chunks using regex-based word extraction with English
  stopword removal, then build a **BM25Okapi** index via `rank_bm25`. Serialize the index
  + chunks + metadata to `bm25_index.pkl` via pickle.
- ChromaDB batch insertion (batch size: 500) to avoid memory limits.

### Stage 3 — Interactive Application (`app.py`)
- **Streamlit** web UI with chat interface.
- On each user query:
  1. **Conversation Memory**: Rewrite follow-up questions into standalone queries using
     the local LLM and recent chat history (last 3 turns).
  2. **Multi-Query Expansion**: Generate 3 query variants via local Ollama, search with
     all variants, and merge results.
  3. **Hybrid Search**: Run semantic search (ChromaDB with BGE query prefix) and BM25
     search (regex-tokenized), fuse via RRF.
  4. **Cross-Encoder Reranking**: Pass top candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2`
     to jointly score each (query, chunk) pair and select the final Top-K.
  5. **LLM Generation**: Build a system prompt with retrieved contexts and stream the
     answer from a local Ollama instance running `llama3.2`. The LLM is instructed to
     cite sources using `[Source: filename, Entity: name]` format.
- Sidebar: Top-K slider, configurable RRF weights (semantic/BM25/K), multi-query toggle,
  database stats (chunk counts).
- Context expander: Shows each retrieved chunk's rank, entity info, source file,
  RRF score, rerank score, and individual search ranks.
- Chat history with session state persistence and conversation memory.

### Stage 4 — Evaluation (`evaluate.py`)
- 40 labeled question-answer pairs across easy/medium/hard difficulties in
  `eval/ground_truth.json`.
- Retrieval metrics: **Hit Rate @K**, **MRR** (Mean Reciprocal Rank),
  **NDCG @K** (Normalized Discounted Cumulative Gain).
- Per-difficulty breakdown and reranking toggle (`--no-rerank`).
- Run: `python evaluate.py --top_k 5`

## 4. Tech Stack

| Layer              | Technology                                  |
|--------------------|---------------------------------------------|
| Data Source         | Wikipedia MediaWiki API                     |
| Embedding Model    | `BAAI/bge-large-en-v1.5` (1024d)            |
| Reranker           | `cross-encoder/ms-marco-MiniLM-L-6-v2`      |
| Vector Database    | ChromaDB (persistent, cosine)               |
| Keyword Search     | BM25Okapi (rank_bm25)                       |
| Fusion Algorithm   | Reciprocal Rank Fusion (RRF)                |
| LLM                | Ollama (`llama3.2`, local)                   |
| Frontend           | Streamlit                                   |
| Language           | Python 3.10+                                |

## 5. Data Specifications

- **People** (21): Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare,
  Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo,
  Mustafa Kemal Atatürk, İsmet İnönü, Fevzi Çakmak, Kazım Karabekir, Sabiha Gökçen,
  Ali Fuat Cebesoy, Halide Edip Adıvar, Enver Paşa, Cemal Paşa, Mithat Paşa, Kanye West.
- **Places** (25): Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu,
  Colosseum, Hagia Sophia, Statue of Liberty, Giza Necropolis, Mount Everest, Dumlupınar,
  Anıtkabir, Topkapı Sarayı, Galata Kulesi, Çanakkale, Kocatepe, Eskişehir, Tuna Nehri,
  Ümraniye, Hatay, Şişli, Selanik, Vienna, Ankara, İstanbul.

## 6. Constraints

- **100% Localhost**: NO external LLM APIs. Everything runs locally via Ollama.
- **No Heavy Frameworks**: No LangChain or LlamaIndex for core pipeline logic. All chunking,
  search orchestration, and query processing use native Python.
- **Anti-Hallucination**: The LLM is strictly prompted to answer ONLY from retrieved context
  with source citations.
- **Windows Compatible**: Handle Turkish character encoding on Windows console
  (`sys.stdout.reconfigure`).
- **Reproducible**: Fixed chunk sizes and separator hierarchies ensure deterministic chunking.
- **Cached Resources**: Streamlit `@st.cache_resource` prevents reloading models and indices
  on every interaction.

## 7. File Structure

```
local-rag/
├── ingest.py            # Stage 1: Wikipedia data fetcher
├── build_db.py          # Stage 2: Sentence-aware chunking + enriched embeddings + BM25
├── app.py               # Stage 3: Streamlit hybrid search UI with reranking & memory
├── evaluate.py          # Stage 4: Retrieval quality evaluation framework
├── requirements.txt     # Python dependencies
├── eval/
│   └── ground_truth.json  # 40 labeled Q&A pairs for evaluation
├── data/                # Raw Wikipedia text files (47 files)
├── chroma_db/           # ChromaDB persistent storage
└── bm25_index.pkl       # Serialized BM25 index + metadata
```

## 8. V2 Pipeline Flow

```
User Query
    ↓
Conversation Memory (rewrite follow-ups using chat history)
    ↓
Multi-Query Expansion (LLM generates 3 query variants)
    ↓
Hybrid Search ×4 (ChromaDB semantic + BM25 keyword per variant)
    ↓
RRF Fusion (configurable weights)
    ↓
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    ↓
Top-K Contexts → LLM Generation (with source citations)
    ↓
Streaming Answer + Context Metadata Display
```

## 9. Success Criteria

- Ingestion pulls all 46+ entities successfully.
- `build_db.py` produces a ChromaDB collection and BM25 pickle with sentence-aware chunks
  and metadata-enriched embeddings.
- Hybrid search with reranking returns the correct source document at Rank 1 for
  evaluation queries (Hit Rate @5 = 1.0, MRR = 1.0).
- The LLM generates grounded answers with source citations using only the provided context.
- The Streamlit UI streams responses in real time, supports follow-up questions via
  conversation memory, and displays retrieval metadata with rerank scores.
- Entity disambiguation works correctly (e.g., "Cemal Pasha" retrieves from
  `person_cemal_paşa.txt` despite the text using "Djemal").
