# Product Requirements Document – Local RAG Hybrid Search System

## 1. Overview

I'm building a local Retrieval-Augmented Generation (RAG) system that answers user questions
by searching a knowledge base of Wikipedia articles. The system uses a **hybrid search** approach—
combining dense semantic search with sparse keyword search—and generates answers via a locally
hosted LLM. Everything runs on-premise with no cloud API calls, making it suitable for
privacy-sensitive use cases.

## 2. Problem Statement

Standard keyword search misses semantically related content, while pure vector search can
ignore exact keyword matches. I want a system that combines both approaches to get the best
retrieval quality, and feeds the retrieved context to a local LLM for grounded,
hallucination-free answers.

## 3. System Architecture

The project consists of three sequential pipeline stages:

### Stage 1 — Data Ingestion (`ingest.py`)
- Fetch plain-text Wikipedia articles via the MediaWiki API.
- Support both English and Turkish Wikipedia (fallback: EN → TR).
- Cover two entity categories: **People** (21 entities) and **Places** (25 entities).
- Save each article as a UTF-8 `.txt` file in the `data/` directory.
- Filename convention: `{type}_{snake_case_name}.txt` (e.g., `person_albert_einstein.txt`).

### Stage 2 — Database Building (`build_db.py`)
- Read all `.txt` files from `data/`.
- **Chunk** each document using a fixed character window (1000 chars, 100 char overlap).
- Parse entity metadata (type, name) from the filename.
- **Semantic Index**: Embed all chunks using `all-MiniLM-L6-v2` (sentence-transformers) and store
  them in a **ChromaDB** persistent collection (`wiki_rag`) with cosine similarity.
- **Keyword Index**: Tokenize all chunks (lowercased whitespace split) and build a **BM25Okapi**
  index via `rank_bm25`. Serialize the index + chunks + metadata to `bm25_index.pkl` via pickle.
- ChromaDB batch insertion (batch size: 500) to avoid memory limits.

### Stage 3 — Interactive Application (`app.py`)
- **Streamlit** web UI with chat interface.
- On each user query:
  1. **Hybrid Search**: Run semantic search (ChromaDB) and BM25 search in parallel.
  2. **RRF Fusion**: Merge results using Reciprocal Rank Fusion
     (k=60, semantic weight=0.9, BM25 weight=0.1).
  3. **LLM Generation**: Build a system prompt with retrieved contexts and stream the answer
     from a local **Ollama** instance running `llama3.2`.
- Sidebar: Top-K slider, database stats (chunk counts).
- Context expander: Shows each retrieved chunk's rank, entity info, source file,
  RRF score, and individual search ranks.
- Chat history with session state persistence.

## 4. Tech Stack

| Layer             | Technology                   |
|-------------------|------------------------------|
| Data Source        | Wikipedia MediaWiki API      |
| Embedding Model   | `all-MiniLM-L6-v2`          |
| Vector Database   | ChromaDB (persistent, cosine)|
| Keyword Search    | BM25Okapi (rank_bm25)       |
| Fusion Algorithm  | Reciprocal Rank Fusion (RRF) |
| LLM               | Ollama (`llama3.2`, local)   |
| Frontend          | Streamlit                    |
| Language          | Python 3.10+                 |

## 5. Data Specifications

- **People** (21): Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare,
  Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo,
  Mustafa Kemal Atatürk, İsmet İnönü, Fevzi Çakmak, Kazım Karabekir, Sabiha Gökçen,
  Ali Fuat Cebesoy, Halide Edip Adıvar, Enver Paşa, Cemal Paşa, Mithat Paşa, Kanye West.
- **Places** (25): Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu,
  Colosseum, Hagia Sophia, Statue of Liberty, Giza Necropolis, Mount Everest, Dumlupınar,
  Anıtkabir, Topkapı Sarayı, Galata Kulesi, Çanakkale, Kocatepe, Eskişehir, Tuna Nehri,
  Ümraniye, Hatay, Şişli, Selanik, Vienna, Ankara, İstanbul.

## 6. Non-Functional Requirements

- **Fully Local**: No external API calls at runtime (Ollama runs on localhost).
- **Windows Compatible**: Handle Turkish character encoding on Windows console (`sys.stdout.reconfigure`).
- **Reproducible**: Fixed chunk sizes and overlap ensure deterministic chunking.
- **Cached Resources**: Streamlit `@st.cache_resource` prevents reloading models and indices
  on every interaction.

## 7. File Structure

```
local-rag/
├── ingest.py            # Stage 1: Wikipedia data fetcher
├── build_db.py          # Stage 2: Chunk + embed + index builder
├── app.py               # Stage 3: Streamlit hybrid search UI
├── requirements.txt     # Python dependencies
├── data/                # Raw Wikipedia text files (46 files)
├── chroma_db/           # ChromaDB persistent storage
└── bm25_index.pkl       # Serialized BM25 index + metadata
```

## 8. Success Criteria

- Ingestion pulls all 46 entities successfully.
- `build_db.py` produces a ChromaDB collection and BM25 pickle without errors.
- Hybrid search returns relevant results that combine semantic and keyword matches.
- The LLM generates grounded answers using only the provided context.
- The Streamlit UI streams responses in real time and displays retrieval metadata.
