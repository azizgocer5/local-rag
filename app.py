"""
app.py – Local RAG Hybrid Search Streamlit App (V2)
==========================================================
Performs Hybrid Search using ChromaDB (semantic) + BM25 (lexical),
merges results with RRF, reranks with Cross-Encoder
and generates LLM response via Ollama (llama3.2).

V2 Changes:
  - BAAI/bge-large-en-v1.5 embedding model (query prefix support)
  - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  - Multi-query expansion (query variants with Ollama)
  - Conversation memory (for follow-up questions)
  - Configurable RRF weights (adjustable from sidebar)
  - Source citations (showing sources in LLM responses)
  - Improved BM25 tokenization (regex + stopword)

Usage:
  streamlit run app.py
"""

import re
import sys
import json
import time
import pickle

import numpy as np
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# Turkish character issue in Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ======================================================================
# Settings
# ======================================================================

CHROMA_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"
COLLECTION_NAME = "wiki_rag"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "llama3.2"

# Rerank: how many times more candidates to fetch from hybrid search
RERANK_FETCH_MULTIPLIER = 4

# Conversation memory: how many last turns to include
MAX_HISTORY_TURNS = 3

SYSTEM_PROMPT = """You are an expert AI assistant.
Answer the user's QUESTION using ONLY the provided CONTEXT.
If the answer is not contained in the context, say "I don't know based on the provided documents."

RULES:
1. Use ONLY information from the CONTEXT below. Do NOT use prior knowledge.
2. After each claim, cite the source using [Source: filename, Entity: name] format.
3. If multiple contexts support a claim, cite all of them.
4. Keep your answer concise, accurate, and well-structured.
5. ALWAYS respond in the same language as the user's question.

CONTEXT:
"""

MULTI_QUERY_PROMPT = """You are a search query generator. Given a user question, generate exactly 3 alternative search queries that would help find relevant information. Each variant should use different keywords or phrasing.

Respond ONLY with a JSON array of 3 strings, nothing else.

User question: {question}"""

REWRITE_PROMPT = """Given this conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that includes all necessary context.

Conversation history:
{history}

Follow-up question: {question}

Rewrite the question as a standalone search query. Output ONLY the rewritten question, nothing else."""

# BM25 stopword list
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "whom", "where", "when", "how",
    "not", "no", "nor", "as", "if", "then", "than", "so", "just", "also",
    "very", "too", "only", "own", "same", "such", "into", "over", "after",
    "before", "between", "under", "above", "up", "down", "out", "off",
    "about", "each", "every", "all", "both", "few", "more", "most", "other",
    "some", "any", "many", "much", "here", "there",
})


def tokenize(text: str) -> list[str]:
    """Regex-based tokenization + stopword filtering."""
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûêô0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ======================================================================
# Resource Loading (Cache)
# ======================================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker():
    """Loads and caches the Cross-encoder reranker model."""
    return CrossEncoder(RERANKER_MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def load_chroma():
    """Loads ChromaDB client and collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    """Loads BM25 index and chunk/metadata lists from pickle."""
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"], data["metadatas"], data["ids"]


# ======================================================================
# Query Expansion & Rewriting
# ======================================================================

def rewrite_query_with_context(question: str, history: list[dict]) -> str:
    """
    Rewrites follow-up questions into independent questions using chat history.
    E.g.: "He" → "Albert Einstein" (based on history)
    """
    if not history:
        return question

    # Get the last N turns
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    history_text = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]
        history_text += f"{role}: {content}\n"

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                history=history_text, question=question
            )}],
            stream=False,
        )
        rewritten = response["message"]["content"].strip()
        if rewritten and len(rewritten) < 500:
            return rewritten
    except Exception:
        pass

    return question


def generate_query_variants(question: str) -> list[str]:
    """
    Generates 3 different variants of the query using Ollama.
    Returns only the original question in case of an error.
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": MULTI_QUERY_PROMPT.format(question=question)}],
            stream=False,
        )
        content = response["message"]["content"].strip()

        # LLM sometimes wraps with markdown code block, clean it
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
            content = content.strip()

        variants = json.loads(content)
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            return [question] + variants[:3]
    except Exception:
        pass

    return [question]


# ======================================================================
# Hybrid Search & RRF
# ======================================================================

def hybrid_search(query: str, top_k: int,
                  semantic_weight: float = 0.9,
                  bm25_weight: float = 0.1,
                  rrf_k: int = 60) -> list[dict]:
    """
    Performs Semantic (ChromaDB) + BM25 hybrid search.
    Merges results with RRF (Reciprocal Rank Fusion).
    RRF weights and K constant can be adjusted externally.
    """
    collection = load_chroma()
    model = load_embedding_model()
    bm25_index, bm25_chunks, bm25_metadatas, bm25_ids = load_bm25()

    fetch_k = min(top_k * 4, len(bm25_chunks))

    # ----- Semantic Search (ChromaDB) -----
    # bge-large-en-v1.5: prefix is required for queries
    query_embedding = model.encode([BGE_QUERY_PREFIX + query]).tolist()

    chroma_results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    # Convert ChromaDB results to ID -> rank map
    semantic_rank_map = {}
    semantic_docs = {}

    chroma_ids = chroma_results["ids"][0]
    chroma_documents = chroma_results["documents"][0]
    chroma_metadatas = chroma_results["metadatas"][0]
    chroma_distances = chroma_results["distances"][0]

    id_to_index = {uid: idx for idx, uid in enumerate(bm25_ids)}

    for rank, (cid, doc, meta, dist) in enumerate(
        zip(chroma_ids, chroma_documents, chroma_metadatas, chroma_distances), start=1
    ):
        idx = id_to_index.get(cid)
        if idx is not None:
            semantic_rank_map[idx] = rank
            semantic_docs[idx] = {"text": doc, "metadata": meta, "distance": dist}

    # ----- BM25 Search -----
    query_tokens = tokenize(query)
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

    bm25_rank_map = {}
    for rank, idx in enumerate(top_bm25_indices, start=1):
        idx = int(idx)
        if bm25_scores[idx] > 0:
            bm25_rank_map[idx] = rank

    # ----- RRF Merging -----
    all_indices = set(semantic_rank_map.keys()) | set(bm25_rank_map.keys())
    rrf_scores = {}

    for idx in all_indices:
        score = 0.0
        if idx in semantic_rank_map:
            score += semantic_weight * (1.0 / (rrf_k + semantic_rank_map[idx]))
        if idx in bm25_rank_map:
            score += bm25_weight * (1.0 / (rrf_k + bm25_rank_map[idx]))
        rrf_scores[idx] = score

    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    results = []
    for final_rank, idx in enumerate(sorted_indices, start=1):
        text = bm25_chunks[idx]
        metadata = bm25_metadatas[idx]

        results.append({
            "rank": final_rank,
            "text": text,
            "source": metadata.get("source", "unknown"),
            "name": metadata.get("name", ""),
            "type": metadata.get("type", ""),
            "rrf_score": rrf_scores[idx],
            "semantic_rank": semantic_rank_map.get(idx),
            "bm25_rank": bm25_rank_map.get(idx),
        })

    return results


# ======================================================================
# Reranking (Cross-Encoder)
# ======================================================================

def rerank_results(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Reranks results using Cross-encoder.
    Scores each (query, chunk) pair together.
    """
    if not candidates:
        return candidates

    reranker = load_reranker()

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    for i, c in enumerate(reranked[:top_k], start=1):
        c["rank"] = i

    return reranked[:top_k]


# ======================================================================
# Multi-Query Hybrid Search
# ======================================================================

def hybrid_search_multi_query(query: str, top_k: int,
                              semantic_weight: float = 0.9,
                              bm25_weight: float = 0.1,
                              rrf_k: int = 60) -> list[dict]:
    """
    Hybrid search with multi-query expansion.
    Performs separate search for each variant, merges the best results.
    """
    variants = generate_query_variants(query)

    all_candidates = {}

    for variant in variants:
        results = hybrid_search(
            variant,
            top_k=top_k * RERANK_FETCH_MULTIPLIER,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
        )
        for r in results:
            key = f"{r['source']}_chunk_{r['text'][:80]}"
            if key not in all_candidates or r["rrf_score"] > all_candidates[key]["rrf_score"]:
                all_candidates[key] = r

    merged = sorted(all_candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_k * RERANK_FETCH_MULTIPLIER]


# ======================================================================
# LLM Response Generation (Ollama – Streaming)
# ======================================================================

def build_prompt_messages(question: str, contexts: list[dict],
                          history: list[dict] = None) -> list[dict]:
    """Converts system prompt + context block + history + user question into a message list."""
    context_block = ""
    for c in contexts:
        source = c["source"]
        name = c["name"]
        rrf = c["rrf_score"]
        rerank = c.get("rerank_score", 0)
        context_block += (
            f"[CONTEXT {c['rank']} | source: {source} | entity: {name} | "
            f"rrf: {rrf:.4f} | rerank: {rerank:.4f}]\n{c['text']}\n\n"
        )

    system_content = SYSTEM_PROMPT + "\n" + context_block.strip()

    messages = [{"role": "system", "content": system_content}]

    # Conversation memory: add last N turns history
    if history:
        recent = history[-(MAX_HISTORY_TURNS * 2):]
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question.strip()})

    return messages


def stream_ollama_response(messages: list[dict]):
    """Generates streaming response from Ollama; returns a generator."""
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


# ======================================================================
# Streamlit UI
# ======================================================================

st.set_page_config(
    page_title="Wiki RAG – Hybrid Search V2",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Wiki RAG – Hybrid Search Q&A (V2)")
st.caption(
    f"Semantic ({EMBEDDING_MODEL_NAME}) + BM25 · RRF Fusion · "
    f"Cross-Encoder Reranking · Multi-Query · Ollama ({OLLAMA_MODEL})"
)


# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Search Settings")

    top_k = st.slider(
        "Top-K Contexts",
        min_value=1,
        max_value=10,
        value=5,
        help="How many context pieces should be sent to the LLM in the final stage?",
    )

    st.divider()

    # Configurable RRF Weights
    st.header("⚖️ RRF Parameters")

    semantic_weight = st.slider(
        "Semantic Weight",
        min_value=0.0, max_value=1.0, value=0.9, step=0.05,
        help="ChromaDB semantic search weight",
    )

    bm25_weight = st.slider(
        "BM25 Weight",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="BM25 keyword search weight",
    )

    rrf_k = st.slider(
        "RRF K Constant",
        min_value=1, max_value=100, value=60,
        help="Higher K → smoother ranking",
    )

    st.divider()

    # Multi-query toggle
    use_multi_query = st.checkbox(
        "🔄 Multi-Query Expansion",
        value=True,
        help="Generates 3 query variants with Ollama and searches all of them",
    )

    st.divider()

    # Database info
    st.header("📊 Database Info")
    try:
        collection = load_chroma()
        chunk_count = collection.count()
        _, bm25_chunks_ref, _, _ = load_bm25()
        st.metric("ChromaDB Chunk", chunk_count)
        st.metric("BM25 Chunk", len(bm25_chunks_ref))
    except Exception as e:
        st.error(f"Failed to load database: {e}")

    st.divider()

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ----- Chat History -----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "contexts" in msg:
            contexts = msg["contexts"]
            search_time = msg.get("search_time_ms", 0)
            query_variants = msg.get("query_variants", [])

            with st.expander(
                f"📚 Retrieved Contexts ({len(contexts)}) · ⏱️ {search_time:.0f} ms"
            ):
                if query_variants and len(query_variants) > 1:
                    st.markdown("**🔄 Query Variants:**")
                    for vi, v in enumerate(query_variants):
                        st.markdown(f"  {vi+1}. {v}")
                    st.divider()

                for c in contexts:
                    entity_icon = "👤" if c["type"] == "person" else "📍"
                    rerank_s = c.get("rerank_score")

                    sem_r = c.get("semantic_rank")
                    bm25_r = c.get("bm25_rank")
                    rank_parts = []
                    if sem_r is not None:
                        rank_parts.append(f"semantic: #{sem_r}")
                    if bm25_r is not None:
                        rank_parts.append(f"BM25: #{bm25_r}")
                    rank_str = f" · ({', '.join(rank_parts)})" if rank_parts else ""

                    rerank_str = f" · rerank: `{rerank_s:.4f}`" if rerank_s is not None else ""

                    st.markdown(
                        f"**Rank {c['rank']}** · {entity_icon} `{c['name']}` · "
                        f"📄 `{c['source']}` · "
                        f"rrf: `{c['rrf_score']:.4f}`{rerank_str}{rank_str}"
                    )
                    st.text(c["text"][:800])
                    st.divider()


# ----- Chat Input -----
if user_input := st.chat_input("Ask a question... (E.g.: What is Einstein's Nobel prize?)"):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()

        # 1) Conversation Memory: rewrite follow-up questions
        history = st.session_state["messages"][:-1]  # excluding the last added user message
        search_query = rewrite_query_with_context(user_input, history)

        # 2) Hybrid Search (multi-query or single)
        with st.spinner("🔎 Performing Hybrid Search + Reranking..."):
            try:
                if use_multi_query:
                    candidates = hybrid_search_multi_query(
                        search_query, top_k=top_k,
                        semantic_weight=semantic_weight,
                        bm25_weight=bm25_weight,
                        rrf_k=rrf_k,
                    )
                else:
                    candidates = hybrid_search(
                        search_query,
                        top_k=top_k * RERANK_FETCH_MULTIPLIER,
                        semantic_weight=semantic_weight,
                        bm25_weight=bm25_weight,
                        rrf_k=rrf_k,
                    )

                # 3) Cross-Encoder Reranking
                contexts = rerank_results(search_query, candidates, top_k=top_k)
                search_time_ms = (time.perf_counter() - t0) * 1000.0

            except Exception as e:
                st.error(f"Search error: {e}")
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"❌ Search error: {e}",
                })
                st.stop()

        # 4) LLM Response (Streaming) — including history
        messages = build_prompt_messages(user_input, contexts, history=history)

        try:
            answer = st.write_stream(stream_ollama_response(messages))
        except Exception as e:
            err_msg = (
                f"❌ Ollama connection error: {e}\n\n"
                f"Ensure Ollama is running and the `{OLLAMA_MODEL}` model "
                f"is installed:\n"
                f"```\nollama pull {OLLAMA_MODEL}\n```"
            )
            st.error(err_msg)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": err_msg,
            })
            st.stop()

        # 5) Context expander
        with st.expander(
            f"📚 Retrieved Contexts ({len(contexts)}) · ⏱️ {search_time_ms:.0f} ms"
        ):
            for c in contexts:
                entity_icon = "👤" if c["type"] == "person" else "📍"
                rerank_s = c.get("rerank_score")

                sem_r = c.get("semantic_rank")
                bm25_r = c.get("bm25_rank")
                rank_parts = []
                if sem_r is not None:
                    rank_parts.append(f"semantic: #{sem_r}")
                if bm25_r is not None:
                    rank_parts.append(f"BM25: #{bm25_r}")
                rank_str = f" · ({', '.join(rank_parts)})" if rank_parts else ""

                rerank_str = f" · rerank: `{rerank_s:.4f}`" if rerank_s is not None else ""

                st.markdown(
                    f"**Rank {c['rank']}** · {entity_icon} `{c['name']}` · "
                    f"📄 `{c['source']}` · "
                    f"rrf: `{c['rrf_score']:.4f}`{rerank_str}{rank_str}"
                )
                st.text(c["text"][:800])
                st.divider()

        # 6) Save to session state
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "contexts": contexts,
            "search_time_ms": search_time_ms,
        })


# ======================================================================
# Footer
# ======================================================================

st.divider()
st.caption(
    "💡 V2: Sentence-aware chunking · BGE-Large embeddings · "
    "Cross-encoder reranking · Multi-query expansion · "
    "Conversation memory · Source citations"
)

# Run:
# streamlit run app.py
