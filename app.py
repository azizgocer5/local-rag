"""
app.py – Local RAG Hybrid Search Streamlit Uygulaması
=====================================================
ChromaDB (semantic) + BM25 (lexical) ile Hybrid Search yapar,
sonuçları RRF ile birleştirir ve Ollama (llama3.2) üzerinden
LLM yanıtı üretir.

Kullanım:
  streamlit run app.py
"""

import sys
import time
import pickle

import numpy as np
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Windows konsolunda Türkçe karakter sorunu
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ======================================================================
# Ayarlar
# ======================================================================

CHROMA_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"
COLLECTION_NAME = "wiki_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

# RRF Parametreleri
SEMANTIC_WEIGHT = 0.9
BM25_WEIGHT = 0.1
RRF_K = 60

SYSTEM_PROMPT = """You are an expert AI assistant.
Answer the user's QUESTION using ONLY the provided CONTEXT.
If the answer is not contained in the context, do not guess or hallucinate; simply say "I don't know based on the provided documents".
Keep your answer concise, accurate, and well-structured.

CONTEXT:
"""


# ======================================================================
# Kaynak Yükleme (Cache)
# ======================================================================

@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def load_embedding_model():
    """SentenceTransformer modelini yükler ve cache'ler."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="ChromaDB bağlantısı kuruluyor...")
def load_chroma():
    """ChromaDB client'ını ve collection'ı yükler."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


@st.cache_resource(show_spinner="BM25 indeksi yükleniyor...")
def load_bm25():
    """BM25 indeksini ve chunk/metadata listelerini pickle'dan yükler."""
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"], data["metadatas"], data["ids"]


# ======================================================================
# Hybrid Search & RRF
# ======================================================================

def hybrid_search(query: str, top_k: int) -> list[dict]:
    """
    Semantic (ChromaDB) + BM25 hibrit araması yapar.
    Sonuçları RRF (Reciprocal Rank Fusion) ile birleştirir.
    """
    collection = load_chroma()
    model = load_embedding_model()
    bm25_index, bm25_chunks, bm25_metadatas, bm25_ids = load_bm25()

    fetch_k = min(top_k * 4, len(bm25_chunks))

    # ----- Semantic Search (ChromaDB) -----
    query_embedding = model.encode([query]).tolist()

    chroma_results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB sonuçlarını ID -> rank haritasına çevir
    semantic_rank_map = {}   # bm25_chunks indeksi -> rank
    semantic_docs = {}       # bm25_chunks indeksi -> {text, metadata, distance}

    chroma_ids = chroma_results["ids"][0]
    chroma_documents = chroma_results["documents"][0]
    chroma_metadatas = chroma_results["metadatas"][0]
    chroma_distances = chroma_results["distances"][0]

    # ChromaDB ID'leri bm25_ids'deki indekslere eşle
    id_to_index = {uid: idx for idx, uid in enumerate(bm25_ids)}

    for rank, (cid, doc, meta, dist) in enumerate(
        zip(chroma_ids, chroma_documents, chroma_metadatas, chroma_distances), start=1
    ):
        idx = id_to_index.get(cid)
        if idx is not None:
            semantic_rank_map[idx] = rank
            semantic_docs[idx] = {"text": doc, "metadata": meta, "distance": dist}

    # ----- BM25 Search -----
    query_tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

    bm25_rank_map = {}  # bm25_chunks indeksi -> rank
    for rank, idx in enumerate(top_bm25_indices, start=1):
        idx = int(idx)
        if bm25_scores[idx] > 0:
            bm25_rank_map[idx] = rank

    # ----- RRF Birleştirme -----
    all_indices = set(semantic_rank_map.keys()) | set(bm25_rank_map.keys())
    rrf_scores = {}

    for idx in all_indices:
        score = 0.0
        if idx in semantic_rank_map:
            score += SEMANTIC_WEIGHT * (1.0 / (RRF_K + semantic_rank_map[idx]))
        if idx in bm25_rank_map:
            score += BM25_WEIGHT * (1.0 / (RRF_K + bm25_rank_map[idx]))
        rrf_scores[idx] = score

    # En yüksek RRF puanına göre sırala, top_k kadar seç
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # Sonuç listesi oluştur
    results = []
    for final_rank, idx in enumerate(sorted_indices, start=1):
        text = bm25_chunks[idx]
        metadata = bm25_metadatas[idx]

        results.append({
            "rank": final_rank,
            "text": text,
            "source": metadata.get("source", "bilinmiyor"),
            "name": metadata.get("name", ""),
            "type": metadata.get("type", ""),
            "rrf_score": rrf_scores[idx],
            "semantic_rank": semantic_rank_map.get(idx),
            "bm25_rank": bm25_rank_map.get(idx),
        })

    return results


# ======================================================================
# LLM Yanıt Üretimi (Ollama – Streaming)
# ======================================================================

def build_prompt_messages(question: str, contexts: list[dict]) -> list[dict]:
    """Sistem promptu + context bloğu + kullanıcı sorusunu mesaj listesine dönüştürür."""
    context_block = ""
    for c in contexts:
        source = c["source"]
        name = c["name"]
        rrf = c["rrf_score"]
        context_block += (
            f"[CONTEXT {c['rank']} | source: {source} | entity: {name} | "
            f"rrf_score: {rrf:.4f}]\n{c['text']}\n\n"
        )

    system_content = SYSTEM_PROMPT + "\n" + context_block.strip()

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question.strip()},
    ]


def stream_ollama_response(messages: list[dict]):
    """Ollama'dan streaming yanıt üretir; bir generator döndürür."""
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
    page_title="Wiki RAG – Hybrid Search",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Wiki RAG – Hybrid Search Q&A")
st.caption(
    f"Semantic (ChromaDB) + BM25 · RRF Fusion · Ollama ({OLLAMA_MODEL}) · "
    f"{EMBEDDING_MODEL_NAME}"
)


# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Ayarlar")

    top_k = st.slider(
        "Top-K Contexts",
        min_value=1,
        max_value=10,
        value=5,
        help="Hybrid Search sonucu kaç bağlam parçası kullanılsın?",
    )

    st.divider()

    # Veritabanı bilgisi
    st.header("📊 Veritabanı Bilgisi")
    try:
        collection = load_chroma()
        chunk_count = collection.count()
        _, bm25_chunks_ref, _, _ = load_bm25()
        st.metric("ChromaDB Chunk", chunk_count)
        st.metric("BM25 Chunk", len(bm25_chunks_ref))
    except Exception as e:
        st.error(f"Veritabanı yüklenemedi: {e}")

    st.divider()

    if st.button("🗑️ Geçmişi Temizle", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ----- Chat Geçmişi -----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Asistan mesajının altında context expander'ı göster
        if msg["role"] == "assistant" and "contexts" in msg:
            contexts = msg["contexts"]
            search_time = msg.get("search_time_ms", 0)

            with st.expander(
                f"📚 Retrieved Contexts ({len(contexts)}) · ⏱️ {search_time:.0f} ms"
            ):
                for c in contexts:
                    entity_icon = "👤" if c["type"] == "person" else "📍"

                    sem_r = c.get("semantic_rank")
                    bm25_r = c.get("bm25_rank")
                    rank_parts = []
                    if sem_r is not None:
                        rank_parts.append(f"semantic: #{sem_r}")
                    if bm25_r is not None:
                        rank_parts.append(f"BM25: #{bm25_r}")
                    rank_str = f" · ({', '.join(rank_parts)})" if rank_parts else ""

                    st.markdown(
                        f"**Rank {c['rank']}** · {entity_icon} `{c['name']}` · "
                        f"📄 `{c['source']}` · "
                        f"score: `{c['rrf_score']:.4f}`{rank_str}"
                    )
                    st.text(c["text"][:800])
                    st.divider()


# ----- Chat Input -----
if user_input := st.chat_input("Bir soru sorun... (Örn: Einstein'ın Nobel ödülü nedir?)"):
    # Kullanıcı mesajını kaydet ve göster
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Asistan yanıtı
    with st.chat_message("assistant"):
        # 1) Hybrid Search
        with st.spinner("🔎 Hybrid Search yapılıyor..."):
            t0 = time.perf_counter()
            try:
                contexts = hybrid_search(user_input, top_k=top_k)
                search_time_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                st.error(f"Arama hatası: {e}")
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"❌ Arama hatası: {e}",
                })
                st.stop()

        # 2) LLM Yanıt (Streaming)
        messages = build_prompt_messages(user_input, contexts)

        try:
            answer = st.write_stream(stream_ollama_response(messages))
        except Exception as e:
            err_msg = (
                f"❌ Ollama bağlantı hatası: {e}\n\n"
                f"Ollama'nın çalıştığından ve `{OLLAMA_MODEL}` modelinin "
                f"yüklü olduğundan emin olun:\n"
                f"```\nollama pull {OLLAMA_MODEL}\n```"
            )
            st.error(err_msg)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": err_msg,
            })
            st.stop()

        # 3) Context expander
        with st.expander(
            f"📚 Retrieved Contexts ({len(contexts)}) · ⏱️ {search_time_ms:.0f} ms"
        ):
            for c in contexts:
                entity_icon = "👤" if c["type"] == "person" else "📍"

                sem_r = c.get("semantic_rank")
                bm25_r = c.get("bm25_rank")
                rank_parts = []
                if sem_r is not None:
                    rank_parts.append(f"semantic: #{sem_r}")
                if bm25_r is not None:
                    rank_parts.append(f"BM25: #{bm25_r}")
                rank_str = f" · ({', '.join(rank_parts)})" if rank_parts else ""

                st.markdown(
                    f"**Rank {c['rank']}** · {entity_icon} `{c['name']}` · "
                    f"📄 `{c['source']}` · "
                    f"score: `{c['rrf_score']:.4f}`{rank_str}"
                )
                st.text(c["text"][:800])
                st.divider()

        # 4) Session state'e kaydet
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
    "💡 Bu uygulama `data/` klasöründeki Wikipedia makalelerini kullanarak "
    "Hybrid Search (Semantic + BM25) ve RRF birleştirme ile çalışır. "
    "LLM yanıtları yerel Ollama API üzerinden üretilir."
)

# Çalıştırma:
# streamlit run app.py
