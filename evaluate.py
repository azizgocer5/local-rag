"""
evaluate.py – RAG Retrieval Evaluation Framework
=================================================
eval/ground_truth.json dosyasındaki soru-cevap çiftlerini kullanarak
hybrid search pipeline'ının retrieval kalitesini ölçer.

Metrikler:
  - Hit Rate @K  : Doğru kaynağın Top-K'da bulunma oranı
  - MRR          : Mean Reciprocal Rank
  - NDCG @K      : Normalized Discounted Cumulative Gain

Kullanım:
  python evaluate.py
  python evaluate.py --top_k 3
  python evaluate.py --no-rerank
"""

import os
import re
import sys
import json
import math
import time
import pickle
import argparse

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Windows konsolunda Türkçe karakter sorunu
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ======================================================================
# Ayarlar (app.py ile aynı)
# ======================================================================

CHROMA_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"
COLLECTION_NAME = "wiki_rag"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROUND_TRUTH_PATH = "./eval/ground_truth.json"

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
    """Regex tabanlı tokenizasyon + stopword filtreleme."""
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûêô0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ======================================================================
# Kaynak Yükleme
# ======================================================================

def load_resources():
    """Tüm kaynakları yükler: model, reranker, collection, BM25."""
    print("[INFO] Kaynaklar yükleniyor...")

    t0 = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"  Embedding modeli yüklendi. ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    print(f"  Reranker modeli yüklendi. ({time.perf_counter() - t0:.1f}s)")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"  ChromaDB collection yüklendi. ({collection.count()} chunk)")

    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_index = bm25_data["bm25"]
    bm25_chunks = bm25_data["chunks"]
    bm25_metadatas = bm25_data["metadatas"]
    bm25_ids = bm25_data["ids"]
    print(f"  BM25 indeksi yüklendi. ({len(bm25_chunks)} chunk)")

    return model, reranker, collection, bm25_index, bm25_chunks, bm25_metadatas, bm25_ids


# ======================================================================
# Hybrid Search (app.py ile aynı mantık)
# ======================================================================

def hybrid_search(query, top_k, model, collection, bm25_index,
                  bm25_chunks, bm25_metadatas, bm25_ids,
                  semantic_weight=0.9, bm25_weight=0.1, rrf_k=60):
    """Semantic + BM25 hybrid search with RRF fusion."""
    fetch_k = min(top_k * 4, len(bm25_chunks))

    query_embedding = model.encode([BGE_QUERY_PREFIX + query]).tolist()
    chroma_results = collection.query(
        query_embeddings=query_embedding, n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    semantic_rank_map = {}
    id_to_index = {uid: idx for idx, uid in enumerate(bm25_ids)}

    for rank, cid in enumerate(chroma_results["ids"][0], start=1):
        idx = id_to_index.get(cid)
        if idx is not None:
            semantic_rank_map[idx] = rank

    query_tokens = tokenize(query)
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

    bm25_rank_map = {}
    for rank, idx in enumerate(top_bm25_indices, start=1):
        idx = int(idx)
        if bm25_scores[idx] > 0:
            bm25_rank_map[idx] = rank

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
        metadata = bm25_metadatas[idx]
        results.append({
            "rank": final_rank,
            "text": bm25_chunks[idx],
            "source": metadata.get("source", ""),
            "name": metadata.get("name", ""),
            "rrf_score": rrf_scores[idx],
        })

    return results


def rerank_results(query, candidates, top_k, reranker):
    """Cross-encoder reranking."""
    if not candidates:
        return candidates

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    for i, c in enumerate(reranked[:top_k], start=1):
        c["rank"] = i

    return reranked[:top_k]


# ======================================================================
# Metrikler
# ======================================================================

def hit_rate_at_k(results: list[dict], expected_sources: list[str], k: int) -> float:
    """Retrieved sonuçlarda beklenen kaynağın bulunma oranı (0 veya 1)."""
    retrieved_sources = {r["source"] for r in results[:k]}
    return 1.0 if any(s in retrieved_sources for s in expected_sources) else 0.0


def mrr(results: list[dict], expected_sources: list[str]) -> float:
    """Mean Reciprocal Rank — doğru kaynağın sıralamasının tersi."""
    for i, r in enumerate(results, start=1):
        if r["source"] in expected_sources:
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: list[dict], expected_sources: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    dcg = 0.0
    for i, r in enumerate(results[:k], start=1):
        rel = 1.0 if r["source"] in expected_sources else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_dcg = sum(1.0 / math.log2(i + 1)
                    for i in range(1, min(len(expected_sources), k) + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ======================================================================
# Ana İşlem
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K for metrics (default: 5)")
    parser.add_argument("--no-rerank", action="store_true", help="Reranking'i devre dışı bırak")
    args = parser.parse_args()

    # Ground truth yükle
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"[HATA] Ground truth dosyası bulunamadı: {GROUND_TRUTH_PATH}")
        return

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"[INFO] {len(ground_truth)} soru yüklendi.")
    print(f"[INFO] Top-K: {args.top_k}, Reranking: {'OFF' if args.no_rerank else 'ON'}\n")

    # Kaynakları yükle
    model, reranker, collection, bm25_index, bm25_chunks, bm25_metadatas, bm25_ids = load_resources()

    # Her soru için metrik hesapla
    all_hr = []
    all_mrr = []
    all_ndcg = []
    per_difficulty = {}

    print(f"\n{'='*70}")
    print(f"  {'ID':<6} {'Difficulty':<10} {'HR@K':>6} {'MRR':>6} {'NDCG':>6}  Question")
    print(f"{'='*70}")

    for item in ground_truth:
        qid = item["id"]
        question = item["question"]
        expected_sources = item["expected_sources"]
        difficulty = item["difficulty"]

        # Hybrid search
        fetch_k = args.top_k * 4 if not args.no_rerank else args.top_k
        results = hybrid_search(
            question, fetch_k, model, collection,
            bm25_index, bm25_chunks, bm25_metadatas, bm25_ids,
        )

        # Rerank (opsiyonel)
        if not args.no_rerank:
            results = rerank_results(question, results, args.top_k, reranker)

        # Metrikler
        hr = hit_rate_at_k(results, expected_sources, args.top_k)
        m = mrr(results, expected_sources)
        n = ndcg_at_k(results, expected_sources, args.top_k)

        all_hr.append(hr)
        all_mrr.append(m)
        all_ndcg.append(n)

        if difficulty not in per_difficulty:
            per_difficulty[difficulty] = {"hr": [], "mrr": [], "ndcg": []}
        per_difficulty[difficulty]["hr"].append(hr)
        per_difficulty[difficulty]["mrr"].append(m)
        per_difficulty[difficulty]["ndcg"].append(n)

        status = "✅" if hr > 0 else "❌"
        print(f"  {qid:<6} {difficulty:<10} {hr:>5.2f}  {m:>5.3f}  {n:>5.3f}  {status} {question[:50]}")

    # Genel sonuçlar
    avg_hr = sum(all_hr) / len(all_hr) if all_hr else 0
    avg_mrr = sum(all_mrr) / len(all_mrr) if all_mrr else 0
    avg_ndcg = sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0

    print(f"\n{'='*70}")
    print(f"  RAG Retrieval Evaluation Results")
    print(f"{'='*70}")
    print(f"  Questions evaluated : {len(ground_truth)}")
    print(f"  Top-K               : {args.top_k}")
    print(f"  Reranking           : {'OFF' if args.no_rerank else 'ON'}")
    print(f"{'='*70}")
    print(f"  Hit Rate @{args.top_k}         : {avg_hr:.3f}")
    print(f"  MRR                 : {avg_mrr:.3f}")
    print(f"  NDCG @{args.top_k}             : {avg_ndcg:.3f}")
    print(f"{'='*70}")

    # Zorluk derecesine göre ayrıştırma
    print(f"\n  Per-Difficulty Breakdown:")
    print(f"  {'Difficulty':<10} {'Count':>6} {'HR@K':>8} {'MRR':>8} {'NDCG':>8}")
    print(f"  {'-'*46}")

    for diff in ["easy", "medium", "hard"]:
        if diff in per_difficulty:
            d = per_difficulty[diff]
            cnt = len(d["hr"])
            d_hr = sum(d["hr"]) / cnt
            d_mrr = sum(d["mrr"]) / cnt
            d_ndcg = sum(d["ndcg"]) / cnt
            print(f"  {diff:<10} {cnt:>6} {d_hr:>8.3f} {d_mrr:>8.3f} {d_ndcg:>8.3f}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
