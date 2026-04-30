"""
build_db.py – Hybrid Search Veritabanı Oluşturucu (V2)
======================================================
data/ klasöründeki Wikipedia .txt dosyalarını okur, cümle-duyarlı
parçalama ile böler ve hem ChromaDB vektör veritabanına hem de
BM25 indeksine kaydeder.

V2 Değişiklikleri:
  - Cümle-duyarlı recursive chunking (sabit pencere yerine)
  - BAAI/bge-large-en-v1.5 embedding modeli (all-MiniLM-L6-v2 yerine)
  - Regex tabanlı tokenizasyon + stopword filtreleme (BM25 için)

Çıktılar:
  ./chroma_db/     – ChromaDB kalıcı veritabanı (collection: wiki_rag)
  bm25_index.pkl   – BM25Okapi indeksi + chunk listesi (pickle)

Kullanım:
  python build_db.py
"""

import os
import re
import sys
import pickle
import time

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Windows konsolunda Türkçe karakter sorunu yaşanmaması için
# stdout'u UTF-8'e zorla
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ======================================================================
# Ayarlar
# ======================================================================

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"
COLLECTION_NAME = "wiki_rag"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

CHUNK_SIZE = 500        # karakter (V1: 1000)
CHUNK_OVERLAP = 50      # karakter (V1: 100)

# Cümle-duyarlı parçalama için hiyerarşik ayraçlar
# Önce paragraf → satır → cümle → boşluk sırasıyla dener
SEPARATORS = ["\n\n", "\n", ". ", " "]

# BM25 tokenizasyonu için İngilizce stopword listesi
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


# ======================================================================
# Yardımcı fonksiyonlar
# ======================================================================

def read_txt_file(filepath: str) -> str:
    """Bir .txt dosyasını UTF-8 ile okur ve içeriğini döndürür."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    """
    Regex tabanlı tokenizasyon.
    Küçük harfe çevirir, kelime token'larını çıkarır, stopword'leri filtreler.
    Türkçe karakterleri de destekler.
    """
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûêô0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ======================================================================
# Cümle-Duyarlı Recursive Chunking (Pure Python – LangChain yok)
# ======================================================================

def split_by_separator(text: str, separator: str) -> list[str]:
    """
    Metni ayraç ile böler, ayracı parçanın sonunda tutar.
    Cümle sonu ('. ') ayracında nokta cümleye dahil edilir.
    """
    if separator == ". ":
        # Noktayı cümleyle birlikte tut
        parts = text.split(separator)
        return [p + ". " for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    else:
        parts = text.split(separator)
        return [p + separator for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def recursive_chunk(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """
    Metni hiyerarşik ayraçlarla chunk_size'a sığacak şekilde recursive böler.
    Ayraç sırası: paragraf (\\n\\n) → satır (\\n) → cümle ('. ') → boşluk (' ')
    
    LangChain RecursiveCharacterTextSplitter mantığının saf Python implementasyonu.
    """
    # Temel durum: metin zaten chunk_size'a sığıyorsa
    if len(text.strip()) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Her ayracı sırayla dene
    for i, sep in enumerate(separators):
        if sep not in text:
            continue

        pieces = split_by_separator(text, sep)
        if len(pieces) <= 1:
            continue

        # Parçaları chunk_size'a sığacak şekilde birleştir
        chunks = []
        current = ""

        for piece in pieces:
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                # Tek bir parça bile chunk_size'ı aşıyorsa, bir sonraki ayraçla recursive böl
                if len(piece) > chunk_size and i + 1 < len(separators):
                    sub_chunks = recursive_chunk(piece, chunk_size, separators[i + 1:])
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = piece

        if current.strip():
            chunks.append(current.strip())

        if chunks:
            return chunks

    # Hiçbir ayraç işe yaramadıysa, hard split (nadiren olur)
    return [text[j:j + chunk_size].strip()
            for j in range(0, len(text), chunk_size)
            if text[j:j + chunk_size].strip()]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Cümle-duyarlı parçalama + overlap ekleme.
    1. Hiyerarşik ayraçlarla recursive böl.
    2. Her chunk'ın başına önceki chunk'ın son chunk_overlap karakterini ekle.
    """
    raw_chunks = recursive_chunk(text, chunk_size, SEPARATORS)

    if len(raw_chunks) <= 1:
        return raw_chunks

    # Overlap ekle: önceki chunk'ın sonundan chunk_overlap karakter al
    final_chunks = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev = raw_chunks[i - 1]
        overlap_text = prev[-chunk_overlap:] if len(prev) >= chunk_overlap else prev
        # Kelime ortasından kesmemek için boşluk sınırına kaydır
        space_idx = overlap_text.find(" ")
        if space_idx != -1:
            overlap_text = overlap_text[space_idx + 1:]
        final_chunks.append(overlap_text + " " + raw_chunks[i])

    return final_chunks


def parse_filename(filename: str) -> dict:
    """
    Dosya isminden varlık tipi ve adını çıkarır.
    Örn: "person_albert_einstein.txt" -> {"type": "person", "name": "albert einstein"}
    """
    name_no_ext = os.path.splitext(filename)[0]  # person_albert_einstein

    # İlk alt çizgi, tipi (person/place) ayırır
    parts = name_no_ext.split("_", 1)

    if len(parts) == 2:
        entity_type = parts[0]                     # "person"
        entity_name = parts[1].replace("_", " ")   # "albert einstein"
    else:
        entity_type = "unknown"
        entity_name = name_no_ext.replace("_", " ")

    return {"type": entity_type, "name": entity_name}


# ======================================================================
# Ana işlem
# ======================================================================

def main():
    t_start = time.perf_counter()

    print("=" * 60)
    print("  Local RAG V2 – Hybrid Search Veritabanı Oluşturucu")
    print("=" * 60)
    print(f"  Embedding modeli : {EMBEDDING_MODEL_NAME}")
    print(f"  Chunk boyutu     : {CHUNK_SIZE} karakter")
    print(f"  Chunk overlap    : {CHUNK_OVERLAP} karakter")
    print(f"  Chunking yöntemi : Cümle-duyarlı recursive")
    print(f"  Tokenizasyon     : Regex + stopword filtreleme")
    print("=" * 60 + "\n")

    # ----- 1. data/ klasöründeki .txt dosyalarını bul -----
    if not os.path.isdir(DATA_DIR):
        print(f"[HATA] '{DATA_DIR}' klasörü bulunamadı!")
        return

    txt_files = sorted([
        f for f in os.listdir(DATA_DIR) if f.endswith(".txt")
    ])

    if not txt_files:
        print(f"[HATA] '{DATA_DIR}' klasöründe .txt dosyası bulunamadı!")
        return

    print(f"[INFO] {len(txt_files)} adet .txt dosyası bulundu.\n")

    # ----- 2. Dosyaları oku ve parçala (cümle-duyarlı) -----
    all_chunks = []       # chunk metinleri (orijinal – BM25 ve görüntüleme için)
    all_enriched = []     # metadata-zenginleştirilmiş metinler (embedding için)
    all_metadatas = []    # her chunk'ın metadata'sı
    all_ids = []          # ChromaDB için benzersiz ID'ler

    for filename in txt_files:
        filepath = os.path.join(DATA_DIR, filename)
        text = read_txt_file(filepath)
        meta_info = parse_filename(filename)

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.splitext(filename)[0]}_chunk_{i}"
            metadata = {
                "type": meta_info["type"],
                "name": meta_info["name"],
                "source": filename,
                "chunk_index": i,
            }

            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

            # Embedding için metadata ile zenginleştirilmiş metin oluştur
            # Bu, varlık adını embedding uzayına yerleştirir
            # Örn: "cemal paşa (person): === Military trial === ..."
            enriched = f"{meta_info['name']} ({meta_info['type']}): {chunk}"
            all_enriched.append(enriched)

        print(f"  [OK] {filename:<45} -> {len(chunks):>4} chunk")

    print(f"\n[INFO] Toplam chunk sayısı: {len(all_chunks)}")

    # Chunk boyut istatistikleri
    chunk_lengths = [len(c) for c in all_chunks]
    avg_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    print(f"[INFO] Chunk boyut istatistikleri: "
          f"min={min(chunk_lengths)}, max={max(chunk_lengths)}, "
          f"avg={avg_len:.0f} karakter")

    # ----- 3. Embedding modeli yükle -----
    print(f"\n[INFO] Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME} ...")
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[INFO] Model yüklendi. ({(time.perf_counter() - t0):.1f}s)\n")

    # ----- 4. Tüm chunk'ları vektöre çevir -----
    print("[INFO] Embedding'ler hesaplanıyor (metadata-enriched) ...")
    t0 = time.perf_counter()
    # bge-large-en-v1.5: dokümanlar için prefix gerekmez
    # Embedding'ler zenginleştirilmiş metinlerden hesaplanır (varlık adı dahil)
    # ancak ChromaDB'ye ve BM25'e orijinal chunk metinleri kaydedilir
    embeddings = model.encode(all_enriched, show_progress_bar=True, batch_size=32)
    embeddings_list = embeddings.tolist()
    print(f"[INFO] Embedding tamamlandı. ({(time.perf_counter() - t0):.1f}s)")
    print(f"[INFO] Embedding boyutu: {len(embeddings_list[0])}d\n")

    # ----- 5. ChromaDB'ye kaydet -----
    print(f"[INFO] ChromaDB oluşturuluyor: {CHROMA_DIR} ...")
    t0 = time.perf_counter()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Varsa eski collection'ı sil
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print(f"  [INFO] Eski '{COLLECTION_NAME}' collection silindi.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB tek seferde çok büyük batch kabul etmeyebilir, parçalayarak ekle
    BATCH_SIZE = 500
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_chunks))
        collection.add(
            ids=all_ids[batch_start:batch_end],
            documents=all_chunks[batch_start:batch_end],
            embeddings=embeddings_list[batch_start:batch_end],
            metadatas=all_metadatas[batch_start:batch_end],
        )

    print(f"  [OK] ChromaDB'ye {collection.count()} chunk kaydedildi. ({(time.perf_counter() - t0):.1f}s)")

    # ----- 6. BM25 indeksi oluştur ve kaydet -----
    print(f"\n[INFO] BM25 indeksi oluşturuluyor ...")
    t0 = time.perf_counter()

    # Geliştirilmiş tokenizasyon: regex + stopword filtreleme
    tokenized_corpus = [tokenize(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # Pickle ile kaydet (BM25 objesi + chunk listesi + metadata listesi)
    bm25_data = {
        "bm25": bm25_index,
        "chunks": all_chunks,
        "metadatas": all_metadatas,
        "ids": all_ids,
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"  [OK] BM25 indeksi '{BM25_PATH}' dosyasına kaydedildi. ({(time.perf_counter() - t0):.1f}s)")

    # ----- Özet -----
    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"  Toplam dosya     : {len(txt_files)}")
    print(f"  Toplam chunk     : {len(all_chunks)}")
    print(f"  Embedding modeli : {EMBEDDING_MODEL_NAME}")
    print(f"  Embedding boyutu : {len(embeddings_list[0])}d")
    print(f"  ChromaDB         : {os.path.abspath(CHROMA_DIR)}")
    print(f"  BM25 dosyası     : {os.path.abspath(BM25_PATH)}")
    print(f"  Toplam süre      : {total_time:.1f}s")
    print("=" * 60)
    print("[OK] Hybrid search veritabanı başarıyla oluşturuldu! (V2)")


if __name__ == "__main__":
    main()
