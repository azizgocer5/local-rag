"""
build_db.py – Hybrid Search Veritabanı Oluşturucu
==================================================
data/ klasöründeki Wikipedia .txt dosyalarını okur, parçalar ve
hem ChromaDB vektör veritabanına hem de BM25 indeksine kaydeder.

Çıktılar:
  ./chroma_db/     – ChromaDB kalıcı veritabanı (collection: wiki_rag)
  bm25_index.pkl   – BM25Okapi indeksi + chunk listesi (pickle)

Kullanım:
  python build_db.py
"""

import os
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

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000       # karakter
CHUNK_OVERLAP = 100     # karakter


# ======================================================================
# Yardımcı fonksiyonlar
# ======================================================================

def read_txt_file(filepath: str) -> str:
    """Bir .txt dosyasını UTF-8 ile okur ve içeriğini döndürür."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Metni sabit karakter boyutlu parçalara böler.
    Her parça bir öncekiyle chunk_overlap karakter örtüşür.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Boş veya sadece boşluk olan parçaları atla
        if chunk.strip():
            chunks.append(chunk)

        # Son parçaya ulaştıysak döngüden çık
        if end >= len(text):
            break

        start += chunk_size - chunk_overlap

    return chunks


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

    # ----- 2. Dosyaları oku ve parçala -----
    all_chunks = []       # chunk metinleri
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

        print(f"  [OK] {filename:<45} -> {len(chunks):>4} chunk")

    print(f"\n[INFO] Toplam chunk sayısı: {len(all_chunks)}")

    # ----- 3. Embedding modeli yükle -----
    print(f"\n[INFO] Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME} ...")
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[INFO] Model yüklendi. ({(time.perf_counter() - t0):.1f}s)\n")

    # ----- 4. Tüm chunk'ları vektöre çevir -----
    print("[INFO] Embedding'ler hesaplanıyor ...")
    t0 = time.perf_counter()
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=64)
    embeddings_list = embeddings.tolist()
    print(f"[INFO] Embedding tamamlandı. ({(time.perf_counter() - t0):.1f}s)\n")

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

    # Basit tokenizasyon: küçük harfe çevir, boşluklardan böl
    tokenized_corpus = [chunk.lower().split() for chunk in all_chunks]
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
    print(f"  Toplam dosya   : {len(txt_files)}")
    print(f"  Toplam chunk   : {len(all_chunks)}")
    print(f"  ChromaDB       : {os.path.abspath(CHROMA_DIR)}")
    print(f"  BM25 dosyası   : {os.path.abspath(BM25_PATH)}")
    print(f"  Toplam süre    : {total_time:.1f}s")
    print("=" * 60)
    print("[OK] Hybrid search veritabanı başarıyla oluşturuldu!")


if __name__ == "__main__":
    main()
