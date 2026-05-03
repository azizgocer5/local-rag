"""
build_db.py – Hybrid Search Database Builder (V2)
======================================================
Reads Wikipedia .txt files from the data/ directory,
splits them with sentence-aware chunking, and saves them
to both the ChromaDB vector database and BM25 index.

V2 Changes:
  - Sentence-aware recursive chunking (instead of fixed window)
  - BAAI/bge-large-en-v1.5 embedding model (instead of all-MiniLM-L6-v2)
  - Regex-based tokenization + stopword filtering (for BM25)

Outputs:
  ./chroma_db/     – ChromaDB persistent database (collection: wiki_rag)
  bm25_index.pkl   – BM25Okapi index + chunk list (pickle)

Usage:
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

# To prevent Turkish character issues in the Windows console
# force stdout to UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ======================================================================
# Settings
# ======================================================================

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"
COLLECTION_NAME = "wiki_rag"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

CHUNK_SIZE = 500        # characters (V1: 1000)
CHUNK_OVERLAP = 50      # characters (V1: 100)

# Hierarchical separators for sentence-aware chunking
# Tries paragraph → line → sentence → space in order
SEPARATORS = ["\n\n", "\n", ". ", " "]

# English stopword list for BM25 tokenization
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
# Helper functions
# ======================================================================

def read_txt_file(filepath: str) -> str:
    """Reads a .txt file with UTF-8 and returns its content."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    """
    Regex-based tokenization.
    Converts to lowercase, extracts word tokens, and filters stopwords.
    Also supports Turkish characters.
    """
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûêô0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ======================================================================
# Sentence-Aware Recursive Chunking (Pure Python – no LangChain)
# ======================================================================

def split_by_separator(text: str, separator: str) -> list[str]:
    """
    Splits text by separator, keeping the separator at the end of the piece.
    For end-of-sentence ('. ') separator, the dot is included in the sentence.
    """
    if separator == ". ":
        # Keep the dot with the sentence
        parts = text.split(separator)
        return [p + ". " for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    else:
        parts = text.split(separator)
        return [p + separator for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def recursive_chunk(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """
    Recursively chunks the text using hierarchical separators to fit chunk_size.
    Separator order: paragraph (\n\n) → line (\n) → sentence ('. ') → space (' ')
    
    Pure Python implementation of LangChain's RecursiveCharacterTextSplitter logic.
    """
    # Base case: if text already fits in chunk_size
    if len(text.strip()) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try each separator in order
    for i, sep in enumerate(separators):
        if sep not in text:
            continue

        pieces = split_by_separator(text, sep)
        if len(pieces) <= 1:
            continue

        # Combine pieces to fit within chunk_size
        chunks = []
        current = ""

        for piece in pieces:
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If even a single piece exceeds chunk_size, split recursively with the next separator
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

    # If no separator worked, hard split (rarely happens)
    return [text[j:j + chunk_size].strip()
            for j in range(0, len(text), chunk_size)
            if text[j:j + chunk_size].strip()]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Sentence-aware chunking + adding overlap.
    1. Split recursively with hierarchical separators.
    2. Prepend the last chunk_overlap characters of the previous chunk to each chunk.
    """
    raw_chunks = recursive_chunk(text, chunk_size, SEPARATORS)

    if len(raw_chunks) <= 1:
        return raw_chunks

    # Add overlap: take the last chunk_overlap characters from the previous chunk
    final_chunks = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev = raw_chunks[i - 1]
        overlap_text = prev[-chunk_overlap:] if len(prev) >= chunk_overlap else prev
        # Shift to space boundary to avoid cutting in the middle of a word
        space_idx = overlap_text.find(" ")
        if space_idx != -1:
            overlap_text = overlap_text[space_idx + 1:]
        final_chunks.append(overlap_text + " " + raw_chunks[i])

    return final_chunks


def parse_filename(filename: str) -> dict:
    """
    Extracts entity type and name from the filename.
    E.g.: "person_albert_einstein.txt" -> {"type": "person", "name": "albert einstein"}
    """
    name_no_ext = os.path.splitext(filename)[0]  # person_albert_einstein

    # The first underscore separates the type (person/place)
    parts = name_no_ext.split("_", 1)

    if len(parts) == 2:
        entity_type = parts[0]                     # "person"
        entity_name = parts[1].replace("_", " ")   # "albert einstein"
    else:
        entity_type = "unknown"
        entity_name = name_no_ext.replace("_", " ")

    return {"type": entity_type, "name": entity_name}


# ======================================================================
# Main process
# ======================================================================

def main():
    t_start = time.perf_counter()

    print("=" * 60)
    print("  Local RAG V2 – Hybrid Search Database Builder")
    print("=" * 60)
    print(f"  Embedding model  : {EMBEDDING_MODEL_NAME}")
    print(f"  Chunk size       : {CHUNK_SIZE} characters")
    print(f"  Chunk overlap    : {CHUNK_OVERLAP} characters")
    print(f"  Chunking method  : Sentence-aware recursive")
    print(f"  Tokenization     : Regex + stopword filtering")
    print("=" * 60 + "\n")

    # ----- 1. Find .txt files in the data/ directory -----
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] '{DATA_DIR}' directory not found!")
        return

    txt_files = sorted([
        f for f in os.listdir(DATA_DIR) if f.endswith(".txt")
    ])

    if not txt_files:
        print(f"[ERROR] No .txt files found in the '{DATA_DIR}' directory!")
        return

    print(f"[INFO] Found {len(txt_files)} .txt files.\n")

    # ----- 2. Read and chunk files (sentence-aware) -----
    all_chunks = []       # chunk texts (original – for BM25 and display)
    all_enriched = []     # metadata-enriched texts (for embedding)
    all_metadatas = []    # metadata of each chunk
    all_ids = []          # unique IDs for ChromaDB

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

            # Create a text enriched with metadata for embedding
            # This embeds the entity name into the embedding space
            # E.g.: "cemal paşa (person): === Military trial === ..."
            enriched = f"{meta_info['name']} ({meta_info['type']}): {chunk}"
            all_enriched.append(enriched)

        print(f"  [OK] {filename:<45} -> {len(chunks):>4} chunk")

    print(f"\n[INFO] Total number of chunks: {len(all_chunks)}")

    # Chunk size statistics
    chunk_lengths = [len(c) for c in all_chunks]
    avg_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    print(f"[INFO] Chunk size statistics: "
          f"min={min(chunk_lengths)}, max={max(chunk_lengths)}, "
          f"avg={avg_len:.0f} characters")

    # ----- 3. Load embedding model -----
    print(f"\n[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[INFO] Model loaded. ({(time.perf_counter() - t0):.1f}s)\n")

    # ----- 4. Convert all chunks to vectors -----
    print("[INFO] Calculating embeddings (metadata-enriched) ...")
    t0 = time.perf_counter()
    # bge-large-en-v1.5: prefix is not required for documents
    # Embeddings are calculated from enriched texts (including entity name)
    # but original chunk texts are saved to ChromaDB and BM25
    embeddings = model.encode(all_enriched, show_progress_bar=True, batch_size=32)
    embeddings_list = embeddings.tolist()
    print(f"[INFO] Embedding completed. ({(time.perf_counter() - t0):.1f}s)")
    print(f"[INFO] Embedding dimension: {len(embeddings_list[0])}d\n")

    # ----- 5. Save to ChromaDB -----
    print(f"[INFO] Creating ChromaDB: {CHROMA_DIR} ...")
    t0 = time.perf_counter()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if it exists
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print(f"  [INFO] Old '{COLLECTION_NAME}' collection deleted.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB might not accept very large batches at once, add in chunks
    BATCH_SIZE = 500
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_chunks))
        collection.add(
            ids=all_ids[batch_start:batch_end],
            documents=all_chunks[batch_start:batch_end],
            embeddings=embeddings_list[batch_start:batch_end],
            metadatas=all_metadatas[batch_start:batch_end],
        )

    print(f"  [OK] {collection.count()} chunks saved to ChromaDB. ({(time.perf_counter() - t0):.1f}s)")

    # ----- 6. Create and save BM25 index -----
    print(f"\n[INFO] Creating BM25 index ...")
    t0 = time.perf_counter()

    # Improved tokenization: regex + stopword filtering
    tokenized_corpus = [tokenize(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # Save with Pickle (BM25 object + chunk list + metadata list)
    bm25_data = {
        "bm25": bm25_index,
        "chunks": all_chunks,
        "metadatas": all_metadatas,
        "ids": all_ids,
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"  [OK] BM25 index saved to '{BM25_PATH}'. ({(time.perf_counter() - t0):.1f}s)")

    # ----- Summary -----
    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"  Total files      : {len(txt_files)}")
    print(f"  Total chunks     : {len(all_chunks)}")
    print(f"  Embedding model  : {EMBEDDING_MODEL_NAME}")
    print(f"  Embedding dim    : {len(embeddings_list[0])}d")
    print(f"  ChromaDB         : {os.path.abspath(CHROMA_DIR)}")
    print(f"  BM25 file        : {os.path.abspath(BM25_PATH)}")
    print(f"  Total time       : {total_time:.1f}s")
    print("=" * 60)
    print("[OK] Hybrid search database successfully created! (V2)")


if __name__ == "__main__":
    main()
