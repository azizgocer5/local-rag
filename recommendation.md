# Technical Recommendations for Improving the RAG System

This document discusses the technical limitations of the current system and
recommends improvements focused on retrieval quality, generation accuracy,
and multilingual capability.

## 1. Multilingual Support — Lessons Learned

I initially aimed for full multilingual support (English + Turkish). The knowledge
base already includes Turkish entities and the ingestion script falls back to Turkish
Wikipedia when English articles aren't available. However, several technical constraints
forced me to scope down to English-only:

- **Embedding model**: `all-MiniLM-L6-v2` is trained primarily on English data. Its
  semantic representations for Turkish text are weak, causing poor recall for
  Turkish queries.
- **BM25 tokenization**: Simple whitespace splitting doesn't handle Turkish morphology.
  Turkish is agglutinative—a single token can carry meaning that requires multiple
  English words—so keyword matching underperforms significantly.
- **LLM response language**: `llama3.2` inconsistently matches the query language,
  especially when the retrieved contexts mix English and Turkish chunks.

**Path forward**: Use a multilingual embedding model (e.g., `multilingual-e5-large`),
language-aware tokenization for BM25 (spaCy/nltk with Turkish models), and
language-routed collections to separate EN/TR chunks.

## 2. Chunking Strategy

The current fixed character window (1000 chars, 100 overlap) is simple but has clear
downsides—it can split sentences mid-word and creates chunks with uneven semantic density.

**Recommendations**:
- **Semantic chunking**: Split on sentence or paragraph boundaries to preserve meaning.
  Tools like LangChain's `RecursiveCharacterTextSplitter` with separators
  (`\n\n`, `\n`, `. `) handle this well.
- **Adaptive chunk sizing**: Shorter chunks improve retrieval precision but lose context;
  longer chunks preserve context but add noise. A parent-child approach (retrieve
  small chunks, expand to parent for LLM context) balances both.

## 3. Retrieval Quality

### Reranking
The current system uses RRF to fuse results but doesn't rerank them. Adding a
**cross-encoder reranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a
second stage after RRF fusion would significantly improve precision by scoring each
query-chunk pair jointly rather than independently.

### Query Expansion
Complex or ambiguous queries often miss relevant chunks. Techniques like **HyDE**
(Hypothetical Document Embeddings)—where the LLM first generates a hypothetical
answer, then that answer is embedded for search—can improve recall on difficult queries.

### Embedding Model
`all-MiniLM-L6-v2` is lightweight but limited. Upgrading to a stronger model like
`e5-large-v2` or `bge-large-en-v1.5` would improve semantic retrieval quality,
at the cost of higher compute and memory requirements.

## 4. RRF Tuning

The current weights (semantic: 0.9, BM25: 0.1) heavily favor semantic search. This
works for most natural language queries but underperforms for exact-match lookups
(specific names, dates, technical terms). An empirical evaluation on a labeled query
set would help calibrate these weights. The RRF constant (k=60) also affects score
distribution and could benefit from tuning.

## 5. Generation & Grounding

- **Citation support**: The LLM should cite which context chunk it drew information
  from, making answers verifiable.
- **Confidence scoring**: Track cases where retrieved contexts have low RRF scores—
  the system should express uncertainty rather than forcing an answer from weak evidence.
- **Guardrails**: Add prompt-level instructions to prevent the LLM from answering
  outside the retrieved context, reducing hallucination risk.

## Summary

| Area                  | Current Approach             | Recommended Improvement              |
|-----------------------|------------------------------|--------------------------------------|
| Chunking              | Fixed 1000-char window       | Semantic / sentence-based splitting  |
| Embedding model       | `all-MiniLM-L6-v2`          | `e5-large-v2` or multilingual model  |
| Post-retrieval        | None                         | Cross-encoder reranking              |
| Query handling        | Direct embedding             | HyDE / query expansion               |
| RRF weights           | 0.9 / 0.1 (hardcoded)       | Empirically tuned on evaluation set  |
| Multilingual          | English only                 | Multilingual embeddings + tokenizer  |
| Grounding             | Basic system prompt          | Citations + confidence scoring       |
