# Technical Recommendations – What's Next for V3

This document reviews the improvements implemented in V2 and outlines the remaining
opportunities for a future V3 iteration.

## What V2 Solved

### Chunking — Fixed ✅
V1 used a naive 1000-char fixed window that split mid-sentence and mid-word. V2 replaced
this with a sentence-aware recursive splitter that respects paragraph, line, and sentence
boundaries. Chunk size was reduced to 500 chars for better retrieval precision, and the
reranker compensates for any context loss.

### Retrieval Precision — Fixed ✅
V1 passed raw RRF output directly to the LLM. V2 adds a cross-encoder reranking stage
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) that jointly scores each (query, chunk) pair.
This improved precision significantly—evaluation shows Hit Rate @5 = 1.0 and MRR = 1.0
across all 40 test questions.

### Embedding Quality — Fixed ✅
V1 used `all-MiniLM-L6-v2` (384d, English-only). V2 upgraded to `BAAI/bge-large-en-v1.5`
(1024d) with metadata-enriched embeddings that prepend entity name/type to each chunk
before encoding. This solved the entity disambiguation problem (e.g., "Cemal Pasha" vs.
"Djemal" in the text).

### BM25 Tokenization — Fixed ✅
V1 used `str.lower().split()` which kept stopwords and punctuation in tokens. V2 uses
regex-based word extraction with a curated English stopword list, improving keyword
matching quality.

### Query Understanding — Fixed ✅
V1 searched with the raw user query. V2 adds multi-query expansion (3 LLM-generated
variants) and conversation memory (follow-up questions are rewritten into standalone
queries using chat history).

### Evaluation — Fixed ✅
V1 had no metrics. V2 includes an evaluation framework (`evaluate.py`) with Hit Rate,
MRR, and NDCG metrics on 40 labeled questions across easy/medium/hard difficulties.

### Grounding — Fixed ✅
V1 had a basic "answer from context" instruction. V2 enforces explicit source citations
in `[Source: filename, Entity: name]` format and instructs the LLM to match the query
language.

## Remaining Opportunities for V3

### 1. Multilingual Support

The knowledge base includes Turkish entities with text from Turkish Wikipedia, but the
current pipeline is English-optimized:

- **Embedding model**: `bge-large-en-v1.5` is English-only. For true multilingual support,
  upgrade to `intfloat/multilingual-e5-large` or `BAAI/bge-m3`.
- **BM25 tokenization**: The regex tokenizer handles Turkish characters but doesn't
  account for Turkish morphology (agglutinative structure). Consider language-aware
  tokenization with spaCy's Turkish model.
- **Language-routed collections**: Separate EN and TR chunks into different ChromaDB
  collections and route queries to the appropriate collection based on detected language.

### 2. HyDE (Hypothetical Document Embeddings)

Multi-query expansion generates variant *questions*, but HyDE generates a hypothetical
*answer* and embeds that for search. Since answers are structurally closer to documents
than questions are, this can improve recall on factoid queries. This could be offered as
a toggle alongside multi-query.

### 3. Parent-Child Chunk Architecture

The current system retrieves 500-char chunks—precise but sometimes lacking context.
A parent-child approach would:
- Index **small chunks** (200 chars) for high-precision retrieval
- Store a mapping to **parent chunks** (1000 chars)
- Send the parent to the LLM for richer context

This provides the best of both worlds: precision in retrieval, context in generation.

### 4. Adaptive RRF Weight Tuning

The sidebar sliders let users manually tune RRF weights, but optimal weights vary by
query type. A learned approach could:
- Analyze query characteristics (entity mention, factoid vs. comparison, etc.)
- Automatically select weights (e.g., favor BM25 for exact-match queries)
- Use the evaluation framework to empirically optimize on the labeled dataset

### 5. Confidence Scoring & Fallback

When retrieved contexts have low RRF and rerank scores, the system should express
uncertainty rather than forcing an answer. Implement:
- A confidence threshold on rerank scores
- A "low confidence" warning in the UI when scores are below threshold
- A fallback response: "I found some potentially relevant information, but I'm not
  confident it directly answers your question."

### 6. Agentic RAG

Transform the pipeline from single-shot to multi-step:
- **Self-reflection**: After retrieval, have the LLM evaluate whether the context is
  sufficient. If not, trigger a refined search with different keywords.
- **Multi-hop reasoning**: Decompose complex questions into sub-questions, retrieve
  for each, and synthesize.
- **Tool selection**: Let the LLM decide whether to search, ask for clarification,
  or refuse to answer.

### 7. Generation Metrics

The evaluation framework currently measures retrieval quality only. Add generation
metrics:
- **Faithfulness**: Does the answer only use context information? (LLM-as-judge)
- **Answer Relevancy**: Does the answer address the question?
- **Citation Accuracy**: Are the cited sources actually the ones that contain the
  claimed information?

## Summary

| Area                  | V1 Status                  | V2 Status                            | V3 Opportunity                       |
|-----------------------|----------------------------|--------------------------------------|--------------------------------------|
| Chunking              | Fixed 1000-char window     | ✅ Sentence-aware recursive          | Parent-child architecture            |
| Embedding model       | `all-MiniLM-L6-v2` (384d)  | ✅ `bge-large-en-v1.5` (1024d)       | Multilingual model                   |
| Entity disambiguation | Broken                     | ✅ Metadata-enriched embeddings      | —                                    |
| Post-retrieval        | None                       | ✅ Cross-encoder reranking           | —                                    |
| Query handling        | Direct embedding           | ✅ Multi-query + conversation memory | HyDE                                 |
| BM25 tokenization     | Naive split                | ✅ Regex + stopwords                 | Language-aware (spaCy)               |
| RRF weights           | Hardcoded                  | ✅ Configurable sidebar              | Adaptive/learned weights             |
| Grounding             | Basic prompt               | ✅ Source citations                  | Confidence scoring                   |
| Evaluation            | None                       | ✅ Hit Rate, MRR, NDCG               | Generation metrics (RAGAS)           |
| Architecture          | Single-shot pipeline       | Single-shot + memory                 | Agentic multi-step RAG              |
| Multilingual          | English only               | English only                         | Multilingual embeddings + tokenizer  |
