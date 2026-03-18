# Architecture — AI Codebase Assistant

## System Overview

The AI Codebase Assistant is an agentic RAG system designed to help developers
understand Python codebases through intelligent retrieval and analysis.

## Design Decisions

### AST-Based Chunking (vs Text Splitting)
Text splitting ignores code structure. AST parsing extracts functions and classes
as natural semantic units with rich metadata (imports, calls, decorators, types).
This enables function-level retrieval precision that text splitting cannot achieve.

### Hybrid Search (BM25 + Vector)
Vector search alone misses exact keyword matches (function names, variable names).
BM25 alone misses semantic meaning. Hybrid combines both for optimal recall.

### Reciprocal Rank Fusion (vs Score Averaging)
Scores from BM25 and vector search are not directly comparable.
RRF uses rank positions instead of scores, making it score-agnostic.
Formula: `score = Σ 1/(k + rank)` where k=60.
This is the industry standard (Microsoft, Meta RAG systems).

### Cross-Encoder Reranking
Vector search is recall-optimized (find many candidates).
Cross-encoder reranking is precision-optimized (select the best).
The two-stage approach (retrieve 20, rerank to 5) gives both.
Typically improves answer quality 20-40%.

### Semantic Cache
Exact-match caching misses semantically identical queries.
"What is RAG?" and "What is retrieval augmented generation?" are the same question.
Embedding similarity cache (threshold 0.92) captures these.
Reduces LLM cost significantly for repeated question patterns.

### Self-Healing RAG Loop
Single-pass RAG fails silently when context is poor.
Self-healing checks confidence and grounding scores after generation.
If low → expands retrieval → regenerates → picks best answer.
This is how production RAG achieves reliability.

### Agent Architecture
Not all questions need retrieval. The agent planner decides:
- Code search (find specific functions)
- Dependency analysis (trace relationships)
- Function explanation (detailed breakdown)
- Full RAG pipeline (general questions)
This reduces latency and improves answer quality for specific query types.

## Pipeline Flow

```
1. Query arrives via FastAPI
2. Agent Planner decides strategy
3. Tool selected and executed:
   a. Code Search → hybrid retrieval → results
   b. Dependency → AST graph analysis → report
   c. Function Explain → targeted retrieval → LLM explanation
   d. RAG → full pipeline (below)
4. RAG Pipeline:
   a. Semantic cache check
   b. Query rewrite
   c. Complexity detection + decomposition
   d. Multi-query expansion
   e. Parallel hybrid retrieval (BM25 + vector)
   f. RRF fusion across queries
   g. Deduplication
   h. Cross-encoder reranking
   i. Context compression
   j. LLM generation (versioned prompts)
   k. Reflection (improve if incomplete)
   l. Verification (SUPPORTED/UNSUPPORTED)
   m. Hallucination detection
   n. Confidence + grounding scoring
   o. Self-healing retry (if scores low)
   p. Cache update
5. Response returned with confidence score + sources
```

## Quality Control Layers

| Layer | Purpose | Mechanism |
|-------|---------|-----------|
| Context Compression | Remove noise | LLM extraction |
| Verification | Check support | LLM SUPPORTED/UNSUPPORTED |
| Hallucination Detection | Catch fabrication | LLM YES/NO check |
| Confidence Score | Quantify certainty | LLM 0-100 score |
| Grounding Score | Measure faithfulness | LLM 0-100 score |
| Self-Healing | Auto-retry on low quality | Expanded retrieval + regeneration |

## Safety Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| MAX_AGENT_STEPS | 4 | Prevent infinite agent loops |
| MAX_RETRIES | 3 | Limit retry attempts |
| MAX_DOCS | 25 | Control token usage |
| MAX_TOKENS | 6000 | Limit LLM input |
| Confidence threshold | 60 | Trigger self-healing |
| Grounding threshold | 50 | Trigger re-evaluation |
