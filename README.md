# AI Codebase Assistant

An **agentic AI system** that analyzes Python codebases using hybrid retrieval, dependency analysis, and tool-driven reasoning to assist developers in understanding unfamiliar systems.

## Overview

Production-grade Retrieval Augmented Generation system with:

- **Hybrid search** (BM25 + vector with Reciprocal Rank Fusion)
- **Cross-encoder reranking** (BGE reranker)
- **AST-based code parsing** (function-level chunking with metadata)
- **Semantic caching** (embedding similarity cache)
- **Self-healing RAG** (adaptive retrieval + regeneration)
- **Agentic tool routing** (planner + tool selection + reflection)
- **Streaming API** (token-by-token response)
- **Observability** (structured logging + Prometheus metrics)

## Architecture

```
User Query
    ↓
FastAPI (async)
    ↓
┌─────────────────────┐
│   Agent Planner     │
│   ↓                 │
│   Tool Selection:   │
│   • Code Search     │
│   • Dependencies    │
│   • Function Explain│
│   • RAG Pipeline    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Retrieval Layer    │
│  • Query Rewrite    │
│  • Multi-query      │
│  • Hybrid Search    │
│  • RRF Fusion       │
│  • Reranking        │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Quality Control    │
│  • Compression      │
│  • Verification     │
│  • Hallucination    │
│  • Confidence Score │
│  • Self-healing     │
└─────────────────────┘
    ↓
  Response
```

## Features

### Retrieval
- Hybrid search (BM25 + vector)
- Reciprocal Rank Fusion scoring
- Multi-query expansion
- Query decomposition
- Cross-encoder reranking
- Adaptive retrieval depth

### Reliability
- Answer verification
- Confidence scoring
- Hallucination detection
- Self-healing retries
- Grounding score

### Agent Capabilities
- Planner agent (decides strategy)
- Tool execution (code search, dependencies, explanation)
- Memory layer (multi-turn context)
- Reflection loop (self-evaluation)

### Performance
- Async FastAPI
- Streaming responses
- Redis cache
- Semantic cache (embedding similarity)
- Circuit breaker pattern

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector DB | Qdrant |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | BGE reranker (sentence-transformers) |
| Cache | Redis |
| Logging | structlog |
| Metrics | Prometheus |
| Container | Docker |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the server

```bash
python app/main.py
```

### 4. Ingest a codebase

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/your/python/project"}'
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the authentication flow"}'
```

### 6. Agent query (with tool routing)

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What depends on the UserService class?"}'
```

## Running with Docker

```bash
cd docker
docker-compose up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Ingest a Python codebase |
| POST | `/query` | RAG query |
| POST | `/agent` | Agentic query (planner + tools) |
| POST | `/query/stream` | Streaming RAG query |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/stats` | Cost and usage stats |

## Project Structure

```
rag-project/
├── app/
│   ├── api.py                    # FastAPI endpoints
│   ├── agent.py                  # Agentic RAG (planner + tools)
│   ├── rag_pipeline.py           # Core RAG pipeline
│   ├── cache.py                  # Redis + semantic cache
│   ├── config.py                 # Configuration
│   ├── resilience.py             # Retry, timeout, fallback
│   ├── main.py                   # Entry point
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # BM25 + vector + RRF
│   │   ├── reranker.py           # Cross-encoder reranker
│   │   ├── multi_query.py        # Query expansion
│   │   ├── query_rewriter.py     # Query rewriting
│   │   └── retrieval_orchestrator.py  # Full pipeline
│   ├── tools/
│   │   ├── tool_base.py          # Tool interface
│   │   ├── code_search.py        # Code search tool
│   │   ├── dependency_finder.py  # Dependency analysis
│   │   └── function_explainer.py # Function explanation
│   ├── prompts/
│   │   └── prompt_manager.py     # Versioned prompts
│   ├── memory/
│   │   └── memory.py             # Conversation memory
│   └── observability/
│       ├── logger.py             # Structured logging
│       └── metrics.py            # Prometheus metrics
├── ingestion/
│   ├── loaders.py                # File loading
│   ├── chunking.py               # AST-based chunking
│   └── indexing.py               # Vector + BM25 indexing
├── evaluation/
│   ├── datasets/testset.json     # Test dataset
│   ├── eval_runner.py            # Evaluation runner
│   └── metrics.py                # Eval metrics
├── benchmarks/
│   ├── latency_test.py           # Latency benchmarks
│   └── load_test.py              # Load testing
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   └── ingest_data.py            # Ingestion script
├── tests/
├── requirements.txt
├── .env.example
├── Makefile
├── README.md
└── architecture.md
```

## Evaluation Results

_Run `make eval` after ingesting a codebase to generate metrics._

| Metric | Score |
|--------|-------|
| Faithfulness | — |
| Answer Relevancy | — |
| Context Precision | — |
| Context Recall | — |

## Future Improvements

- [ ] Multi-language support (JavaScript, Java)
- [ ] Call graph visualization
- [ ] Interactive dependency explorer UI
- [ ] Fine-tuned code embeddings
- [ ] RAG evaluation dashboard
- [ ] Prompt auto-optimization
- [ ] Cost optimization engine
