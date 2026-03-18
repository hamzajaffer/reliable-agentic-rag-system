"""
FastAPI Application — async API with streaming, health checks, and metrics.

Endpoints:
    POST /query          — RAG query
    POST /agent          — Agentic query (planner + tools)
    POST /ingest         — Ingest a codebase
    GET  /health         — Health check
    GET  /metrics        — Prometheus metrics
    POST /query/stream   — Streaming response
"""

import time
import asyncio
import structlog
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import OPENAI_API_KEY
from app.observability.logger import setup_logging, log_query, log_response
from app.observability.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
    ACTIVE_QUERIES,
    DOCUMENTS_INDEXED,
    get_metrics,
    cost_tracker,
)

logger = structlog.get_logger()

# ─── Initialize Logging ──────────────────────────────────
setup_logging()

# ─── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="AI Codebase Assistant",
    description="Production-grade Agentic RAG system for understanding Python codebases",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global State ────────────────────────────────────────
# These are initialized when a codebase is ingested
_rag_pipeline = None
_agent = None
_retriever = None
_indexer = None


# ─── Request/Response Models ─────────────────────────────

class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True
    prompt_version: Optional[str] = None

class IngestRequest(BaseModel):
    path: str
    collection_name: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: int = 0
    sources: list = []
    cached: bool = False
    metadata: dict = {}

class AgentResponse(BaseModel):
    answer: str
    tool_trace: list = []
    steps: int = 0

class HealthResponse(BaseModel):
    status: str
    indexed: bool
    documents: int = 0


# ─── Health Check ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        indexed=_rag_pipeline is not None,
        documents=0,
    )


# ─── Prometheus Metrics ──────────────────────────────────

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type="text/plain",
    )


# ─── Cost Stats ──────────────────────────────────────────

@app.get("/stats")
async def stats():
    """Cost and usage statistics."""
    return cost_tracker.get_stats()


# ─── Ingest Codebase ─────────────────────────────────────

@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Ingest a Python codebase for analysis.
    Loads files, chunks with AST, builds vector + BM25 indexes.
    """
    global _rag_pipeline, _agent, _retriever, _indexer

    start = time.time()

    try:
        from ingestion.loaders import load_directory
        from ingestion.chunking import chunk_codebase
        from ingestion.indexing import CodebaseIndexer
        from app.retrieval.hybrid_retriever import HybridRetriever
        from app.retrieval.retrieval_orchestrator import RetrievalOrchestrator
        from app.cache import CacheManager
        from app.rag_pipeline import RAGPipeline
        from app.agent import CodeAgent
        from app.tools.tool_base import ToolRegistry
        from app.tools.code_search import create_code_search_tool
        from app.tools.dependency_finder import create_dependency_tool, get_analyzer
        from app.tools.function_explainer import create_function_explainer_tool
        from app.memory.memory import Memory

        # Step 1: Load files
        logger.info("ingestion_started", path=request.path)
        files = load_directory(request.path)

        if not files:
            raise HTTPException(status_code=400, detail="No Python files found in the specified path")

        # Step 2: AST chunking
        chunks = chunk_codebase(files)
        logger.info("chunking_complete", chunks=len(chunks))

        # Step 3: Index (embeddings + BM25)
        _indexer = CodebaseIndexer(request.collection_name)
        _indexer.index_chunks(chunks)

        # Step 4: Build retriever
        vectorstore = _indexer.get_vectorstore()
        bm25, bm25_texts, documents = _indexer.get_bm25()

        _retriever = HybridRetriever(vectorstore, bm25, bm25_texts, documents)

        # Step 5: Build retrieval orchestrator
        orchestrator = RetrievalOrchestrator(_retriever)

        # Step 6: Build cache
        cache = CacheManager()

        # Step 7: Build RAG pipeline
        _rag_pipeline = RAGPipeline(orchestrator, cache)

        # Step 8: Build dependency analyzer
        analyzer = get_analyzer()
        analyzer.build_from_chunks(chunks)

        # Step 9: Build agent with tools
        registry = ToolRegistry()
        registry.register(create_code_search_tool(_retriever))
        registry.register(create_dependency_tool())
        registry.register(create_function_explainer_tool(_retriever))

        # Register RAG as a tool
        from app.tools.tool_base import Tool

        async def rag_tool_func(query):
            result = await _rag_pipeline.query(query)
            return result.get("answer", "No answer")

        registry.register(Tool(
            name="rag",
            description="Full RAG pipeline — use for general code questions, explanations, and analysis",
            func=rag_tool_func,
        ))

        _agent = CodeAgent(registry, _rag_pipeline, Memory())

        elapsed = time.time() - start
        DOCUMENTS_INDEXED.set(len(chunks))

        logger.info(
            "ingestion_complete",
            files=len(files),
            chunks=len(chunks),
            elapsed=round(elapsed, 2)
        )

        return {
            "status": "success",
            "files_loaded": len(files),
            "chunks_created": len(chunks),
            "elapsed_seconds": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("ingestion_error", error=str(e))
        ERROR_COUNT.labels(endpoint="ingest", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── RAG Query ────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the codebase using the RAG pipeline.
    Returns answer with confidence score and source references.
    """
    if not _rag_pipeline:
        raise HTTPException(
            status_code=400,
            detail="No codebase indexed. Use POST /ingest first."
        )

    start = time.time()
    ACTIVE_QUERIES.inc()
    log_query(request.query)

    try:
        REQUEST_COUNT.labels(endpoint="query", status="started").inc()

        result = await _rag_pipeline.query(
            query=request.query,
            use_cache=request.use_cache,
            prompt_version=request.prompt_version,
        )

        elapsed = (time.time() - start) * 1000
        REQUEST_LATENCY.labels(endpoint="query").observe(elapsed / 1000)
        REQUEST_COUNT.labels(endpoint="query", status="success").inc()
        log_response(request.query, result.get("confidence", 0), elapsed)

        return QueryResponse(
            answer=result["answer"],
            confidence=result.get("confidence", 0),
            sources=result.get("sources", []),
            cached=result.get("cached", False),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        ERROR_COUNT.labels(endpoint="query", error_type=type(e).__name__).inc()
        logger.error("query_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_QUERIES.dec()


# ─── Agent Query ──────────────────────────────────────────

@app.post("/agent", response_model=AgentResponse)
async def agent_query(request: QueryRequest):
    """
    Query using the agentic RAG system.
    The agent decides which tools to use (search, explain, dependencies, or RAG).
    """
    if not _agent:
        raise HTTPException(
            status_code=400,
            detail="No codebase indexed. Use POST /ingest first."
        )

    start = time.time()
    ACTIVE_QUERIES.inc()

    try:
        REQUEST_COUNT.labels(endpoint="agent", status="started").inc()

        result = await _agent.run(request.query)

        elapsed = (time.time() - start) * 1000
        REQUEST_LATENCY.labels(endpoint="agent").observe(elapsed / 1000)
        REQUEST_COUNT.labels(endpoint="agent", status="success").inc()

        return AgentResponse(
            answer=result["answer"],
            tool_trace=result.get("tool_trace", []),
            steps=result.get("steps", 0),
        )

    except Exception as e:
        ERROR_COUNT.labels(endpoint="agent", error_type=type(e).__name__).inc()
        logger.error("agent_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_QUERIES.dec()


# ─── Streaming Query ─────────────────────────────────────

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming RAG query — returns tokens as they are generated.
    Lower latency perception, better UX.
    """
    if not _rag_pipeline:
        raise HTTPException(
            status_code=400,
            detail="No codebase indexed. Use POST /ingest first."
        )

    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage

    async def generate():
        try:
            # Get context via retrieval
            docs = await _rag_pipeline.retriever.retrieve(request.query, k=5)
            context = "\n\n".join(d.page_content for d in docs)

            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.2,
                streaming=True,
            )

            prompt = f"""Answer using only the provided code context.

Context:
{context}

Question:
{request.query}

Answer:"""

            async for chunk in llm.astream([HumanMessage(content=prompt)]):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            yield f"\nError: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )
