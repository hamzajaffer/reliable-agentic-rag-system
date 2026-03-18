"""
Central configuration for the AI Codebase Assistant.
Loads environment variables and provides typed config constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ─── Qdrant ───────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "codebase_docs")

# ─── Redis ────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ─── Reranker ─────────────────────────────────────────────
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# ─── LangSmith (optional) ────────────────────────────────
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false") == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-code-assistant")

# ─── Pipeline Limits ─────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_DOCS = int(os.getenv("MAX_DOCS", "25"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "6000"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "4"))
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))

# ─── Retrieval Defaults ──────────────────────────────────
VECTOR_SEARCH_K = 20
BM25_SEARCH_K = 20
RRF_K = 60
RERANK_TOP_N = 5
RETRIEVAL_SCORE_THRESHOLD = 0.7

# ─── Confidence Thresholds ───────────────────────────────
CONFIDENCE_THRESHOLD = 60
GROUNDING_THRESHOLD = 50
LOW_CONFIDENCE_FALLBACK = 40
