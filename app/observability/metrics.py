"""
Prometheus Metrics — production monitoring.
Tracks request counts, latency, errors, and token usage.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest


# ─── Request Metrics ──────────────────────────────────────

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG query requests",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total errors",
    ["endpoint", "error_type"]
)

# ─── RAG Pipeline Metrics ─────────────────────────────────

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Cache hits",
    ["cache_type"]
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Cache misses",
    ["cache_type"]
)

# ─── Quality Metrics ─────────────────────────────────────

CONFIDENCE_SCORE = Histogram(
    "rag_confidence_score",
    "Answer confidence scores",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

SELF_HEALING_TRIGGERS = Counter(
    "rag_self_healing_total",
    "Self-healing retry triggers"
)

# ─── Resource Metrics ────────────────────────────────────

DOCUMENTS_INDEXED = Gauge(
    "rag_documents_indexed",
    "Total documents indexed"
)

ACTIVE_QUERIES = Gauge(
    "rag_active_queries",
    "Currently active queries"
)


# ─── Cost Tracking ───────────────────────────────────────

class CostTracker:
    """Track token usage and estimated cost."""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0
        self.cost_per_token = 0.000002  # GPT-4o-mini approximate
    
    def add(self, token_count: int) -> None:
        self.total_tokens += token_count
        self.total_requests += 1
    
    def estimate_cost(self) -> float:
        return self.total_tokens * self.cost_per_token
    
    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "estimated_cost_usd": round(self.estimate_cost(), 4),
        }


# Global cost tracker
cost_tracker = CostTracker()


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest()
