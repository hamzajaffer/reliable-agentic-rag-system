"""
Latency Benchmark — measures pipeline performance.
"""

import time
import asyncio
import statistics


async def benchmark_retrieval(retriever, queries: list) -> dict:
    """Benchmark retrieval latency."""
    latencies = []
    
    for query in queries:
        start = time.time()
        try:
            results = retriever.search(query, k=5)
        except Exception:
            pass
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
    
    return {
        "operation": "retrieval",
        "queries": len(queries),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


async def benchmark_rag(rag_pipeline, queries: list) -> dict:
    """Benchmark full RAG pipeline latency."""
    latencies = []
    
    for query in queries:
        start = time.time()
        try:
            await rag_pipeline.query(query)
        except Exception:
            pass
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
    
    return {
        "operation": "rag_pipeline",
        "queries": len(queries),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


if __name__ == "__main__":
    test_queries = [
        "What is the main function?",
        "How does authentication work?",
        "Explain the database layer",
        "What does process_payment do?",
        "Find all API endpoints",
    ]
    
    print("Latency benchmark — run after ingesting a codebase")
    print(f"Test queries: {len(test_queries)}")
