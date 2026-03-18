"""
Cache layer — Redis async cache + Semantic cache.

Redis: Fast key-value cache for exact query matches.
Semantic: Embedding-based similarity cache for semantically similar queries.

Production systems always use semantic cache to reduce LLM cost.
Example: "What is RAG?" and "What is retrieval augmented generation?" → cache hit.
"""

import json
import structlog
import numpy as np
from typing import Optional, Dict, Any

from langchain_openai import OpenAIEmbeddings

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    REDIS_HOST,
    REDIS_PORT,
    SEMANTIC_CACHE_THRESHOLD,
)

logger = structlog.get_logger()


# ─── Redis Cache ──────────────────────────────────────────

class RedisCache:
    """Async Redis cache for exact query matching."""

    def __init__(self):
        self.client = None
        self._connect()

    def _connect(self):
        """Initialize Redis connection."""
        try:
            import redis.asyncio as aioredis
            self.client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
            )
            logger.info("redis_connected")
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e))
            self.client = None

    async def get(self, key: str) -> Optional[str]:
        """Get cached value by key."""
        if not self.client:
            return None
        try:
            value = await self.client.get(f"rag:{key}")
            if value:
                logger.debug("cache_hit", key=key[:50])
            return value
        except Exception as e:
            logger.error("redis_get_error", error=str(e))
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """Set cached value with TTL (default 1 hour)."""
        if not self.client:
            return
        try:
            await self.client.set(
                f"rag:{key}",
                value,
                ex=ttl,
            )
            logger.debug("cache_set", key=key[:50])
        except Exception as e:
            logger.error("redis_set_error", error=str(e))

    async def clear(self) -> None:
        """Clear all RAG cache entries."""
        if not self.client:
            return
        try:
            keys = []
            async for key in self.client.scan_iter("rag:*"):
                keys.append(key)
            if keys:
                await self.client.delete(*keys)
            logger.info("cache_cleared", keys=len(keys))
        except Exception as e:
            logger.error("redis_clear_error", error=str(e))


# ─── Semantic Cache ───────────────────────────────────────

class SemanticCache:
    """
    Embedding-based similarity cache.
    Finds cached answers for semantically similar queries.
    
    Threshold: 0.92 cosine similarity → cache hit.
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold or SEMANTIC_CACHE_THRESHOLD
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
        )
        self.cache = []  # List of (embedding, query, response) tuples
        self.max_cache_size = 1000

    def add(self, query: str, response: str) -> None:
        """Add a query-response pair to semantic cache."""
        try:
            emb = self.embeddings.embed_query(query)
            self.cache.append((emb, query, response))

            # Evict oldest entries if cache is full
            if len(self.cache) > self.max_cache_size:
                self.cache = self.cache[-self.max_cache_size:]

            logger.debug("semantic_cache_add", query=query[:50])
        except Exception as e:
            logger.error("semantic_cache_add_error", error=str(e))

    def search(self, query: str) -> Optional[str]:
        """
        Search for a semantically similar cached query.
        Returns cached response if similarity > threshold.
        """
        if not self.cache:
            return None

        try:
            q_emb = self.embeddings.embed_query(query)

            best_response = None
            best_score = 0.0

            for cached_emb, cached_query, cached_response in self.cache:
                # Cosine similarity
                sim = float(np.dot(q_emb, cached_emb) / (
                    np.linalg.norm(q_emb) * np.linalg.norm(cached_emb) + 1e-8
                ))

                if sim > best_score:
                    best_score = sim
                    best_response = cached_response

            if best_score > self.threshold:
                logger.info(
                    "semantic_cache_hit",
                    query=query[:50],
                    similarity=round(best_score, 4)
                )
                return best_response

            return None

        except Exception as e:
            logger.error("semantic_cache_search_error", error=str(e))
            return None

    def clear(self) -> None:
        """Clear all entries from semantic cache."""
        self.cache = []
        logger.info("semantic_cache_cleared")


# ─── Combined Cache ───────────────────────────────────────

class CacheManager:
    """
    Combined cache manager that checks:
    1. Semantic cache first (embedding similarity)
    2. Redis cache second (exact match)
    """

    def __init__(self):
        self.redis = RedisCache()
        self.semantic = SemanticCache()

    async def get(self, query: str) -> Optional[str]:
        """Check both caches."""
        # Semantic cache first (higher value)
        result = self.semantic.search(query)
        if result:
            return result

        # Redis exact match
        result = await self.redis.get(query)
        return result

    async def set(self, query: str, response: str) -> None:
        """Store in both caches."""
        self.semantic.add(query, response)
        await self.redis.set(query, response)

    async def clear(self) -> None:
        """Clear both caches."""
        self.semantic.clear()
        await self.redis.clear()
