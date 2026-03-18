"""
Retrieval Orchestrator — the full advanced retrieval pipeline.

Pipeline:
    Query → Rewrite → Decompose (if complex) → Multi-query expansion
    → Parallel hybrid retrieval → RRF fusion → Dedup → Rerank
    → Adaptive retrieval depth

This is the highest-quality retrieval layer.
"""

import asyncio
import structlog
from collections import defaultdict
from typing import List, Optional

from langchain.schema import Document

from app.config import MAX_DOCS, CONFIDENCE_THRESHOLD
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.multi_query import (
    expand_query,
    decompose_query,
    is_complex_query,
)
from app.retrieval.reranker import rerank

logger = structlog.get_logger()


class RetrievalOrchestrator:
    """
    Full advanced retrieval pipeline with adaptive depth.
    """

    def __init__(self, hybrid_retriever):
        self.retriever = hybrid_retriever

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        use_rewrite: bool = True,
        use_expansion: bool = True,
    ) -> List[Document]:
        """
        Full retrieval pipeline.
        
        Args:
            query: User query
            k: Final number of documents to return
            use_rewrite: Enable query rewriting
            use_expansion: Enable multi-query expansion
            
        Returns:
            Ranked, deduplicated list of Documents
        """
        # ─── Step 1: Query Rewrite ────────────────────────
        if use_rewrite:
            improved_query = await rewrite_query(query)
        else:
            improved_query = query

        # ─── Step 2: Complexity Detection + Decomposition ─
        if is_complex_query(improved_query):
            sub_queries = await decompose_query(improved_query)
            logger.info("complex_query_decomposed", sub_queries=len(sub_queries))
        else:
            sub_queries = [improved_query]

        # ─── Step 3: Multi-query Expansion ────────────────
        all_queries = []
        for sq in sub_queries:
            if use_expansion:
                expanded = await expand_query(sq, num_queries=3)
                all_queries.extend(expanded)
            else:
                all_queries.append(sq)

        # Deduplicate queries
        all_queries = list(dict.fromkeys(all_queries))

        logger.info(
            "retrieval_queries",
            total_queries=len(all_queries)
        )

        # ─── Step 4: Parallel Hybrid Retrieval ────────────
        all_docs = await self._parallel_retrieve(all_queries)

        # ─── Step 5: Cross-query RRF Fusion ───────────────
        fused = self._fuse_results(all_docs)

        # ─── Step 6: Deduplication ────────────────────────
        deduped = self._deduplicate(fused)

        # ─── Step 7: Rerank ──────────────────────────────
        top_docs = deduped[:MAX_DOCS]  # Limit before reranking
        ranked = rerank(query, top_docs, top_n=k)

        logger.info(
            "retrieval_complete",
            query=query[:80],
            total_retrieved=len(all_docs),
            after_fusion=len(fused),
            after_dedup=len(deduped),
            final_count=len(ranked)
        )

        return ranked

    async def _parallel_retrieve(
        self,
        queries: List[str]
    ) -> List[Document]:
        """Run hybrid search for all queries in parallel."""
        tasks = [
            asyncio.to_thread(
                self.retriever.search,
                q,
                15  # Per-query limit
            )
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("parallel_retrieval_error", error=str(result))
                continue
            all_docs.extend(result)

        return all_docs

    def _fuse_results(
        self,
        docs: List[Document],
        k: int = 60
    ) -> List[Document]:
        """Apply RRF fusion across multi-query results."""
        scores = defaultdict(float)
        doc_map = {}

        for rank, doc in enumerate(docs):
            key = doc.page_content[:200]
            scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc_map[key] for key, score in ranked if key in doc_map]

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents."""
        seen = set()
        unique = []

        for doc in docs:
            key = doc.page_content[:300]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique

    async def adaptive_retrieve(
        self,
        query: str,
        confidence_scorer=None,
    ) -> List[Document]:
        """
        Adaptive retrieval — retrieves more documents if confidence is low.
        
        First retrieves k=5, checks confidence.
        If low → retrieves k=15.
        If very low → uses full multi-query retrieval.
        """
        # Start with minimal retrieval
        docs = await self.retrieve(query, k=5, use_expansion=False)

        if confidence_scorer:
            context = "\n".join(d.page_content for d in docs)
            score = await confidence_scorer(query, context)

            if score < 40:
                # Very low confidence: full multi-query retrieval
                logger.info("adaptive_retrieval_full", score=score)
                docs = await self.retrieve(query, k=15, use_expansion=True)

            elif score < CONFIDENCE_THRESHOLD:
                # Low confidence: expand search
                logger.info("adaptive_retrieval_expanded", score=score)
                docs = await self.retrieve(query, k=10, use_expansion=True)

        return docs
