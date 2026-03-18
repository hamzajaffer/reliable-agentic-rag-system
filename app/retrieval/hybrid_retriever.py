"""
Hybrid Retriever — combines BM25 keyword search with vector similarity search
using Reciprocal Rank Fusion (RRF) scoring.

RRF formula: score = Σ 1/(k + rank)
This is how Microsoft, Meta, and production RAG systems do hybrid search.
"""

import structlog
from collections import defaultdict
from typing import List, Tuple, Optional

from langchain.schema import Document

from app.config import VECTOR_SEARCH_K, BM25_SEARCH_K, RRF_K

logger = structlog.get_logger()


class HybridRetriever:
    """
    Production hybrid retriever combining:
    - Vector similarity search (semantic)
    - BM25 keyword search (exact match)
    - RRF fusion scoring
    - Deduplication
    """

    def __init__(self, vectorstore, bm25_index, bm25_texts, documents):
        """
        Args:
            vectorstore: LangChain vector store (Qdrant/FAISS)
            bm25_index: BM25Okapi index
            bm25_texts: Raw texts for BM25 lookup
            documents: LangChain Document objects matching bm25_texts
        """
        self.vectorstore = vectorstore
        self.bm25 = bm25_index
        self.bm25_texts = bm25_texts
        self.documents = documents

    def search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform hybrid search with RRF fusion.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of Documents ranked by RRF score
        """
        # ─── Vector Search ────────────────────────────────
        vector_docs = self._vector_search(query, VECTOR_SEARCH_K)

        # ─── BM25 Keyword Search ─────────────────────────
        keyword_docs = self._bm25_search(query, BM25_SEARCH_K)

        # ─── RRF Fusion ──────────────────────────────────
        fused = self._rrf_fusion(vector_docs, keyword_docs)

        # ─── Deduplication ────────────────────────────────
        deduped = self._deduplicate(fused)

        logger.info(
            "hybrid_search",
            query_preview=query[:80],
            vector_results=len(vector_docs),
            bm25_results=len(keyword_docs),
            fused_results=len(deduped)
        )

        return deduped[:k]

    def _vector_search(self, query: str, k: int) -> List[Document]:
        """Perform vector similarity search."""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error("vector_search_error", error=str(e))
            return []

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """Perform BM25 keyword search."""
        if not self.bm25:
            return []
        
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]
            
            # Convert to Documents
            results = [self.documents[i] for i in top_indices if scores[i] > 0]
            return results
            
        except Exception as e:
            logger.error("bm25_search_error", error=str(e))
            return []

    def _rrf_fusion(
        self,
        vector_docs: List[Document],
        keyword_docs: List[Document],
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion — combines rankings from multiple sources.
        
        Formula: score = Σ 1/(k + rank + 1)
        where k=60 is standard constant.
        """
        k = k or RRF_K
        scores = defaultdict(float)
        doc_map = {}

        # Score vector results
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:200]  # Use content prefix as key
            scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc

        # Score keyword results
        for rank, doc in enumerate(keyword_docs):
            key = doc.page_content[:200]
            scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc

        # Sort by fused score
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [(doc_map[key], score) for key, score in ranked if key in doc_map]

    def _deduplicate(
        self, 
        ranked_docs: List[Tuple[Document, float]]
    ) -> List[Document]:
        """Remove duplicate documents based on content."""
        seen = set()
        unique = []

        for doc, score in ranked_docs:
            content_key = doc.page_content[:300]
            if content_key not in seen:
                seen.add(content_key)
                unique.append(doc)

        return unique
