"""
Cross-Encoder Reranker — improves precision by rescoring retrieved documents.

Flow: Retriever → Top 20 → Reranker → Top 5 → LLM
Vector search is recall-optimized; reranking gives precision.
Typically improves answer quality 20-40%.
"""

import structlog
from typing import List

from langchain.schema import Document
from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL, RERANK_TOP_N

logger = structlog.get_logger()

# Lazy-load model to avoid startup cost
_reranker_model = None


def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder model."""
    global _reranker_model
    if _reranker_model is None:
        logger.info("loading_reranker", model=RERANKER_MODEL)
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        logger.info("reranker_loaded")
    return _reranker_model


def rerank(
    query: str,
    docs: List[Document],
    top_n: int = None
) -> List[Document]:
    """
    Rerank documents using cross-encoder model.
    
    Args:
        query: The search query
        docs: List of candidate documents
        top_n: Number of top documents to return
        
    Returns:
        Reranked list of documents (top_n best)
    """
    top_n = top_n or RERANK_TOP_N
    
    if not docs:
        return []

    if len(docs) <= top_n:
        return docs

    try:
        model = _get_model()

        # Create query-document pairs
        pairs = [
            [query, doc.page_content]
            for doc in docs
        ]

        # Score all pairs
        scores = model.predict(pairs)

        # Sort by score descending
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        result = [doc for doc, score in ranked[:top_n]]

        logger.info(
            "reranking_complete",
            input_docs=len(docs),
            output_docs=len(result),
            top_score=float(max(scores)),
            bottom_score=float(min(scores))
        )

        return result

    except Exception as e:
        logger.error("reranking_error", error=str(e))
        # Fallback: return first top_n docs without reranking
        return docs[:top_n]


def rerank_with_scores(
    query: str,
    docs: List[Document],
    top_n: int = None
) -> List[tuple]:
    """
    Rerank and return documents with their scores.
    Useful for confidence assessment.
    """
    top_n = top_n or RERANK_TOP_N
    
    if not docs:
        return []

    try:
        model = _get_model()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_n]

    except Exception as e:
        logger.error("reranking_error", error=str(e))
        return [(doc, 0.0) for doc in docs[:top_n]]
