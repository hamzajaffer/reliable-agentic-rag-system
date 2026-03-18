"""
Multi-query expansion + Query decomposition.

Multi-query: Generate 4 diverse search queries from one user question.
Decomposition: Split complex questions into atomic sub-questions.

This is how Perplexity / Microsoft Copilot style systems improve recall.
"""

import structlog
from typing import List

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from app.config import OPENAI_API_KEY, OPENAI_MODEL

logger = structlog.get_logger()


def _get_llm():
    """Get LLM instance for query processing."""
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.3,
    )


async def expand_query(query: str, num_queries: int = 4) -> List[str]:
    """
    Generate multiple diverse search queries from a single query.
    
    Example:
        Input: "How does RAG reduce hallucinations?"
        Output: [
            "How does retrieval augmented generation work?",
            "How does RAG improve factual accuracy?",
            "Why does RAG reduce hallucinations?",
            "RAG architecture explanation"
        ]
    """
    llm = _get_llm()

    prompt = f"""Generate {num_queries} different search queries to find code and documentation related to:

{query}

The queries should be diverse — cover different aspects, use different terminology, 
and approach the topic from different angles.

Return only the queries, one per line. No numbering, no explanations."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        queries = [
            q.strip()
            for q in response.content.strip().split("\n")
            if len(q.strip()) > 5
        ]

        # Always include original query
        if query not in queries:
            queries.insert(0, query)

        logger.info(
            "query_expanded",
            original=query[:80],
            expanded_count=len(queries)
        )

        return queries[:num_queries + 1]

    except Exception as e:
        logger.error("query_expansion_error", error=str(e))
        return [query]


async def decompose_query(query: str) -> List[str]:
    """
    Break a complex question into atomic sub-questions.
    
    Example:
        Input: "Compare RAG vs fine tuning and when to use each"
        Output: [
            "What is RAG?",
            "What is fine tuning?",
            "RAG advantages?",
            "Fine tuning advantages?",
            "When should RAG be used?"
        ]
    """
    llm = _get_llm()

    prompt = f"""Break this complex question into simple, atomic sub-questions:

{query}

Each sub-question should be answerable independently.
Return only the questions, one per line. No numbering."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        sub_queries = [
            q.strip()
            for q in response.content.strip().split("\n")
            if len(q.strip()) > 5
        ]

        logger.info(
            "query_decomposed",
            original=query[:80],
            sub_queries=len(sub_queries)
        )

        return sub_queries if sub_queries else [query]

    except Exception as e:
        logger.error("query_decomposition_error", error=str(e))
        return [query]


def is_complex_query(query: str) -> bool:
    """
    Detect if a query is complex and needs decomposition.
    
    Heuristic-based detection. Production improvement: use a classifier.
    """
    complexity_signals = [
        " and ",
        " vs ",
        " versus ",
        "compare",
        "difference between",
        "how does .* relate to",
        "explain .* and .*",
        " both ",
        " multiple ",
    ]

    query_lower = query.lower()

    for signal in complexity_signals:
        if signal in query_lower:
            return True

    # Long queries are often complex
    if len(query.split()) > 15:
        return True

    return False
