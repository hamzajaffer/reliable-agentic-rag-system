"""
Query Rewriter — rewrites user queries for better retrieval.

Example:
    User: "How does it work?"
    Rewritten: "How does retrieval augmented generation work?"

Improves retrieval massively by adding context and specificity.
"""

import structlog
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from app.config import OPENAI_API_KEY, OPENAI_MODEL

logger = structlog.get_logger()


async def rewrite_query(query: str) -> str:
    """
    Rewrite a query to be more specific and retrieval-friendly.
    
    Args:
        query: Original user query
        
    Returns:
        Improved query for retrieval
    """
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.2,
    )

    prompt = f"""Rewrite this query to be more specific and suitable for searching a Python codebase.
Add technical terms and context that would help find relevant code.

Original query:
{query}

Return only the rewritten query, nothing else."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        rewritten = response.content.strip()
        
        logger.info(
            "query_rewritten",
            original=query[:80],
            rewritten=rewritten[:80]
        )
        
        return rewritten

    except Exception as e:
        logger.error("query_rewrite_error", error=str(e))
        return query
