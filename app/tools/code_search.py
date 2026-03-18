"""
Code Search Tool — searches for relevant code functions/classes via hybrid retrieval.
Agent tool for finding specific code components.
"""

import structlog
from typing import Any

logger = structlog.get_logger()


async def code_search(query: str, retriever=None) -> str:
    """
    Search for relevant code in the indexed codebase.
    
    Args:
        query: Search query (function name, concept, etc.)
        retriever: HybridRetriever instance
        
    Returns:
        Formatted search results with code snippets
    """
    if not retriever:
        return "Error: No codebase has been indexed yet."

    try:
        docs = retriever.search(query, k=5)

        if not docs:
            return f"No code found matching: {query}"

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            result = f"""
--- Result {i} ---
File: {meta.get('file_path', 'unknown')}
Type: {meta.get('chunk_type', 'unknown')}
Name: {meta.get('name', 'unknown')}
Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}

{doc.page_content[:500]}
"""
            results.append(result)

        logger.info("code_search", query=query[:80], results=len(docs))
        return "\n".join(results)

    except Exception as e:
        logger.error("code_search_error", error=str(e))
        return f"Search error: {str(e)}"


def create_code_search_tool(retriever):
    """Factory function to create a code search tool with bound retriever."""
    from app.tools.tool_base import Tool

    async def _search(query: str) -> str:
        return await code_search(query, retriever)

    return Tool(
        name="code_search",
        description="Search the codebase for relevant functions, classes, or code patterns. "
                    "Use this when you need to find specific code components.",
        func=_search,
    )
