"""
Function Explainer Tool — explains specific functions using the RAG pipeline.
Provides detailed explanations of purpose, parameters, logic, and dependencies.
"""

import structlog
from typing import Any

from app.prompts.prompt_manager import FUNCTION_EXPLAIN_PROMPT

logger = structlog.get_logger()


async def explain_function(query: str, retriever=None, llm=None) -> str:
    """
    Find and explain a specific function from the codebase.
    
    Args:
        query: Function name or description
        retriever: HybridRetriever instance
        llm: LLM instance for generation
        
    Returns:
        Detailed function explanation
    """
    if not retriever:
        return "Error: No codebase has been indexed yet."

    try:
        # Search for the function
        docs = retriever.search(query, k=3)

        if not docs:
            return f"No function found matching: {query}"

        # Find the best matching function chunk
        best_doc = None
        for doc in docs:
            if doc.metadata.get("chunk_type") in ("function", "method", "async_function", "async_method"):
                best_doc = doc
                break

        if not best_doc:
            best_doc = docs[0]

        meta = best_doc.metadata

        # Generate explanation
        if llm:
            from langchain.schema import HumanMessage

            prompt = FUNCTION_EXPLAIN_PROMPT.format(
                function_name=meta.get("name", query),
                file_path=meta.get("file_path", "unknown"),
                code=best_doc.page_content[:2000],
            )

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            explanation = response.content.strip()
        else:
            # Simple fallback explanation
            explanation = f"""
Function: {meta.get('name', 'unknown')}
File: {meta.get('file_path', 'unknown')}
Type: {meta.get('chunk_type', 'unknown')}
Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}
Docstring: {meta.get('docstring', 'None')}
Arguments: {meta.get('arguments', 'None')}
Returns: {meta.get('return_type', 'None')}
Calls: {meta.get('calls', 'None')}

Code:
{best_doc.page_content[:1000]}
"""

        logger.info(
            "function_explained",
            query=query[:80],
            function=meta.get("name", "unknown")
        )

        return explanation

    except Exception as e:
        logger.error("function_explain_error", error=str(e))
        return f"Error explaining function: {str(e)}"


def create_function_explainer_tool(retriever, llm=None):
    """Factory to create the function explainer tool."""
    from app.tools.tool_base import Tool

    async def _explain(query: str) -> str:
        return await explain_function(query, retriever, llm)

    return Tool(
        name="function_explainer",
        description="Explain a specific function or method in detail — its purpose, parameters, "
                    "return value, logic, and dependencies. Use when asked to explain code.",
        func=_explain,
    )
