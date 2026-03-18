"""
Agent — the agentic decision layer.

This is where the system decides HOW to answer instead of always doing retrieval → generation.
Includes: Planner, Tool Selection, Execution Engine, Reflection Loop.

Pipeline:
    Query → Planner → Tool Decision → Execution → RAG → Verification
    → Memory Update → Response

This is Agentic RAG — what modern AI systems (Copilot, Perplexity, Cursor) use.
"""

import json
import structlog
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from app.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_AGENT_STEPS,
)
from app.tools.tool_base import ToolRegistry
from app.memory.memory import Memory

logger = structlog.get_logger()


def _get_llm() -> ChatOpenAI:
    """Get LLM for agent decisions."""
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.1,
    )


# ─── Planner ─────────────────────────────────────────────

async def plan(query: str, tool_registry: ToolRegistry, memory: Memory) -> Dict:
    """
    Plan the best strategy to answer a query.
    Returns structured decision with tool selection and reasoning.
    """
    llm = _get_llm()
    tool_desc = tool_registry.describe()
    context = memory.get_context(n=3)

    prompt = f"""You are a code intelligence agent planner. Decide the best tool to answer this query.

Available tools:
{tool_desc}

{context}

Query: {query}

Return a JSON object with:
- "tool": the tool name to use
- "reason": brief explanation of why this tool is best

Return ONLY valid JSON, nothing else."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Parse JSON response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        decision = json.loads(content)
        
        logger.info(
            "plan_decision",
            tool=decision.get("tool"),
            reason=decision.get("reason", "")[:100]
        )
        
        return decision

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("plan_parse_error", error=str(e))
        # Default to RAG pipeline
        return {
            "tool": "rag",
            "reason": "Defaulting to RAG pipeline"
        }


# ─── Tool Selection ──────────────────────────────────────

async def select_tool(query: str, tool_registry: ToolRegistry) -> str:
    """Select the best tool for a query."""
    llm = _get_llm()
    tool_desc = tool_registry.describe()

    prompt = f"""Select the most appropriate tool for this query.

Tools:
{tool_desc}

Query: {query}

Return only the tool name, nothing else."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        tool_name = response.content.strip().lower()
        
        # Validate tool exists
        if tool_registry.get(tool_name):
            return tool_name
        
        # Fuzzy match
        for tool in tool_registry.list_tools():
            if tool.name in tool_name or tool_name in tool.name:
                return tool.name
        
        return "rag"  # Default

    except Exception as e:
        logger.error("tool_selection_error", error=str(e))
        return "rag"


# ─── Execution Engine ────────────────────────────────────

async def execute_tool(
    tool_name: str,
    query: str,
    tool_registry: ToolRegistry
) -> str:
    """Execute a selected tool."""
    tool = tool_registry.get(tool_name)
    
    if not tool:
        logger.warning("tool_not_found", name=tool_name)
        return f"Tool '{tool_name}' not found."

    try:
        result = await tool.run(query)
        
        logger.info(
            "tool_executed",
            tool=tool_name,
            result_length=len(str(result))
        )
        
        return str(result)

    except Exception as e:
        logger.error("tool_execution_error", tool=tool_name, error=str(e))
        return f"Tool execution error: {str(e)}"


# ─── Agent Reflection ────────────────────────────────────

async def agent_reflect(query: str, answer: str) -> str:
    """
    Agent evaluates its own answer.
    Returns: GOOD, RETRY, or EXPAND
    """
    llm = _get_llm()

    prompt = f"""Evaluate this answer to the query. Is it sufficient and accurate?

Query: {query}

Answer: {answer}

Return exactly one of:
- GOOD (answer is sufficient)
- RETRY (answer needs improvement, try again)
- EXPAND (need more information, search more)

Return only one word."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        decision = response.content.strip().upper()
        
        if decision not in ("GOOD", "RETRY", "EXPAND"):
            decision = "GOOD"
        
        logger.info("agent_reflection", decision=decision)
        return decision

    except Exception as e:
        logger.error("agent_reflect_error", error=str(e))
        return "GOOD"


# ═══════════════════════════════════════════════════════════
# AGENT LOOP — the core agentic execution engine
# ═══════════════════════════════════════════════════════════

class CodeAgent:
    """
    Agentic RAG system — the full agent with:
    - Planning
    - Tool selection & execution
    - Reflection loop
    - Memory
    - Step limits (safety)
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        rag_pipeline=None,
        memory: Memory = None,
    ):
        self.tools = tool_registry
        self.rag = rag_pipeline
        self.memory = memory or Memory()
        self.max_steps = MAX_AGENT_STEPS

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full agent loop.
        
        Think → Act → Observe → Think (up to MAX_STEPS)
        
        Returns:
            Dict with answer, tool_trace, confidence, etc.
        """
        steps = 0
        tool_trace = []
        answer = None

        logger.info("agent_start", query=query[:100])

        while steps < self.max_steps:
            steps += 1

            # ─── Step 1: Plan ─────────────────────────────
            decision = await plan(query, self.tools, self.memory)
            tool_name = decision.get("tool", "rag")
            reason = decision.get("reason", "")

            tool_trace.append({
                "step": steps,
                "tool": tool_name,
                "reason": reason,
            })

            # ─── Step 2: Execute ──────────────────────────
            if tool_name == "rag" and self.rag:
                # Use full RAG pipeline
                result = await self.rag.query(query)
                answer = result.get("answer", "")
                confidence = result.get("confidence", 0)
                
                tool_trace[-1]["result_preview"] = answer[:100]
                tool_trace[-1]["confidence"] = confidence
                
                # If RAG gives high confidence, we're done
                if confidence >= 70:
                    break
            else:
                # Execute specific tool
                result = await execute_tool(tool_name, query, self.tools)
                answer = result
                tool_trace[-1]["result_preview"] = result[:100]

            # ─── Step 3: Reflect ──────────────────────────
            reflection = await agent_reflect(query, answer)
            tool_trace[-1]["reflection"] = reflection

            if reflection == "GOOD":
                break
            elif reflection == "RETRY":
                # Try RAG pipeline on next iteration
                continue
            elif reflection == "EXPAND":
                # Will trigger different tool selection on next loop
                continue

        # ─── Memory Update ────────────────────────────────
        self.memory.add(
            query=query,
            answer=answer or "No answer generated",
            tool_used=tool_trace[-1]["tool"] if tool_trace else "none",
        )

        logger.info(
            "agent_complete",
            steps=steps,
            final_tool=tool_trace[-1]["tool"] if tool_trace else "none"
        )

        return {
            "answer": answer or "Unable to generate an answer.",
            "tool_trace": tool_trace,
            "steps": steps,
            "memory_size": self.memory.size,
        }
