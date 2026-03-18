"""
Core RAG Pipeline — the full production pipeline with self-healing.

Pipeline:
    Semantic cache check → Advanced retrieval → Context compression
    → Generation → Reflection → Verification → Hallucination check
    → Confidence score → Self-healing retry → Cache update

This is the heart of the system.
"""

import asyncio
import structlog
from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, Document

from app.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_RETRIES,
    MAX_TOKENS,
    CONFIDENCE_THRESHOLD,
    GROUNDING_THRESHOLD,
    LOW_CONFIDENCE_FALLBACK,
)
from app.prompts.prompt_manager import (
    get_prompt,
    CONTEXT_COMPRESSION_PROMPT,
    VERIFICATION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    CONFIDENCE_SCORE_PROMPT,
    REFLECTION_PROMPT,
    RETRIEVAL_CONFIDENCE_PROMPT,
)
from app.resilience import retry_async, with_timeout

logger = structlog.get_logger()


def _get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Get LLM instance."""
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=temperature,
    )


# ─── Context Compression ─────────────────────────────────

async def compress_context(query: str, docs: List[Document]) -> str:
    """
    Compress context by extracting only relevant information.
    Reduces tokens 40-70% and improves answer quality.
    """
    text = "\n\n---\n\n".join(d.page_content for d in docs)

    # If context is short enough, skip compression
    if len(text) < 1500:
        return text

    llm = _get_llm(temperature=0.1)

    prompt = CONTEXT_COMPRESSION_PROMPT.format(
        query=query,
        context=text[:MAX_TOKENS * 4]  # Limit input size
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        compressed = response.content.strip()

        logger.info(
            "context_compressed",
            original_len=len(text),
            compressed_len=len(compressed),
            reduction=f"{(1 - len(compressed)/len(text))*100:.0f}%"
        )

        return compressed

    except Exception as e:
        logger.error("compression_error", error=str(e))
        return text[:MAX_TOKENS * 3]  # Fallback: truncate


# ─── Generation ──────────────────────────────────────────

async def generate(query: str, context: str, prompt_version: str = None) -> str:
    """Generate answer using the RAG prompt template."""
    llm = _get_llm()

    prompt_template = get_prompt(prompt_version)
    prompt = prompt_template.format(query=query, context=context)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    except Exception as e:
        logger.error("generation_error", error=str(e))
        raise


async def generate_direct(query: str) -> str:
    """Generate answer without retrieval (for simple questions)."""
    llm = _get_llm()

    prompt = f"""You are a helpful code assistant. Answer this question concisely:

{query}

Answer:"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ─── Verification ────────────────────────────────────────

async def verify_answer(query: str, answer: str, context: str) -> str:
    """Check if answer is supported by context. Returns SUPPORTED or UNSUPPORTED."""
    llm = _get_llm(temperature=0.0)

    prompt = VERIFICATION_PROMPT.format(
        query=query,
        context=context,
        answer=answer
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content.strip().upper()
        logger.info("verification", result=result)
        return result

    except Exception as e:
        logger.error("verification_error", error=str(e))
        return "SUPPORTED"  # Fail open


# ─── Hallucination Detection ─────────────────────────────

async def hallucination_check(answer: str, context: str) -> str:
    """Check if answer contains claims not in context. Returns YES or NO."""
    llm = _get_llm(temperature=0.0)

    prompt = HALLUCINATION_CHECK_PROMPT.format(
        answer=answer,
        context=context
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content.strip().upper()
        logger.info("hallucination_check", result=result)
        return result

    except Exception as e:
        logger.error("hallucination_check_error", error=str(e))
        return "NO"  # Fail open


# ─── Confidence Scoring ──────────────────────────────────

async def confidence_score(query: str, answer: str, context: str) -> int:
    """Score answer confidence 0-100."""
    llm = _get_llm(temperature=0.0)

    prompt = CONFIDENCE_SCORE_PROMPT.format(
        query=query,
        context=context,
        answer=answer
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        score = int(response.content.strip())
        score = max(0, min(100, score))
        logger.info("confidence_scored", score=score)
        return score

    except Exception as e:
        logger.error("confidence_score_error", error=str(e))
        return 50  # Default middle confidence


# ─── Grounding Score ─────────────────────────────────────

async def grounding_score(answer: str, context: str) -> int:
    """Score how grounded the answer is in context (0-100)."""
    llm = _get_llm(temperature=0.0)

    prompt = f"""Score how well grounded this answer is in the provided context.
A grounded answer only makes claims supported by the context.

Answer:
{answer}

Context:
{context}

Return a single number from 0 to 100. Nothing else."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        score = int(response.content.strip())
        return max(0, min(100, score))

    except Exception as e:
        logger.error("grounding_score_error", error=str(e))
        return 50


# ─── Reflection ──────────────────────────────────────────

async def reflect(query: str, answer: str, context: str) -> str:
    """Review and improve answer if incomplete."""
    llm = _get_llm()

    prompt = REFLECTION_PROMPT.format(
        query=query,
        context=context,
        answer=answer
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        improved = response.content.strip()

        if len(improved) > len(answer) * 0.5:  # Sanity check
            return improved
        return answer

    except Exception as e:
        logger.error("reflection_error", error=str(e))
        return answer


# ─── Retrieval Confidence ────────────────────────────────

async def retrieval_confidence(query: str, context: str) -> int:
    """Score how well retrieved context can answer the question (0-100)."""
    llm = _get_llm(temperature=0.0)

    prompt = RETRIEVAL_CONFIDENCE_PROMPT.format(
        query=query,
        context=context
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        score = int(response.content.strip())
        return max(0, min(100, score))

    except Exception as e:
        logger.error("retrieval_confidence_error", error=str(e))
        return 50


# ─── Regeneration (self-healing) ─────────────────────────

async def regenerate(query: str, context: str, attempts: int = 2) -> str:
    """
    Regenerate answer multiple times, pick the best one.
    Part of the self-healing strategy.
    """
    best_answer = None
    best_score = 0

    for i in range(attempts):
        answer = await generate(query, context)
        score = await confidence_score(query, answer, context)

        if score > best_score:
            best_score = score
            best_answer = answer

        logger.debug(
            "regeneration_attempt",
            attempt=i + 1,
            score=score
        )

    return best_answer or await generate(query, context)


# ═══════════════════════════════════════════════════════════
# FULL PRODUCTION RAG PIPELINE
# ═══════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Full production RAG pipeline with self-healing.
    
    Flow:
        Cache check → Retrieval → Compression → Generation
        → Reflection → Verification → Hallucination check
        → Confidence → Self-healing retry → Cache update
    """

    def __init__(self, retrieval_orchestrator, cache_manager):
        self.retriever = retrieval_orchestrator
        self.cache = cache_manager

    async def query(
        self,
        query: str,
        use_cache: bool = True,
        prompt_version: str = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline.
        
        Args:
            query: User question
            use_cache: Enable caching
            prompt_version: Prompt template version
            
        Returns:
            Dict with answer, confidence, sources, and metadata
        """
        # ─── Step 1: Cache Check ──────────────────────────
        if use_cache:
            cached = await self.cache.get(query)
            if cached:
                logger.info("cache_hit", query=query[:80])
                return {
                    "answer": cached,
                    "confidence": 95,
                    "cached": True,
                    "sources": [],
                }

        # ─── Step 2: Advanced Retrieval ───────────────────
        docs = await self.retriever.retrieve(query, k=8)
        
        if not docs:
            return {
                "answer": "No relevant code found in the indexed codebase.",
                "confidence": 0,
                "cached": False,
                "sources": [],
            }

        # ─── Step 3: Context Compression ──────────────────
        context = await compress_context(query, docs)

        # ─── Step 4: Generation ───────────────────────────
        answer = await generate(query, context, prompt_version)

        # ─── Step 5: Reflection ───────────────────────────
        answer = await reflect(query, answer, context)

        # ─── Step 6: Verification ─────────────────────────
        verification = await verify_answer(query, answer, context)
        
        if "UNSUPPORTED" in verification:
            answer = "The retrieved code context does not contain enough information to answer this question accurately."

        # ─── Step 7: Hallucination Check ──────────────────
        hallucination = await hallucination_check(answer, context)
        
        if "YES" in hallucination:
            logger.warning("hallucination_detected", query=query[:80])
            answer = "Answer removed due to insufficient grounding in the codebase."

        # ─── Step 8: Confidence Score ─────────────────────
        conf = await confidence_score(query, answer, context)
        ground = await grounding_score(answer, context)

        # ─── Step 9: Self-Healing Retry ───────────────────
        if conf < CONFIDENCE_THRESHOLD or ground < GROUNDING_THRESHOLD:
            logger.info(
                "self_healing_triggered",
                confidence=conf,
                grounding=ground
            )

            # Expand retrieval
            docs = await self.retriever.retrieve(query, k=15)
            context = await compress_context(query, docs)
            
            # Regenerate with multiple attempts
            answer = await regenerate(query, context, attempts=2)
            conf = await confidence_score(query, answer, context)
            ground = await grounding_score(answer, context)

        # ─── Step 10: Failure Fallback ────────────────────
        if conf < LOW_CONFIDENCE_FALLBACK:
            answer = "Insufficient information available in the codebase to provide a reliable answer."
            conf = conf  # Keep the low score

        # ─── Step 11: Cache Update ────────────────────────
        if use_cache and conf >= CONFIDENCE_THRESHOLD:
            await self.cache.set(query, answer)

        # ─── Build Response ───────────────────────────────
        sources = []
        for doc in docs[:5]:
            source = {
                "file": doc.metadata.get("file_path", "unknown"),
                "name": doc.metadata.get("name", "unknown"),
                "type": doc.metadata.get("chunk_type", "unknown"),
                "lines": f"{doc.metadata.get('start_line', '?')}-{doc.metadata.get('end_line', '?')}",
            }
            sources.append(source)

        return {
            "answer": answer,
            "confidence": conf,
            "grounding": ground,
            "cached": False,
            "sources": sources,
            "metadata": {
                "docs_retrieved": len(docs),
                "context_length": len(context),
                "prompt_version": prompt_version or "v3",
            },
        }
