"""
Resilience layer — retry logic, timeouts, and fallback model support.
Production RAG must survive: LLM failure, Vector DB failure, timeouts, rate limits.
"""

import asyncio
import structlog
from typing import Callable, Any, Optional
from functools import wraps

from app.config import MAX_RETRIES

logger = structlog.get_logger()


async def retry_async(
    func: Callable,
    *args,
    retries: int = None,
    backoff: float = 1.0,
    max_backoff: float = 10.0,
    **kwargs
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        retries: Number of retry attempts
        backoff: Initial backoff in seconds
        max_backoff: Maximum backoff duration
    """
    retries = retries or MAX_RETRIES
    last_error = None

    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            wait = min(backoff * (2 ** attempt), max_backoff)
            
            logger.warning(
                "retry_attempt",
                attempt=attempt + 1,
                max_retries=retries,
                wait=wait,
                error=str(e)
            )
            
            await asyncio.sleep(wait)

    logger.error(
        "all_retries_failed",
        function=func.__name__,
        error=str(last_error)
    )
    raise last_error


async def with_timeout(
    coro,
    timeout: float = 15.0,
    fallback: Any = None
) -> Any:
    """
    Execute coroutine with timeout protection.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        fallback: Value to return on timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("operation_timeout", timeout=timeout)
        if fallback is not None:
            return fallback
        raise


async def with_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Try primary function, fall back to secondary on failure.
    Used for fallback model support.
    
    Example:
        result = await with_fallback(gpt4_call, gpt35_call, query)
    """
    try:
        return await primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(
            "primary_failed_using_fallback",
            primary=primary_func.__name__,
            fallback=fallback_func.__name__,
            error=str(e)
        )
        return await fallback_func(*args, **kwargs)


class CircuitBreaker:
    """
    Simple circuit breaker pattern.
    After N consecutive failures, short-circuit to avoid cascading failures.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.is_open = False

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.is_open:
            import time
            elapsed = time.time() - self.last_failure_time
            if elapsed < self.reset_timeout:
                logger.warning("circuit_breaker_open")
                raise Exception("Circuit breaker is open")
            else:
                self.is_open = False
                self.failures = 0

        try:
            result = await func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception as e:
            import time
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.is_open = True
                logger.error(
                    "circuit_breaker_tripped",
                    failures=self.failures
                )
            
            raise
