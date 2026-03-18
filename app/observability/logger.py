"""
Structured Logging — production logging via structlog.
Provides JSON-formatted logs with context for debugging and monitoring.
"""

import structlog
import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging for the application.
    
    Uses structlog for JSON-formatted, contextual logging.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()  # Pretty for dev, use JSONRenderer for prod
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )


def get_logger(name: str = None):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_query(query: str, source: str = "api") -> None:
    """Log an incoming query."""
    logger = structlog.get_logger()
    logger.info(
        "query_received",
        query=query[:100],
        source=source,
    )


def log_response(query: str, confidence: int, latency_ms: float) -> None:
    """Log a response with metrics."""
    logger = structlog.get_logger()
    logger.info(
        "response_sent",
        query=query[:80],
        confidence=confidence,
        latency_ms=round(latency_ms, 2),
    )
