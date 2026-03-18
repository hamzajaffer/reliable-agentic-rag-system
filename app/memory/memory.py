"""
Memory Layer — conversation memory for agent context.
Tracks past queries and answers for multi-turn interactions.
"""

import structlog
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = structlog.get_logger()


@dataclass
class MemoryEntry:
    """A single memory entry."""
    query: str
    answer: str
    timestamp: str
    confidence: int = 0
    tool_used: str = ""


class Memory:
    """
    Conversation memory for agent.
    Tracks query-answer history for multi-turn reasoning.
    """

    def __init__(self, max_entries: int = 50):
        self.history: List[MemoryEntry] = []
        self.max_entries = max_entries

    def add(
        self,
        query: str,
        answer: str,
        confidence: int = 0,
        tool_used: str = ""
    ) -> None:
        """Add a query-answer pair to memory."""
        entry = MemoryEntry(
            query=query,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            tool_used=tool_used,
        )
        self.history.append(entry)

        # Evict oldest entries if full
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]

        logger.debug("memory_added", query=query[:50])

    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Get the N most recent entries."""
        return self.history[-n:]

    def get_context(self, n: int = 3) -> str:
        """Get formatted recent context for the agent."""
        recent = self.get_recent(n)
        
        if not recent:
            return "No previous conversation history."

        lines = ["Previous conversation:"]
        for entry in recent:
            lines.append(f"Q: {entry.query}")
            lines.append(f"A: {entry.answer[:200]}")
            lines.append("")

        return "\n".join(lines)

    def search(self, query: str) -> Optional[MemoryEntry]:
        """Simple keyword search in memory."""
        query_lower = query.lower()
        
        for entry in reversed(self.history):
            if any(word in entry.query.lower() for word in query_lower.split()):
                return entry

        return None

    def clear(self) -> None:
        """Clear all memory."""
        self.history = []
        logger.info("memory_cleared")

    @property
    def size(self) -> int:
        """Number of entries in memory."""
        return len(self.history)
