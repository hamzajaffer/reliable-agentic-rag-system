"""
Tool Base — standardized interface for all agent tools.
All tools follow this interface for consistent registration and execution.
"""

from typing import Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class Tool:
    """
    Standardized tool interface for the agent.
    
    All tools must have:
    - name: unique identifier
    - description: what the tool does (used by planner for selection)
    - func: async callable that executes the tool
    """
    name: str
    description: str
    func: Callable

    async def run(self, input_data: Any) -> Any:
        """Execute the tool with given input."""
        return await self.func(input_data)

    def to_dict(self) -> dict:
        """Convert to dict for LLM function calling schema."""
        return {
            "name": self.name,
            "description": self.description,
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self.tools.values())

    def describe(self) -> str:
        """Get formatted description of all tools (for LLM planner)."""
        lines = []
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)
