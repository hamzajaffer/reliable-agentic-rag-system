"""
Tests for the agent components.
"""

import pytest
from app.tools.tool_base import Tool, ToolRegistry
from app.memory.memory import Memory


class TestToolRegistry:
    """Test tool registration and lookup."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        
        async def dummy(x):
            return x
        
        tool = Tool(name="test", description="A test tool", func=dummy)
        registry.register(tool)
        
        assert registry.get("test") is not None
        assert registry.get("test").name == "test"
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        
        async def dummy(x):
            return x
        
        registry.register(Tool("a", "Tool A", dummy))
        registry.register(Tool("b", "Tool B", dummy))
        
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_describe(self):
        registry = ToolRegistry()
        
        async def dummy(x):
            return x
        
        registry.register(Tool("search", "Search code", dummy))
        desc = registry.describe()
        
        assert "search" in desc
        assert "Search code" in desc


class TestMemory:
    """Test conversation memory."""

    def test_add_and_retrieve(self):
        mem = Memory()
        mem.add("What is X?", "X is Y", confidence=80)
        
        assert mem.size == 1
        recent = mem.get_recent(1)
        assert recent[0].query == "What is X?"
        assert recent[0].answer == "X is Y"

    def test_max_entries(self):
        mem = Memory(max_entries=3)
        
        for i in range(5):
            mem.add(f"Q{i}", f"A{i}")
        
        assert mem.size == 3
        # Should keep the most recent
        recent = mem.get_recent(3)
        assert recent[0].query == "Q2"

    def test_context_format(self):
        mem = Memory()
        mem.add("Hello?", "Hi there")
        
        context = mem.get_context()
        assert "Previous conversation:" in context
        assert "Hello?" in context

    def test_clear(self):
        mem = Memory()
        mem.add("Q", "A")
        mem.clear()
        assert mem.size == 0

    def test_search(self):
        mem = Memory()
        mem.add("How does auth work?", "Auth uses JWT tokens")
        mem.add("What is the DB?", "PostgreSQL")
        
        result = mem.search("auth")
        assert result is not None
        assert "auth" in result.query.lower()
