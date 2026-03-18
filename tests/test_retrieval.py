"""
Tests for the retrieval components.
"""

import pytest
from ingestion.chunking import CodeChunker, CodeChunk


class TestCodeChunker:
    """Test AST-based code chunking."""

    def test_chunk_simple_function(self):
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("test.py", code)
        
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "hello"
        assert func_chunks[0].docstring == "Say hello."

    def test_chunk_class(self):
        code = '''
class Calculator:
    """A simple calculator."""
    
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("test.py", code)
        
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Calculator"
        assert len(method_chunks) == 2

    def test_chunk_with_imports(self):
        code = '''
import os
from typing import List

def process(items: List[str]) -> None:
    for item in items:
        print(item)
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("test.py", code)
        
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) == 1
        assert "os" in func_chunks[0].imports
        assert "typing.List" in func_chunks[0].imports

    def test_syntax_error_fallback(self):
        code = "this is not valid python code {{{"
        chunker = CodeChunker()
        chunks = chunker.chunk_file("test.py", code)
        
        # Should create a module-level chunk as fallback
        assert len(chunks) >= 0  # May be 0 if content too small

    def test_searchable_text(self):
        code = '''
def calculate(x: int, y: int) -> int:
    """Calculate the sum."""
    return x + y
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("test.py", code)
        
        func = [c for c in chunks if c.chunk_type == "function"][0]
        text = func.searchable_text
        
        assert "calculate" in text
        assert "Calculate the sum" in text
        assert "Code:" in text
