"""
AST-based code chunking — extracts functions, classes, and module-level code
with rich metadata for intelligent retrieval.

This is the KEY differentiator from basic text-splitting RAG.
Function-level chunks with dependency info enable precise code understanding.
"""

import ast
import os
import structlog
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = structlog.get_logger()


@dataclass
class CodeChunk:
    """A chunk of code with rich metadata for retrieval."""
    content: str
    chunk_type: str  # "function", "class", "method", "module"
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    return_type: Optional[str] = None

    @property
    def metadata(self) -> Dict:
        """Return metadata dict for vector store indexing."""
        return {
            "chunk_type": self.chunk_type,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring or "",
            "imports": ", ".join(self.imports),
            "calls": ", ".join(self.calls),
            "decorators": ", ".join(self.decorators),
            "parent_class": self.parent_class or "",
            "arguments": ", ".join(self.arguments),
            "return_type": self.return_type or "",
        }

    @property
    def searchable_text(self) -> str:
        """
        Build a rich text representation for embedding.
        Includes name, docstring, and code for better semantic search.
        """
        parts = []
        
        if self.parent_class:
            parts.append(f"Class: {self.parent_class}")
        
        type_label = self.chunk_type.capitalize()
        parts.append(f"{type_label}: {self.name}")
        
        if self.docstring:
            parts.append(f"Description: {self.docstring}")
        
        if self.arguments:
            parts.append(f"Arguments: {', '.join(self.arguments)}")
        
        if self.return_type:
            parts.append(f"Returns: {self.return_type}")
        
        if self.calls:
            parts.append(f"Calls: {', '.join(self.calls)}")
        
        parts.append(f"Code:\n{self.content}")
        
        return "\n".join(parts)


class CodeChunker:
    """
    AST-based code chunker that extracts functions, classes, and methods
    with rich metadata including dependencies and call relationships.
    """

    def __init__(self, max_chunk_size: int = 3000, min_chunk_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """
        Parse a Python file and extract code chunks.
        
        Args:
            file_path: Path to the source file
            content: File content string
            
        Returns:
            List of CodeChunk objects
        """
        chunks: List[CodeChunk] = []
        lines = content.split("\n")
        
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            logger.warning(
                "ast_parse_error",
                file=file_path,
                error=str(e)
            )
            # Fallback: treat entire file as one chunk
            if len(content.strip()) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=content,
                    chunk_type="module",
                    name=os.path.basename(file_path),
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                ))
            return chunks

        # Extract file-level imports
        file_imports = self._extract_imports(tree)

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                chunk = self._extract_function(node, file_path, lines, file_imports)
                if chunk and len(chunk.content) >= self.min_chunk_size:
                    chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                # Extract class-level chunk
                class_chunk = self._extract_class(node, file_path, lines, file_imports)
                if class_chunk and len(class_chunk.content) >= self.min_chunk_size:
                    chunks.append(class_chunk)

                # Extract individual methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = self._extract_function(
                            item, file_path, lines, file_imports,
                            parent_class=node.name
                        )
                        if method_chunk and len(method_chunk.content) >= self.min_chunk_size:
                            chunks.append(method_chunk)

        # If no chunks extracted, create a module-level chunk
        if not chunks and len(content.strip()) >= self.min_chunk_size:
            chunks.append(CodeChunk(
                content=content[:self.max_chunk_size],
                chunk_type="module",
                name=os.path.basename(file_path),
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                imports=file_imports,
            ))

        logger.debug(
            "file_chunked",
            file=file_path,
            chunks=len(chunks)
        )

        return chunks

    def _extract_function(
        self,
        node: ast.FunctionDef,
        file_path: str,
        lines: List[str],
        file_imports: List[str],
        parent_class: str = None
    ) -> Optional[CodeChunk]:
        """Extract a function/method as a chunk."""
        try:
            start = node.lineno - 1
            end = node.end_lineno or (start + 1)
            content = "\n".join(lines[start:end])

            # Truncate if too long
            if len(content) > self.max_chunk_size:
                content = content[:self.max_chunk_size] + "\n# ... (truncated)"

            # Extract docstring
            docstring = ast.get_docstring(node)

            # Extract function calls
            calls = self._extract_calls(node)

            # Extract decorators
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Attribute):
                    decorators.append(ast.dump(dec))

            # Extract arguments
            arguments = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    try:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    except Exception:
                        pass
                arguments.append(arg_str)

            # Extract return type
            return_type = None
            if node.returns:
                try:
                    return_type = ast.unparse(node.returns)
                except Exception:
                    pass

            chunk_type = "method" if parent_class else "function"
            if isinstance(node, ast.AsyncFunctionDef):
                chunk_type = f"async_{chunk_type}"

            return CodeChunk(
                content=content,
                chunk_type=chunk_type,
                name=node.name,
                file_path=file_path,
                start_line=node.lineno,
                end_line=end,
                docstring=docstring,
                imports=file_imports,
                calls=calls,
                decorators=decorators,
                parent_class=parent_class,
                arguments=arguments,
                return_type=return_type,
            )
        except Exception as e:
            logger.error(
                "function_extract_error",
                file=file_path,
                name=getattr(node, 'name', 'unknown'),
                error=str(e)
            )
            return None

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        lines: List[str],
        file_imports: List[str]
    ) -> Optional[CodeChunk]:
        """Extract a class definition as a chunk (without method bodies)."""
        try:
            start = node.lineno - 1
            end = node.end_lineno or (start + 1)
            
            # For the class chunk, include the class signature and docstring
            # but not the full method implementations
            class_lines = []
            class_lines.append(lines[start])  # class definition line
            
            docstring = ast.get_docstring(node)
            
            # Add method signatures only
            method_names = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_names.append(item.name)
                    sig_line = lines[item.lineno - 1].strip()
                    class_lines.append(f"    {sig_line}")
            
            content = "\n".join(class_lines)
            
            if len(content) > self.max_chunk_size:
                content = content[:self.max_chunk_size]
            
            # Extract base classes
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    pass

            return CodeChunk(
                content=content,
                chunk_type="class",
                name=node.name,
                file_path=file_path,
                start_line=node.lineno,
                end_line=end,
                docstring=docstring,
                imports=file_imports,
                calls=method_names,
                decorators=[],
            )
        except Exception as e:
            logger.error(
                "class_extract_error",
                file=file_path,
                name=getattr(node, 'name', 'unknown'),
                error=str(e)
            )
            return None

    def _extract_imports(self, tree: ast.Module) -> List[str]:
        """Extract all import statements from the AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function call names from an AST node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(set(calls))  # Deduplicate


def chunk_codebase(
    files: list,
    max_chunk_size: int = 3000,
    min_chunk_size: int = 50
) -> List[CodeChunk]:
    """
    Chunk an entire codebase (list of CodeFile objects).
    
    Args:
        files: List of CodeFile objects from loader
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        
    Returns:
        List of CodeChunk objects
    """
    chunker = CodeChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size
    )
    
    all_chunks: List[CodeChunk] = []
    
    for code_file in files:
        chunks = chunker.chunk_file(code_file.path, code_file.content)
        all_chunks.extend(chunks)
    
    logger.info(
        "codebase_chunked",
        total_files=len(files),
        total_chunks=len(all_chunks),
        chunk_types={
            ct: sum(1 for c in all_chunks if c.chunk_type == ct)
            for ct in set(c.chunk_type for c in all_chunks)
        }
    )
    
    return all_chunks
