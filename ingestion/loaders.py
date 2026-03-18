"""
Code file loader — walks a directory tree and loads Python source files.
Supports filtering by extension and directory exclusion.
"""

import os
import structlog
from typing import List, Dict
from dataclasses import dataclass, field

logger = structlog.get_logger()


@dataclass
class CodeFile:
    """Represents a loaded source code file."""
    path: str
    filename: str
    content: str
    extension: str
    size_bytes: int
    line_count: int


EXCLUDED_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", "egg-info", ".eggs"
}

SUPPORTED_EXTENSIONS = {".py"}


def load_directory(
    root_path: str,
    extensions: set = None,
    exclude_dirs: set = None,
    max_file_size: int = 500_000  # 500KB limit
) -> List[CodeFile]:
    """
    Recursively load all supported source files from a directory.
    
    Args:
        root_path: Root directory to scan
        extensions: File extensions to include (default: .py)
        exclude_dirs: Directory names to skip
        max_file_size: Maximum file size in bytes
        
    Returns:
        List of CodeFile objects
    """
    extensions = extensions or SUPPORTED_EXTENSIONS
    exclude_dirs = exclude_dirs or EXCLUDED_DIRS
    
    files: List[CodeFile] = []
    
    if not os.path.isdir(root_path):
        logger.error("directory_not_found", path=root_path)
        return files
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out excluded directories (in-place to prevent os.walk descent)
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs and not d.startswith(".")
        ]
        
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in extensions:
                continue
            
            filepath = os.path.join(dirpath, filename)
            
            try:
                file_size = os.path.getsize(filepath)
                
                if file_size > max_file_size:
                    logger.warning(
                        "file_too_large",
                        path=filepath,
                        size=file_size
                    )
                    continue
                
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                code_file = CodeFile(
                    path=filepath,
                    filename=filename,
                    content=content,
                    extension=ext,
                    size_bytes=file_size,
                    line_count=content.count("\n") + 1,
                )
                
                files.append(code_file)
                
                logger.debug(
                    "file_loaded",
                    path=filepath,
                    lines=code_file.line_count
                )
                
            except Exception as e:
                logger.error(
                    "file_read_error",
                    path=filepath,
                    error=str(e)
                )
    
    logger.info(
        "directory_loaded",
        root=root_path,
        total_files=len(files),
        total_lines=sum(f.line_count for f in files)
    )
    
    return files


def load_single_file(filepath: str) -> CodeFile:
    """Load a single source file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    return CodeFile(
        path=filepath,
        filename=os.path.basename(filepath),
        content=content,
        extension=os.path.splitext(filepath)[1].lower(),
        size_bytes=os.path.getsize(filepath),
        line_count=content.count("\n") + 1,
    )
