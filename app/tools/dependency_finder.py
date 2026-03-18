"""
Dependency Finder Tool — finds imports, call relationships, and builds dependency graphs.
High engineering signal feature — most RAG systems can't do this.
"""

import ast
import os
import structlog
from typing import List, Dict, Set, Any

logger = structlog.get_logger()


class DependencyAnalyzer:
    """Analyzes code dependencies using AST parsing."""

    def __init__(self):
        self.import_graph: Dict[str, Set[str]] = {}  # file -> imports
        self.call_graph: Dict[str, Set[str]] = {}    # function -> called functions
        self.reverse_deps: Dict[str, Set[str]] = {}  # function -> callers
        self._indexed = False

    def build_from_chunks(self, chunks: list) -> None:
        """Build dependency graphs from indexed code chunks."""
        for chunk in chunks:
            name = f"{chunk.file_path}:{chunk.name}"

            # Import graph
            self.import_graph[name] = set(chunk.imports)

            # Call graph
            self.call_graph[name] = set(chunk.calls)

            # Build reverse dependencies
            for called in chunk.calls:
                if called not in self.reverse_deps:
                    self.reverse_deps[called] = set()
                self.reverse_deps[called].add(name)

        self._indexed = True

        logger.info(
            "dependency_graph_built",
            nodes=len(self.call_graph),
            edges=sum(len(v) for v in self.call_graph.values())
        )

    def find_dependencies(self, component_name: str) -> Dict[str, Any]:
        """
        Find all dependencies for a component.
        
        Returns:
            Dict with imports, calls, callers, and dependency chain
        """
        if not self._indexed:
            return {"error": "Dependency graph not built. Index codebase first."}

        # Find matching components
        matches = [
            k for k in self.call_graph.keys()
            if component_name.lower() in k.lower()
        ]

        if not matches:
            return {
                "component": component_name,
                "error": f"No component found matching '{component_name}'"
            }

        results = []
        for match in matches[:3]:  # Limit to top 3 matches
            imports = list(self.import_graph.get(match, set()))
            calls = list(self.call_graph.get(match, set()))

            # Find callers (reverse deps)
            name_part = match.split(":")[-1] if ":" in match else match
            callers = list(self.reverse_deps.get(name_part, set()))

            results.append({
                "component": match,
                "imports": imports,
                "calls": calls,
                "called_by": callers,
                "total_dependencies": len(imports) + len(calls),
            })

        return {
            "query": component_name,
            "matches": results,
        }

    def get_call_chain(self, function_name: str, depth: int = 3) -> Dict:
        """
        Trace the call chain for a function (who calls what).
        """
        visited = set()
        chain = self._trace_calls(function_name, depth, visited)
        return {
            "function": function_name,
            "call_chain": chain,
            "depth": depth,
        }

    def _trace_calls(self, name: str, depth: int, visited: set) -> Dict:
        """Recursively trace call chain."""
        if depth <= 0 or name in visited:
            return {"name": name, "calls": []}

        visited.add(name)
        
        # Find matching node
        matching_nodes = [
            k for k in self.call_graph.keys()
            if name.lower() in k.lower()
        ]

        calls = set()
        for node in matching_nodes:
            calls.update(self.call_graph.get(node, set()))

        children = []
        for call in list(calls)[:10]:  # Limit breadth
            child = self._trace_calls(call, depth - 1, visited)
            children.append(child)

        return {
            "name": name,
            "calls": children,
        }

    def format_report(self, component_name: str) -> str:
        """Format a human-readable dependency report."""
        deps = self.find_dependencies(component_name)

        if "error" in deps:
            return deps["error"]

        lines = [f"Dependency Analysis for: {component_name}\n"]

        for match in deps.get("matches", []):
            lines.append(f"Component: {match['component']}")
            lines.append(f"  Total dependencies: {match['total_dependencies']}")

            if match['imports']:
                lines.append(f"  Imports:")
                for imp in match['imports'][:10]:
                    lines.append(f"    - {imp}")

            if match['calls']:
                lines.append(f"  Calls:")
                for call in match['calls'][:10]:
                    lines.append(f"    - {call}")

            if match['called_by']:
                lines.append(f"  Called by:")
                for caller in match['called_by'][:10]:
                    lines.append(f"    - {caller}")

            lines.append("")

        return "\n".join(lines)


# ─── Singleton analyzer ──────────────────────────────────
_analyzer = DependencyAnalyzer()


def get_analyzer() -> DependencyAnalyzer:
    """Get the global dependency analyzer instance."""
    return _analyzer


async def dependency_search(query: str) -> str:
    """Async wrapper for dependency analysis."""
    analyzer = get_analyzer()
    return analyzer.format_report(query)


def create_dependency_tool():
    """Factory to create the dependency finder tool."""
    from app.tools.tool_base import Tool

    return Tool(
        name="dependency_finder",
        description="Find imports, dependencies, and call relationships for a code component. "
                    "Use this when asked about what depends on something or what a component uses.",
        func=dependency_search,
    )
