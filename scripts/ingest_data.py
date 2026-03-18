"""
Ingestion Script — CLI tool to ingest a Python codebase.

Usage:
    python scripts/ingest_data.py --path /path/to/your/project
    python scripts/ingest_data.py --path ./my_project --collection my_project
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loaders import load_directory
from ingestion.chunking import chunk_codebase
from ingestion.indexing import CodebaseIndexer


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a Python codebase into the RAG system"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the Python project to ingest"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="codebase_docs",
        help="Qdrant collection name (default: codebase_docs)"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=3000,
        help="Maximum chunk size in characters (default: 3000)"
    )

    args = parser.parse_args()

    print(f"🔍 Loading files from: {args.path}")
    files = load_directory(args.path)

    if not files:
        print("❌ No Python files found.")
        sys.exit(1)

    print(f"📄 Loaded {len(files)} files")
    print(f"   Total lines: {sum(f.line_count for f in files):,}")

    print(f"\n🔪 Chunking with AST parser...")
    chunks = chunk_codebase(files, max_chunk_size=args.max_chunk_size)
    print(f"   Created {len(chunks)} chunks")

    # Show chunk type breakdown
    type_counts = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    for ct, count in sorted(type_counts.items()):
        print(f"   • {ct}: {count}")

    print(f"\n📦 Indexing into collection: {args.collection}")
    indexer = CodebaseIndexer(args.collection)
    indexer.index_chunks(chunks)

    # Save BM25 index
    indexer.save_bm25(f"data/bm25_{args.collection}.pkl")

    print(f"\n✅ Ingestion complete!")
    print(f"   Files: {len(files)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Collection: {args.collection}")
    print(f"\n🚀 Start the server with: python app/main.py")


if __name__ == "__main__":
    main()
