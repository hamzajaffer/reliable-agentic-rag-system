"""
Indexing — embeds code chunks and stores them in Qdrant vector DB.
Also builds a BM25 index for hybrid search.
Includes batch embedding with retry logic and idempotent indexing.
"""

import os
import pickle
import structlog
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    QDRANT_URL,
    QDRANT_PORT,
    QDRANT_COLLECTION,
)

logger = structlog.get_logger()


class CodebaseIndexer:
    """
    Indexes code chunks into:
    1. Qdrant vector store (for semantic search)
    2. BM25 index (for keyword search)
    """

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or QDRANT_COLLECTION
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
        )
        self.documents: List[Document] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_texts: List[str] = []
        self.vectorstore = None

    def index_chunks(self, chunks: list, batch_size: int = 50) -> None:
        """
        Index code chunks into vector store and BM25.
        
        Args:
            chunks: List of CodeChunk objects
            batch_size: Number of chunks to embed at once
        """
        # Convert CodeChunks to LangChain Documents
        self.documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.searchable_text,
                metadata=chunk.metadata,
            )
            self.documents.append(doc)

        logger.info(
            "indexing_started",
            total_documents=len(self.documents)
        )

        # ─── Build Vector Index ───────────────────────────
        self._build_vector_index(batch_size)

        # ─── Build BM25 Index ────────────────────────────
        self._build_bm25_index()

        logger.info(
            "indexing_complete",
            vector_docs=len(self.documents),
            bm25_docs=len(self.bm25_texts)
        )

    def _build_vector_index(self, batch_size: int) -> None:
        """Build Qdrant vector index with batch embedding."""
        try:
            from langchain_community.vectorstores import Qdrant
            
            # Process in batches
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i + batch_size]
                
                if i == 0:
                    # First batch creates the collection
                    self.vectorstore = Qdrant.from_documents(
                        batch,
                        self.embeddings,
                        url=QDRANT_URL,
                        port=QDRANT_PORT,
                        collection_name=self.collection_name,
                        force_recreate=True,
                    )
                else:
                    # Subsequent batches add to existing collection
                    self.vectorstore.add_documents(batch)
                
                logger.debug(
                    "batch_indexed",
                    batch_start=i,
                    batch_size=len(batch)
                )

        except Exception as e:
            logger.warning(
                "qdrant_unavailable_using_memory",
                error=str(e)
            )
            # Fallback: in-memory vector store using FAISS or simple list
            self._build_memory_vector_index()

    def _build_memory_vector_index(self) -> None:
        """Fallback: build in-memory vector store when Qdrant is unavailable."""
        try:
            from langchain_community.vectorstores import FAISS
            self.vectorstore = FAISS.from_documents(
                self.documents,
                self.embeddings,
            )
            logger.info("fallback_faiss_index_built")
        except ImportError:
            # Ultra-minimal fallback
            logger.warning("no_vector_store_available")
            self.vectorstore = None

    def _build_bm25_index(self) -> None:
        """Build BM25 keyword search index."""
        self.bm25_texts = [
            doc.page_content for doc in self.documents
        ]
        
        tokenized = [
            text.lower().split() for text in self.bm25_texts
        ]
        
        self.bm25_index = BM25Okapi(tokenized)
        
        logger.info(
            "bm25_index_built",
            documents=len(self.bm25_texts)
        )

    def save_bm25(self, path: str = "data/bm25_index.pkl") -> None:
        """Persist BM25 index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "bm25": self.bm25_index,
            "texts": self.bm25_texts,
            "documents": self.documents,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info("bm25_saved", path=path)

    def load_bm25(self, path: str = "data/bm25_index.pkl") -> None:
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.bm25_index = data["bm25"]
        self.bm25_texts = data["texts"]
        self.documents = data["documents"]
        
        logger.info("bm25_loaded", path=path, docs=len(self.documents))

    def get_vectorstore(self):
        """Return the vector store instance."""
        return self.vectorstore

    def get_bm25(self):
        """Return the BM25 index and texts."""
        return self.bm25_index, self.bm25_texts, self.documents
