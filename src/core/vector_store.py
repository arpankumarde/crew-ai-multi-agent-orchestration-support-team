# ChromaDB Vector Store Management
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import hashlib
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document

from .config import get_settings


class VectorStore:
    """
    ChromaDB vector store manager for knowledge base storage and retrieval.
    Handles document embedding, storage, and similarity search operations.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # ChromaDB client and collection
        self._client = None
        self._collection = None

        # OpenAI embeddings
        self._embeddings = None

        # Text splitter for document chunking
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.knowledge_base.chunk_size,
            chunk_overlap=self.settings.knowledge_base.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Document cache for tracking changes
        self._document_cache = {}

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client
            self._create_client()

            # Initialize embeddings
            self._initialize_embeddings()

            # Get or create collection
            await self._get_or_create_collection()

            self.logger.info("Vector store initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise

    def _create_client(self) -> None:
        """Create ChromaDB client with persistent storage"""
        try:
            # Ensure storage directory exists
            storage_path = Path(self.settings.vector_store.path)
            storage_path.mkdir(parents=True, exist_ok=True)

            # Create ChromaDB client with persistent storage
            self._client = chromadb.PersistentClient(
                path=str(storage_path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )

            self.logger.info(f"ChromaDB client created with storage at {storage_path}")

        except Exception as e:
            self.logger.error(f"Failed to create ChromaDB client: {e}")
            raise

    def _initialize_embeddings(self) -> None:
        """Initialize OpenAI embeddings"""
        try:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.ai_models.embedding_model,
                openai_api_key=self.settings.ai_models.openai_api_key,
            )
            self.logger.info("OpenAI embeddings initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            raise

    async def _get_or_create_collection(self) -> None:
        """Get existing collection or create new one"""
        try:
            collection_name = self.settings.vector_store.collection_name

            # Try to get existing collection
            try:
                self._collection = self._client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                        api_key=self.settings.ai_models.openai_api_key,
                        model_name=self.settings.ai_models.embedding_model,
                    ),
                )
                self.logger.info(f"Retrieved existing collection: {collection_name}")

            except Exception:
                # Create new collection if it doesn't exist
                self._collection = self._client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                        api_key=self.settings.ai_models.openai_api_key,
                        model_name=self.settings.ai_models.embedding_model,
                    ),
                    metadata={
                        "hnsw:space": self.settings.vector_store.distance_function
                    },
                )
                self.logger.info(f"Created new collection: {collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to get/create collection: {e}")
            raise

    async def add_documents_from_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Load and add documents from a directory to the vector store.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            Dictionary with operation results
        """
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                raise FileNotFoundError(f"Directory {directory_path} does not exist")

            # Load documents from directory
            documents = []
            supported_extensions = [
                f".{ext}" for ext in self.settings.knowledge_base.supported_formats
            ]

            for ext in supported_extensions:
                pattern = f"**/*{ext}"
                loader = DirectoryLoader(
                    str(directory_path),
                    glob=pattern,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                )

                try:
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load files with extension {ext}: {e}"
                    )

            if not documents:
                self.logger.warning(f"No documents found in {directory_path}")
                return {
                    "status": "warning",
                    "message": "No documents found",
                    "count": 0,
                }

            # Process and add documents
            result = await self._process_and_add_documents(documents)

            self.logger.info(
                f"Added {result['added_count']} documents from {directory_path}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to add documents from directory: {e}")
            raise

    async def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a single document to the vector store.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            Document ID
        """
        try:
            # Create document object
            document = Document(page_content=content, metadata=metadata or {})

            # Process and add document
            result = await self._process_and_add_documents([document])

            if result["added_count"] > 0:
                return result["document_ids"][0]
            else:
                raise RuntimeError("Failed to add document")

        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            raise

    async def _process_and_add_documents(
        self, documents: List[Document]
    ) -> Dict[str, Any]:
        """Process documents and add them to the vector store"""
        try:
            # Split documents into chunks
            chunks = []
            for doc in documents:
                doc_chunks = self._text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)

            if not chunks:
                return {
                    "status": "warning",
                    "message": "No chunks created",
                    "added_count": 0,
                    "document_ids": [],
                }

            # Prepare data for ChromaDB
            ids = []
            documents_content = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                # Generate unique ID for chunk
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
                chunk_id = f"{content_hash}_{i}"

                # Skip if document already exists (based on hash)
                if chunk_id in self._document_cache:
                    continue

                # Prepare metadata
                metadata = chunk.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": i,
                        "content_length": len(chunk.page_content),
                        "source": metadata.get("source", "unknown"),
                        "category": self._determine_category(
                            chunk.page_content, metadata
                        ),
                    }
                )

                ids.append(chunk_id)
                documents_content.append(chunk.page_content)
                metadatas.append(metadata)

                # Cache document
                self._document_cache[chunk_id] = {
                    "content": chunk.page_content,
                    "metadata": metadata,
                }

            if not ids:
                return {
                    "status": "info",
                    "message": "All documents already exist",
                    "added_count": 0,
                    "document_ids": [],
                }

            # Add to ChromaDB collection
            self._collection.add(
                ids=ids, documents=documents_content, metadatas=metadatas
            )

            self.logger.info(f"Added {len(ids)} document chunks to vector store")

            return {
                "status": "success",
                "message": f"Successfully added {len(ids)} document chunks",
                "added_count": len(ids),
                "document_ids": ids,
                "total_chunks": len(chunks),
                "skipped_count": len(chunks) - len(ids),
            }

        except Exception as e:
            self.logger.error(f"Failed to process and add documents: {e}")
            raise

    def _determine_category(self, content: str, metadata: Dict[str, Any]) -> str:
        """Determine document category based on content and metadata"""
        source = metadata.get("source", "").lower()
        content_lower = content.lower()

        # Category mapping based on filename or content
        if "faq" in source or "frequently asked questions" in content_lower:
            return "faqs"
        elif "product" in source or "documentation" in content_lower:
            return "product_docs"
        elif (
            "troubleshoot" in source
            or "error" in content_lower
            or "problem" in content_lower
        ):
            return "troubleshooting"
        elif (
            "policy" in source or "terms" in content_lower or "privacy" in content_lower
        ):
            return "policies"
        else:
            return "general"

    async def similarity_search(
        self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform similarity search in the vector store.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Filter conditions

        Returns:
            Search results with documents and metadata
        """
        try:
            # Query the collection
            results = self._collection.query(
                query_texts=[query],
                n_results=min(
                    n_results, self.settings.knowledge_base.max_chunks_per_query
                ),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Process results
            processed_results = {
                "query": query,
                "total_results": (
                    len(results["documents"][0]) if results["documents"] else 0
                ),
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],
                "ids": results.get("ids", [[]])[0],
            }

            return processed_results

        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            raise

    async def similarity_search_with_embeddings(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform similarity search using pre-computed embeddings.

        Args:
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            where: Filter conditions

        Returns:
            Search results with documents and metadata
        """
        try:
            # Query the collection with embeddings
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(
                    n_results, self.settings.knowledge_base.max_chunks_per_query
                ),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Process results
            processed_results = {
                "total_results": (
                    len(results["documents"][0]) if results["documents"] else 0
                ),
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],
                "ids": results.get("ids", [[]])[0],
            }

            return processed_results

        except Exception as e:
            self.logger.error(f"Embedding similarity search failed: {e}")
            raise

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            result = self._collection.get(
                ids=[document_id], include=["documents", "metadatas"]
            )

            if result["documents"] and len(result["documents"]) > 0:
                return {
                    "id": document_id,
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            return None

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store"""
        try:
            self._collection.delete(ids=[document_id])

            # Remove from cache
            if document_id in self._document_cache:
                del self._document_cache[document_id]

            self.logger.info(f"Deleted document {document_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def update_document(
        self, document_id: str, content: str, metadata: Dict[str, Any] = None
    ) -> bool:
        """Update an existing document"""
        try:
            # Delete existing document
            await self.delete_document(document_id)

            # Add updated document
            new_id = await self.add_document(content, metadata)

            self.logger.info(f"Updated document {document_id} -> {new_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update document {document_id}: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            # Get collection count
            count = self._collection.count()

            # Get sample of documents for category analysis
            sample = self._collection.get(limit=min(count, 100), include=["metadatas"])

            # Analyze categories
            categories = {}
            if sample["metadatas"]:
                for metadata in sample["metadatas"]:
                    category = metadata.get("category", "unknown")
                    categories[category] = categories.get(category, 0) + 1

            return {
                "total_documents": count,
                "collection_name": self.settings.vector_store.collection_name,
                "categories": categories,
                "storage_path": self.settings.vector_store.path,
                "embedding_model": self.settings.ai_models.embedding_model,
            }

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            # Test basic operations
            test_query = "health check test"
            results = await self.similarity_search(test_query, n_results=1)

            stats = await self.get_collection_stats()

            return {
                "status": "healthy",
                "client_status": "connected" if self._client else "disconnected",
                "collection_status": "available" if self._collection else "unavailable",
                "document_count": stats.get("total_documents", 0),
                "search_test": "passed" if results else "failed",
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "client_status": "connected" if self._client else "disconnected",
                "collection_status": "available" if self._collection else "unavailable",
            }

    async def reset_collection(self) -> None:
        """Reset the collection (delete all documents)"""
        try:
            self._client.delete_collection(self.settings.vector_store.collection_name)
            await self._get_or_create_collection()
            self._document_cache.clear()

            self.logger.info("Collection reset successfully")

        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            raise


# Global vector store instance
_vector_store: Optional[VectorStore] = None


async def get_vector_store() -> VectorStore:
    """Get global vector store instance"""
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()
        await _vector_store.initialize()

    return _vector_store


async def close_vector_store() -> None:
    """Close vector store (cleanup if needed)"""
    global _vector_store
    # ChromaDB doesn't require explicit closing for persistent client
    _vector_store = None


# Features:

# ChromaDB persistent client with configurable storage

# OpenAI embeddings integration for semantic search

# Document chunking with overlap for better retrieval

# Automatic categorization based on content analysis

# Similarity search with filtering capabilities

# Document management (add, update, delete, get)

# Collection statistics and health monitoring

# Key Capabilities:

# Directory-based document loading

# Metadata-driven categorization (faqs, product_docs, troubleshooting, policies)

# Embedding-based and text-based search

# Change detection with document hashing
