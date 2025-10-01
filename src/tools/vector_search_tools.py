# Vector Search Tools for CrewAI Agents
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings

from src.core.config import get_settings
from src.core.vector_store import get_vector_store
from src.core.utils import sanitize_input, measure_execution_time


class VectorSearchInput(BaseModel):
    """Input schema for vector search tool"""

    query: str = Field(..., description="Search query text")
    intent: str = Field("general", description="Intent category to focus search")
    max_results: int = Field(5, description="Maximum number of results to return")
    similarity_threshold: float = Field(0.7, description="Minimum similarity score")


class DocumentRetrievalInput(BaseModel):
    """Input schema for document retrieval tool"""

    document_ids: List[str] = Field(..., description="List of document IDs to retrieve")


class KnowledgeBaseInput(BaseModel):
    """Input schema for knowledge base tool"""

    query: str = Field(..., description="Knowledge base search query")
    categories: Optional[List[str]] = Field(None, description="Categories to search in")
    max_results: int = Field(5, description="Maximum number of results")


class VectorSearchTool(BaseTool):
    """
    Advanced vector search tool for semantic similarity search in the knowledge base.
    Uses ChromaDB with OpenAI embeddings for high-quality results.
    """

    name: str = "vector_search_tool"
    description: str = """Perform semantic search in the knowledge base using vector embeddings.
    Finds relevant documents based on meaning rather than exact keyword matches."""
    args_schema: type[BaseModel] = VectorSearchInput

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Intent-based category mapping for focused search
        self.intent_to_categories = {
            "account": ["faqs", "policies"],
            "billing": ["faqs", "policies"],
            "technical": ["troubleshooting", "product_docs"],
            "product": ["product_docs", "faqs"],
            "order": ["faqs", "policies"],
            "support": ["faqs", "troubleshooting"],
            "complaint": ["policies", "faqs"],
            "cancel": ["policies", "faqs"],
        }

    @measure_execution_time
    async def _arun(
        self,
        query: str,
        intent: str = "general",
        max_results: int = 5,
        similarity_threshold: float = 0.7,
    ) -> str:
        """Perform vector search asynchronously"""
        try:
            # Sanitize input
            query = sanitize_input(query)
            if not query:
                return "Error: Empty search query provided"

            # Get vector store instance
            vector_store = await get_vector_store()

            # Prepare filters based on intent
            filters = self._prepare_search_filters(intent)

            # Perform similarity search
            results = await vector_store.similarity_search(
                query=query, n_results=max_results, where=filters
            )

            if not results or not results.get("documents"):
                return f"No relevant documents found for query: '{query}'"

            # Process and filter results by similarity threshold
            processed_results = self._process_search_results(
                results, similarity_threshold
            )

            if not processed_results:
                return f"No documents found above similarity threshold {similarity_threshold} for query: '{query}'"

            # Format results for agent consumption
            formatted_results = self._format_search_results(processed_results, query)

            return f"Found {len(processed_results)} relevant documents:\n{formatted_results}"

        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return f"Vector search failed: {str(e)}"

    def _run(
        self,
        query: str,
        intent: str = "general",
        max_results: int = 5,
        similarity_threshold: float = 0.7,
    ) -> str:
        """Synchronous wrapper for async vector search"""
        return asyncio.run(self._arun(query, intent, max_results, similarity_threshold))

    def _prepare_search_filters(self, intent: str) -> Dict[str, Any]:
        """Prepare search filters based on intent category"""
        filters = {}

        # Map intent to relevant categories
        if intent in self.intent_to_categories:
            categories = self.intent_to_categories[intent]
            if categories:
                filters["category"] = {"$in": categories}

        return filters

    def _process_search_results(
        self, results: Dict[str, Any], similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Process raw search results and apply similarity filtering"""
        processed_results = []

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        ids = results.get("ids", [])

        for i, (doc, metadata, distance, doc_id) in enumerate(
            zip(documents, metadatas, distances, ids)
        ):
            if doc and distance is not None:
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = (
                    1 - distance if distance <= 1 else max(0, 1 - distance)
                )

                # Apply similarity threshold
                if similarity_score >= similarity_threshold:
                    # Extract relevant excerpt
                    excerpt = self._extract_key_excerpt(doc, results["query"])

                    processed_result = {
                        "id": doc_id,
                        "content": doc,
                        "excerpt": excerpt,
                        "similarity_score": round(similarity_score, 3),
                        "metadata": metadata or {},
                        "source": (
                            metadata.get("source", "unknown") if metadata else "unknown"
                        ),
                        "category": (
                            metadata.get("category", "general")
                            if metadata
                            else "general"
                        ),
                        "title": (
                            metadata.get("title", f"Document {doc_id}")
                            if metadata
                            else f"Document {doc_id}"
                        ),
                    }

                    processed_results.append(processed_result)

        # Sort by similarity score descending
        processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return processed_results

    def _extract_key_excerpt(
        self, document: str, query: str, max_length: int = 200
    ) -> str:
        """Extract the most relevant excerpt from a document"""
        try:
            # Split query into terms
            query_terms = [term.lower() for term in query.split() if len(term) > 2]

            # Split document into sentences
            sentences = [s.strip() for s in document.split(".") if len(s.strip()) > 10]

            if not sentences:
                return (
                    document[:max_length] + "..."
                    if len(document) > max_length
                    else document
                )

            # Score sentences based on query term matches
            sentence_scores = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for term in query_terms if term in sentence_lower)
                sentence_scores.append((sentence, score))

            # Find best scoring sentence
            best_sentence = max(sentence_scores, key=lambda x: x[1])[0]

            if len(best_sentence) <= max_length:
                return best_sentence
            else:
                return best_sentence[:max_length] + "..."

        except Exception:
            # Fallback to first part of document
            return (
                document[:max_length] + "..."
                if len(document) > max_length
                else document
            )

    def _format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for agent consumption"""
        formatted = []

        for i, result in enumerate(results, 1):
            formatted.append(
                f"""
{i}. **{result['title']}** (Score: {result['similarity_score']})
   Category: {result['category']}
   Source: {result['source']}
   Excerpt: {result['excerpt']}
   """
            )

        return "\n".join(formatted)


class DocumentRetrievalTool(BaseTool):
    """
    Tool for retrieving specific documents by their IDs.
    Used when agents need to access full document content.
    """

    name: str = "document_retrieval_tool"
    description: str = """Retrieve specific documents from the knowledge base using document IDs.
    Returns full document content with metadata."""
    args_schema: type[BaseModel] = DocumentRetrievalInput

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @measure_execution_time
    async def _arun(self, document_ids: List[str]) -> str:
        """Retrieve documents by IDs asynchronously"""
        try:
            if not document_ids:
                return "Error: No document IDs provided"

            # Sanitize document IDs
            clean_ids = [
                sanitize_input(doc_id) for doc_id in document_ids if doc_id.strip()
            ]
            if not clean_ids:
                return "Error: No valid document IDs provided"

            vector_store = await get_vector_store()

            # Retrieve documents
            retrieved_docs = []
            for doc_id in clean_ids:
                doc = await vector_store.get_document(doc_id)
                if doc:
                    retrieved_docs.append(doc)
                else:
                    self.logger.warning(f"Document not found: {doc_id}")

            if not retrieved_docs:
                return f"No documents found for IDs: {clean_ids}"

            # Format retrieved documents
            formatted_docs = []
            for i, doc in enumerate(retrieved_docs, 1):
                formatted_docs.append(
                    f"""
Document {i}: {doc['id']}
Title: {doc['metadata'].get('title', 'Untitled')}
Category: {doc['metadata'].get('category', 'general')}
Source: {doc['metadata'].get('source', 'unknown')}

Content:
{doc['content']}
                """
                )

            return f"Retrieved {len(retrieved_docs)} documents:\n" + "\n---\n".join(
                formatted_docs
            )

        except Exception as e:
            self.logger.error(f"Document retrieval error: {e}")
            return f"Document retrieval failed: {str(e)}"

    def _run(self, document_ids: List[str]) -> str:
        """Synchronous wrapper for async document retrieval"""
        return asyncio.run(self._arun(document_ids))


class KnowledgeBaseTool(BaseTool):
    """
    Comprehensive knowledge base search tool that combines vector search
    with category filtering and relevance ranking.
    """

    name: str = "knowledge_base_tool"
    description: str = """Search the knowledge base with advanced filtering options.
    Supports category filtering and combines multiple search strategies for best results."""
    args_schema: type[BaseModel] = KnowledgeBaseInput

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Available categories in knowledge base
        self.available_categories = {
            "faqs": "Frequently Asked Questions",
            "product_docs": "Product Documentation",
            "troubleshooting": "Technical Troubleshooting Guides",
            "policies": "Company Policies and Procedures",
        }

    @measure_execution_time
    async def _arun(
        self, query: str, categories: Optional[List[str]] = None, max_results: int = 5
    ) -> str:
        """Perform comprehensive knowledge base search"""
        try:
            # Sanitize input
            query = sanitize_input(query)
            if not query:
                return "Error: Empty search query provided"

            # Validate categories if provided
            valid_categories = []
            if categories:
                for cat in categories:
                    clean_cat = cat.lower().strip()
                    if clean_cat in self.available_categories:
                        valid_categories.append(clean_cat)
                    else:
                        self.logger.warning(f"Invalid category: {cat}")

            vector_store = await get_vector_store()

            # Prepare search filters
            filters = {}
            if valid_categories:
                filters["category"] = {"$in": valid_categories}

            # Perform vector search
            results = await vector_store.similarity_search(
                query=query, n_results=max_results, where=filters
            )

            if not results or not results.get("documents"):
                category_info = (
                    f" in categories {valid_categories}" if valid_categories else ""
                )
                return (
                    f"No relevant information found{category_info} for query: '{query}'"
                )

            # Process results with enhanced formatting
            processed_results = self._process_kb_results(results, query)

            # Format comprehensive response
            response = self._format_kb_response(
                processed_results, query, valid_categories
            )

            return response

        except Exception as e:
            self.logger.error(f"Knowledge base search error: {e}")
            return f"Knowledge base search failed: {str(e)}"

    def _run(
        self, query: str, categories: Optional[List[str]] = None, max_results: int = 5
    ) -> str:
        """Synchronous wrapper for async knowledge base search"""
        return asyncio.run(self._arun(query, categories, max_results))

    def _process_kb_results(
        self, results: Dict[str, Any], query: str
    ) -> List[Dict[str, Any]]:
        """Process knowledge base results with enhanced metadata"""
        processed_results = []

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        ids = results.get("ids", [])

        for doc, metadata, distance, doc_id in zip(
            documents, metadatas, distances, ids
        ):
            if doc and distance is not None:
                similarity_score = (
                    1 - distance if distance <= 1 else max(0, 1 - distance)
                )

                # Enhanced excerpt extraction
                excerpt = self._extract_contextual_excerpt(doc, query)

                # Determine relevance level
                relevance_level = self._determine_relevance_level(similarity_score)

                processed_result = {
                    "id": doc_id,
                    "content": doc,
                    "excerpt": excerpt,
                    "similarity_score": round(similarity_score, 3),
                    "relevance_level": relevance_level,
                    "metadata": metadata or {},
                    "source": (
                        metadata.get("source", "unknown") if metadata else "unknown"
                    ),
                    "category": (
                        metadata.get("category", "general") if metadata else "general"
                    ),
                    "category_name": self.available_categories.get(
                        metadata.get("category", "general") if metadata else "general",
                        "General",
                    ),
                    "title": (
                        metadata.get("title", f"Document {doc_id}")
                        if metadata
                        else f"Document {doc_id}"
                    ),
                }

                processed_results.append(processed_result)

        # Sort by similarity score
        processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return processed_results

    def _extract_contextual_excerpt(
        self, document: str, query: str, max_length: int = 300
    ) -> str:
        """Extract contextual excerpt with surrounding context"""
        try:
            query_terms = [term.lower() for term in query.split() if len(term) > 2]

            # Find best matching paragraph
            paragraphs = [
                p.strip() for p in document.split("\n\n") if len(p.strip()) > 20
            ]

            if not paragraphs:
                # Fallback to sentences
                sentences = [
                    s.strip() for s in document.split(".") if len(s.strip()) > 10
                ]
                paragraphs = sentences

            best_paragraph = ""
            max_matches = 0

            for paragraph in paragraphs:
                paragraph_lower = paragraph.lower()
                matches = sum(1 for term in query_terms if term in paragraph_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_paragraph = paragraph

            if not best_paragraph:
                best_paragraph = paragraphs[0] if paragraphs else document

            # Truncate if needed
            if len(best_paragraph) > max_length:
                # Try to break at sentence boundary
                truncated = best_paragraph[:max_length]
                last_sentence = truncated.rfind(".")
                if (
                    last_sentence > max_length * 0.7
                ):  # If we can keep most of the content
                    return truncated[: last_sentence + 1]
                else:
                    return truncated + "..."

            return best_paragraph

        except Exception:
            return (
                document[:max_length] + "..."
                if len(document) > max_length
                else document
            )

    def _determine_relevance_level(self, similarity_score: float) -> str:
        """Determine relevance level based on similarity score"""
        if similarity_score >= 0.9:
            return "Highly Relevant"
        elif similarity_score >= 0.8:
            return "Very Relevant"
        elif similarity_score >= 0.7:
            return "Relevant"
        elif similarity_score >= 0.6:
            return "Somewhat Relevant"
        else:
            return "Low Relevance"

    def _format_kb_response(
        self, results: List[Dict[str, Any]], query: str, categories: List[str]
    ) -> str:
        """Format comprehensive knowledge base response"""
        if not results:
            return "No relevant information found in the knowledge base."

        # Header
        category_info = f" in {', '.join(categories)}" if categories else ""
        header = f"Knowledge Base Search Results for: '{query}'{category_info}\n"
        header += f"Found {len(results)} relevant document(s):\n"
        header += "=" * 50 + "\n"

        # Format each result
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"""
{i}. **{result['title']}** ({result['relevance_level']})
   Category: {result['category_name']}
   Source: {result['source']}
   Similarity Score: {result['similarity_score']}
   
   Key Information:
   {result['excerpt']}
   
   Document ID: {result['id']}
            """
            formatted_results.append(formatted_result.strip())

        # Combine all parts
        full_response = header + "\n\n".join(formatted_results)

        # Add usage tips
        if len(results) > 1:
            full_response += f"\n\n💡 Found multiple relevant documents. The most relevant information is in the first result."

        return full_response


# Advanced Features:

# VectorSearchTool: Semantic search with intent-based filtering

# DocumentRetrievalTool: Specific document retrieval by IDs

# KnowledgeBaseTool: Comprehensive search with category filtering

# Intelligence Features:

# Intent-to-category mapping for focused search

# Contextual excerpt extraction with surrounding context

# Relevance level determination (Highly Relevant → Low Relevance)

# Similarity threshold filtering for quality results
