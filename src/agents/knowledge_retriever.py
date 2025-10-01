# Knowledge Retriever Agent
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
import os
from src.core.config import settings
from src.core.vector_store import VectorStore


class KnowledgeRetrieverAgent:
    """
    Agent responsible for searching the knowledge base using vector similarity
    to find relevant information for customer queries.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.ai_models.openai_model,
            temperature=0.2,
            api_key=settings.ai_models.openai_api_key,
        )

        self.embeddings = OpenAIEmbeddings(
            model=settings.ai_models.embedding_model,
            api_key=settings.ai_models.openai_api_key,
        )

        self.vector_store = VectorStore()

        # Knowledge base categories
        self.knowledge_categories = {
            "faqs": "Frequently Asked Questions",
            "product_docs": "Product Documentation",
            "troubleshooting": "Technical Troubleshooting Guides",
            "policies": "Company Policies and Procedures",
        }

    @tool
    def vector_search_tool(
        self, query: str, intent: str = "general", max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search the vector database for relevant knowledge base content.

        Args:
            query: Search query text
            intent: Intent category to focus search
            max_results: Maximum number of results to return

        Returns:
            Dictionary containing search results with relevance scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)

            # Prepare filters based on intent
            filters = self._prepare_filters(intent)

            # Search vector database
            results = self.vector_store.similarity_search(
                query_embedding=query_embedding, n_results=max_results, where=filters
            )

            # Process and rank results
            processed_results = self._process_search_results(results, query)

            return {
                "results": processed_results,
                "total_found": len(processed_results),
                "search_query": query,
                "filters_applied": filters,
            }

        except Exception as e:
            print(f"Error in vector search: {e}")
            return {
                "results": [],
                "total_found": 0,
                "error": str(e),
                "search_query": query,
            }

    @tool
    def document_retrieval_tool(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve specific documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            Dictionary containing retrieved documents
        """
        try:
            documents = []

            for doc_id in document_ids:
                doc = self.vector_store.get_document(doc_id)
                if doc:
                    documents.append(
                        {
                            "id": doc_id,
                            "content": doc.get("content", ""),
                            "metadata": doc.get("metadata", {}),
                            "source": doc.get("source", "unknown"),
                        }
                    )

            return {
                "documents": documents,
                "retrieved_count": len(documents),
                "requested_count": len(document_ids),
            }

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return {"documents": [], "retrieved_count": 0, "error": str(e)}

    def _prepare_filters(self, intent: str) -> Dict[str, Any]:
        """Prepare search filters based on intent category"""
        filters = {}

        # Intent-based category mapping
        intent_to_category = {
            "account": ["faqs", "policies"],
            "billing": ["faqs", "policies"],
            "technical": ["troubleshooting", "product_docs"],
            "product": ["product_docs", "faqs"],
            "order": ["faqs", "policies"],
            "support": ["faqs", "troubleshooting"],
            "complaint": ["policies", "faqs"],
            "cancel": ["policies", "faqs"],
        }

        if intent in intent_to_category:
            filters["category"] = {"$in": intent_to_category[intent]}

        return filters

    def _process_search_results(
        self, results: Dict[str, Any], query: str
    ) -> List[Dict[str, Any]]:
        """Process and enhance search results"""
        if not results or "documents" not in results:
            return []

        processed_results = []

        documents = (
            results.get("documents", [[]])[0] if results.get("documents") else []
        )
        metadatas = (
            results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        )
        distances = (
            results.get("distances", [[]])[0] if results.get("distances") else []
        )
        ids = results.get("ids", [[]])[0] if results.get("ids") else []

        for i, (doc, metadata, distance, doc_id) in enumerate(
            zip(documents, metadatas, distances, ids)
        ):
            if doc and distance is not None:
                # Convert distance to similarity score
                similarity_score = 1 - distance if distance <= 1 else 0

                # Extract relevant excerpt
                excerpt = self._extract_relevant_excerpt(doc, query)

                processed_result = {
                    "id": doc_id,
                    "content": doc,
                    "excerpt": excerpt,
                    "similarity_score": round(similarity_score, 3),
                    "relevance_rank": i + 1,
                    "metadata": metadata or {},
                    "source": (
                        metadata.get("source", "unknown") if metadata else "unknown"
                    ),
                    "category": (
                        metadata.get("category", "general") if metadata else "general"
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

    def _extract_relevant_excerpt(
        self, document: str, query: str, max_length: int = 300
    ) -> str:
        """Extract most relevant excerpt from document based on query"""
        try:
            # Simple approach: find sentences containing query terms
            query_terms = query.lower().split()
            sentences = document.split(".")

            best_sentence = ""
            max_matches = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue

                sentence_lower = sentence.lower()
                matches = sum(1 for term in query_terms if term in sentence_lower)

                if matches > max_matches:
                    max_matches = matches
                    best_sentence = sentence

            if best_sentence:
                # Extend to include surrounding context
                excerpt = best_sentence
                if len(excerpt) > max_length:
                    excerpt = excerpt[:max_length] + "..."
                return excerpt
            else:
                # Fallback to first part of document
                return (
                    document[:max_length] + "..."
                    if len(document) > max_length
                    else document
                )

        except Exception as e:
            print(f"Error extracting excerpt: {e}")
            return (
                document[:max_length] + "..."
                if len(document) > max_length
                else document
            )

    def create_agent(self) -> Agent:
        """Create and configure the CrewAI agent"""
        return Agent(
            role="Knowledge Base Specialist",
            goal="Find the most relevant and accurate information from the company knowledge base",
            backstory="""You are a research specialist with deep knowledge of the company's 
            products, services, and policies. You can quickly locate the most relevant information 
            to help customers by searching through documentation, FAQs, troubleshooting guides, 
            and company policies. You excel at finding precise answers and providing proper 
            source attribution.""",
            llm=self.llm,
            tools=[self.vector_search_tool, self.document_retrieval_tool],
            max_iter=5,
            verbose=True,
            allow_delegation=False,
        )


# Role: Information retrieval from vector database

# Capabilities:

# Vector similarity search using ChromaDB

# Intent-based filtering for focused results

# Relevant excerpt extraction from documents

# Source attribution and relevance ranking

# Key Features:

# Semantic search across knowledge base categories

# Query embedding generation with OpenAI

# Context-aware excerpt extraction

# Similarity scoring and result ranking
