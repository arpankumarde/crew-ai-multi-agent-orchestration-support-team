# Response Generator Agent
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from src.core.config import settings


class ResponseGeneratorAgent:
    """
    Agent responsible for synthesizing information from multiple sources
    to generate comprehensive, empathetic customer service responses.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.ai_models.openai_model,
            temperature=0.3,
            api_key=settings.ai_models.openai_api_key,
        )

        # Response templates for different scenarios
        self.response_templates = {
            "greeting": "Hello! Thank you for contacting our support team.",
            "acknowledgment": "I understand your concern about {issue}.",
            "solution": "Here's how we can resolve this:",
            "steps": "Please follow these steps:",
            "escalation": "I'll escalate this to our specialized team.",
            "closing": "Is there anything else I can help you with today?",
        }

        # Tone guidelines
        self.tone_guidelines = {
            "professional": "Maintain a professional and courteous tone",
            "empathetic": "Show understanding and empathy for customer concerns",
            "helpful": "Provide clear, actionable solutions",
            "concise": "Be clear and concise while being thorough",
        }

    @tool
    def response_generation_tool(
        self,
        query: str,
        intent_data: Dict[str, Any],
        knowledge_data: Dict[str, Any],
        database_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive customer service response using all available information.

        Args:
            query: Original customer query
            intent_data: Intent classification results
            knowledge_data: Knowledge base search results
            database_data: Database query results

        Returns:
            Dictionary containing generated response and metadata
        """
        try:
            # Extract relevant information
            intent = intent_data.get("intent", "general")
            urgency = intent_data.get("urgency", "medium")
            sentiment = intent_data.get("sentiment", "neutral")
            confidence = intent_data.get("confidence", 0.5)

            # Build context for response generation
            context = self._build_response_context(
                query, intent_data, knowledge_data, database_data
            )

            # Generate response using LLM
            response_prompt = self._create_response_prompt(
                query, context, intent, urgency, sentiment
            )

            llm_response = self.llm.invoke(response_prompt)
            generated_response = llm_response.content

            # Post-process response
            final_response = self._post_process_response(
                generated_response, intent, urgency
            )

            # Determine if escalation is needed
            requires_escalation = self._assess_escalation_need(
                intent_data, confidence, sentiment
            )

            return {
                "response": final_response,
                "confidence_score": confidence,
                "tone": self._determine_tone(sentiment, urgency),
                "requires_escalation": requires_escalation,
                "response_type": intent,
                "sources_used": self._identify_sources_used(
                    knowledge_data, database_data
                ),
                "word_count": len(final_response.split()),
                "estimated_reading_time": len(final_response.split()) // 200
                + 1,  # minutes
            }

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please contact our support team directly for assistance.",
                "error": str(e),
                "requires_escalation": True,
            }

    def _build_response_context(
        self, query: str, intent_data: Dict, knowledge_data: Dict, database_data: Dict
    ) -> Dict[str, Any]:
        """Build comprehensive context for response generation"""
        context = {
            "customer_query": query,
            "intent": intent_data.get("intent", "general"),
            "urgency": intent_data.get("urgency", "medium"),
            "entities": intent_data.get("entities", {}),
            "knowledge_articles": [],
            "customer_info": {},
            "order_info": {},
            "product_info": {},
        }

        # Process knowledge base results
        if knowledge_data and "results" in knowledge_data:
            for result in knowledge_data["results"][:3]:  # Top 3 results
                context["knowledge_articles"].append(
                    {
                        "title": result.get("title", "Unknown"),
                        "excerpt": result.get("excerpt", ""),
                        "source": result.get("source", "knowledge_base"),
                        "relevance": result.get("similarity_score", 0),
                    }
                )

        # Process database results
        if database_data:
            context["customer_info"] = database_data.get("customer", {})
            context["order_info"] = database_data.get(
                "order", database_data.get("orders", {})
            )
            context["product_info"] = database_data.get(
                "product", database_data.get("products", {})
            )

        return context

    def _create_response_prompt(
        self, query: str, context: Dict, intent: str, urgency: str, sentiment: str
    ) -> str:
        """Create a detailed prompt for response generation"""

        prompt = f"""
        You are a professional customer service representative. Generate a helpful, empathetic, and accurate response to the customer query using the provided context.

        CUSTOMER QUERY: "{query}"
        
        CONTEXT INFORMATION:
        # Intent: {intent}
        # Urgency: {urgency}
        Customer Sentiment: {sentiment}
        
        Customer Information: {context.get('customer_info', 'Not available')}
        Order Information: {context.get('order_info', 'Not available')}
        Product Information: {context.get('product_info', 'Not available')}
        
        Relevant Knowledge Base Articles:
        """

        # Add knowledge base articles
        for i, article in enumerate(context.get("knowledge_articles", [])[:3], 1):
            prompt += f"""
        {i}. {article.get('title', 'Unknown Title')}
           Excerpt: {article.get('excerpt', 'No excerpt available')}
           Relevance: {article.get('relevance', 'Unknown')}
        """

        prompt += f"""
        
        RESPONSE GUIDELINES:
        1. Start with a friendly greeting and acknowledgment
        2. Directly address the customer's specific question or concern
        3. Provide clear, actionable steps when applicable
        4. Include relevant personal information (orders, account details) when available
        5. Be empathetic, especially if the customer seems frustrated
        6. End with an offer for additional help
        7. Keep the response professional but warm
        8. If urgency is high, prioritize immediate solutions
        9. If information is incomplete, ask for specific details needed
        
        TONE REQUIREMENTS:
        - Professional and courteous
        - Empathetic to customer concerns
        - Clear and easy to understand
        - Solution-oriented
        - Appropriate for {urgency} urgency level
        
        Generate a complete customer service response that follows these guidelines:
        """

        return prompt

    def _post_process_response(self, response: str, intent: str, urgency: str) -> str:
        """Post-process the generated response for quality and consistency"""

        # Remove any unwanted formatting
        response = response.strip()

        # Ensure appropriate urgency handling
        if urgency == "urgent" and "urgent" not in response.lower():
            if not any(
                word in response.lower()
                for word in ["immediately", "right away", "asap", "priority"]
            ):
                response = response.replace(".", " immediately.", 1)

        # Ensure proper closing
        closing_phrases = ["let me know", "anything else", "help you", "assist you"]
        if not any(phrase in response.lower() for phrase in closing_phrases):
            response += "\n\nIs there anything else I can help you with today?"

        # Ensure reasonable length
        if len(response) > 1000:
            # Truncate while preserving sentences
            sentences = response.split(".")
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + ".") <= 950:
                    truncated += sentence + "."
                else:
                    break
            response = truncated + " Please let me know if you need any clarification."

        return response

    def _determine_tone(self, sentiment: str, urgency: str) -> str:
        """Determine appropriate tone based on sentiment and urgency"""
        if sentiment == "negative":
            return "empathetic_professional"
        elif urgency == "urgent":
            return "urgent_helpful"
        elif sentiment == "positive":
            return "friendly_professional"
        else:
            return "professional_courteous"

    def _assess_escalation_need(
        self, intent_data: Dict, confidence: float, sentiment: str
    ) -> bool:
        """Assess if the query needs escalation based on various factors"""

        # Low confidence in intent classification
        if confidence < 0.6:
            return True

        # Negative sentiment with certain intents
        if sentiment == "negative" and intent_data.get("intent") in [
            "complaint",
            "billing",
            "cancel",
        ]:
            return True

        # Explicit escalation indicators
        escalation_keywords = intent_data.get("entities", {}).get(
            "escalation_keywords", []
        )
        if escalation_keywords:
            return True

        # Complex queries flagged by intent classifier
        if intent_data.get("complexity") == "complex":
            return True

        # High urgency items
        if intent_data.get("urgency") == "urgent":
            return True

        return False

    def _identify_sources_used(
        self, knowledge_data: Dict, database_data: Dict
    ) -> List[str]:
        """Identify which sources were used in generating the response"""
        sources = []

        if knowledge_data and knowledge_data.get("results"):
            sources.append("knowledge_base")

        if database_data:
            if database_data.get("customer"):
                sources.append("customer_database")
            if database_data.get("order") or database_data.get("orders"):
                sources.append("order_database")
            if database_data.get("product") or database_data.get("products"):
                sources.append("product_database")

        return sources

    def create_agent(self) -> Agent:
        """Create and configure the CrewAI agent"""
        return Agent(
            role="Customer Support Representative",
            goal="Generate helpful, accurate, and empathetic responses to customer inquiries",
            backstory="""You are an experienced customer support representative known for providing 
            clear, helpful, and friendly responses. You always maintain a professional tone while 
            being empathetic to customer concerns. You excel at synthesizing information from multiple 
            sources to provide comprehensive solutions. You understand the importance of personalized 
            service and always try to address the specific needs of each customer.""",
            llm=self.llm,
            tools=[self.response_generation_tool],
            max_iter=4,
            verbose=True,
            allow_delegation=False,
        )


# Role: Comprehensive response synthesis

# Capabilities:

# Multi-source information synthesis

# Context-aware response generation

# Tone adjustment based on sentiment/urgency

# Escalation need assessment

# Key Features:

# Advanced prompt engineering for response quality

# Template-based response structure

# Post-processing for tone and length optimization

# Source attribution and metadata tracking
