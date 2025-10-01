# Intent Classification Agent - Corrected Version
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional
import re
import json
from src.core.config import get_settings
from src.tools.database_tools import CustomerLookupTool


class IntentClassifierAgent:
    """
    Agent responsible for analyzing customer queries and classifying their intent,
    urgency level, and complexity to route requests appropriately.
    """

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.ai_models.openai_model,
            temperature=0.1,
            openai_api_key=self.settings.ai_models.openai_api_key,
        )

        self.customer_lookup = CustomerLookupTool()

        # Intent categories mapping
        self.intent_categories = {
            "account": [
                "login",
                "password",
                "reset",
                "access",
                "account",
                "profile",
                "settings",
            ],
            "billing": [
                "payment",
                "invoice",
                "charge",
                "bill",
                "subscription",
                "refund",
                "credit",
            ],
            "technical": [
                "error",
                "bug",
                "not working",
                "crash",
                "installation",
                "setup",
                "configure",
            ],
            "product": [
                "feature",
                "how to",
                "tutorial",
                "guide",
                "documentation",
                "manual",
            ],
            "order": [
                "order",
                "purchase",
                "delivery",
                "shipping",
                "tracking",
                "status",
            ],
            "support": [
                "help",
                "assistance",
                "question",
                "inquiry",
                "issue",
                "problem",
            ],
            "complaint": [
                "frustrated",
                "angry",
                "disappointed",
                "terrible",
                "awful",
                "complaint",
            ],
            "cancel": ["cancel", "close", "terminate", "stop", "discontinue", "delete"],
        }

        # Urgency keywords
        self.urgency_keywords = {
            "urgent": [
                "urgent",
                "emergency",
                "critical",
                "immediately",
                "asap",
                "breaking",
            ],
            "high": ["important", "priority", "soon", "quickly", "fast"],
            "medium": ["when possible", "convenient", "help me", "please"],
            "low": ["whenever", "no rush", "question", "wondering"],
        }

    @tool
    def intent_classification_tool(
        self, query: str, customer_id: Optional[str] = None
    ) -> str:
        """
        Classify customer intent based on query content and context.

        Args:
            query: Customer query text
            customer_id: Optional customer identifier for context

        Returns:
            JSON string containing classification results
        """

        # Get customer context if available
        customer_context = {}
        if customer_id:
            try:
                customer_context = self.customer_lookup._run(customer_id=customer_id)
            except Exception as e:
                print(f"Warning: Could not fetch customer context: {e}")

        # Analyze intent using LLM
        classification_prompt = f"""
        Analyze the following customer query and provide a structured classification:
        
        Customer Query: "{query}"
        
        Customer Context: {customer_context if customer_context else "Unknown customer"}
        
        Classify the query according to these categories:
        
        INTENT CATEGORIES:
        - account: Login, password, profile, account management issues
        - billing: Payment, invoices, charges, refunds, subscription issues  
        - technical: Bugs, errors, installation, configuration problems
        - product: Feature questions, how-to guides, documentation requests
        - order: Order status, shipping, delivery, purchase inquiries
        - support: General help requests, questions, assistance
        - complaint: Customer complaints, frustrations, negative feedback
        - cancel: Cancellation requests, account closure, service termination
        
        URGENCY LEVELS:
        - urgent: Emergency situations, critical business impact
        - high: Important issues needing quick resolution
        - medium: Standard support requests
        - low: General questions, non-critical issues
        
        COMPLEXITY LEVELS:
        - simple: Can be resolved with standard procedures/FAQ
        - moderate: Requires some investigation or multi-step resolution
        - complex: Needs specialized knowledge or escalation
        
        Respond with ONLY a valid JSON object in this exact format:
        {{
            "intent": "primary_category",
            "confidence": 0.95,
            "urgency": "urgency_level",
            "complexity": "complexity_level",
            "entities": {{
                "order_numbers": [],
                "product_names": [],
                "email_addresses": [],
                "phone_numbers": []
            }},
            "sentiment": "positive|neutral|negative",
            "requires_human": false,
            "reasoning": "Brief explanation of classification"
        }}
        """

        try:
            response = self.llm.invoke(classification_prompt)
            classification = json.loads(response.content)

            # Extract entities using regex
            entities = self._extract_entities(query)
            classification["entities"].update(entities)

            # Validate and enhance classification
            classification = self._validate_classification(classification, query)

            return json.dumps(classification)

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            return json.dumps(self._fallback_classification(query))
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return json.dumps(self._fallback_classification(query))

    def _extract_entities(self, query: str) -> Dict[str, list]:
        """Extract entities like order numbers, emails, phone numbers from query"""
        entities = {
            "order_numbers": [],
            "product_names": [],
            "email_addresses": [],
            "phone_numbers": [],
        }

        # Order number patterns
        order_patterns = [
            r"(?:order|ord|order#|ord#)\s*[:\-]?\s*([a-zA-Z0-9\-]{6,15})",
            r"\b(ORD-\d{6})\b",
        ]
        for pattern in order_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["order_numbers"].extend(matches)

        # Email patterns
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities["email_addresses"] = re.findall(email_pattern, query)

        # Phone number patterns
        phone_patterns = [
            r"\b\d{3}-\d{3}-\d{4}\b",
            r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",
            r"\b\d{10}\b",
        ]
        for pattern in phone_patterns:
            matches = re.findall(pattern, query)
            entities["phone_numbers"].extend(matches)

        return entities

    def _validate_classification(
        self, classification: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Validate and enhance classification results"""

        # Ensure confidence is reasonable
        if classification.get("confidence", 0) < 0.5:
            classification["requires_human"] = True
            classification["complexity"] = "complex"

        # Check for escalation triggers
        escalation_keywords = [
            "legal",
            "lawsuit",
            "fraud",
            "breach",
            "lawyer",
            "attorney",
        ]
        if any(keyword in query.lower() for keyword in escalation_keywords):
            classification["urgency"] = "urgent"
            classification["requires_human"] = True
            classification["complexity"] = "complex"

        # Sentiment-based adjustments
        negative_words = [
            "angry",
            "frustrated",
            "terrible",
            "awful",
            "horrible",
            "hate",
        ]
        if any(word in query.lower() for word in negative_words):
            classification["sentiment"] = "negative"
            if classification["urgency"] == "low":
                classification["urgency"] = "medium"

        return classification

    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Fallback classification when LLM fails"""

        # Simple keyword-based classification
        query_lower = query.lower()

        # Determine intent
        intent = "support"  # default
        max_matches = 0

        for category, keywords in self.intent_categories.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                intent = category

        # Determine urgency
        urgency = "medium"  # default
        for level, keywords in self.urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                urgency = level
                break

        return {
            "intent": intent,
            "confidence": 0.6,  # Lower confidence for fallback
            "urgency": urgency,
            "complexity": "moderate",
            "entities": self._extract_entities(query),
            "sentiment": "neutral",
            "requires_human": False,
            "reasoning": "Fallback classification using keyword matching",
        }

    def create_agent(self) -> Agent:
        """Create and configure the CrewAI agent"""
        return Agent(
            role="Customer Query Analyst",
            goal="Accurately classify customer intents and determine the best response strategy",
            backstory="""You are an expert in understanding customer needs with years of experience 
            in customer service. You excel at quickly identifying what customers are trying to 
            accomplish and routing their requests appropriately. You can detect urgency levels, 
            assess complexity, and identify when human intervention is needed.""",
            llm=self.llm,
            tools=[self.intent_classification_tool],
            max_iter=3,
            verbose=True,
            allow_delegation=False,
        )
