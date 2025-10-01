# Agent Factory Functions
"""
Factory functions to create agent instances for the CrewAI workflow.
These functions provide a clean interface for agent initialization.
"""

from .intent_classifier import IntentClassifierAgent
from .knowledge_retriever import KnowledgeRetrieverAgent
from .database_agent import DatabaseAgent
from .response_generator import ResponseGeneratorAgent
from .escalation_agent import EscalationAgent
from .quality_assurance import QualityAssuranceAgent


def create_intent_classifier():
    """Create and return an Intent Classifier agent"""
    agent_instance = IntentClassifierAgent()
    return agent_instance.create_agent()


def create_knowledge_retriever():
    """Create and return a Knowledge Retriever agent"""
    agent_instance = KnowledgeRetrieverAgent()
    return agent_instance.create_agent()


def create_database_agent():
    """Create and return a Database agent"""
    agent_instance = DatabaseAgent()
    return agent_instance.create_agent()


def create_response_generator():
    """Create and return a Response Generator agent"""
    agent_instance = ResponseGeneratorAgent()
    return agent_instance.create_agent()


def create_escalation_agent():
    """Create and return an Escalation agent"""
    agent_instance = EscalationAgent()
    return agent_instance.create_agent()


def create_quality_assurance():
    """Create and return a Quality Assurance agent"""
    agent_instance = QualityAssuranceAgent()
    return agent_instance.create_agent()


# Export all factory functions
__all__ = [
    "create_intent_classifier",
    "create_knowledge_retriever",
    "create_database_agent",
    "create_response_generator",
    "create_escalation_agent",
    "create_quality_assurance",
]
