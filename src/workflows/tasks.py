# CrewAI Tasks Definition
"""
Task definitions for the customer service multi-agent workflow.
Each task corresponds to a specific step in the customer service process.
"""

from crewai import Task
from typing import Dict, Any, List


def create_tasks(agents: Dict[str, Any]) -> List[Task]:
    """
    Create all tasks for the customer service workflow.

    Args:
        agents: Dictionary of agent instances

    Returns:
        List of CrewAI Task objects
    """

    # Task 1: Intent Classification
    classify_intent_task = Task(
        description="""
        Analyze the customer query and classify the intent. Determine:
        1. Primary intent category (billing, technical, account, product, etc.)
        2. Urgency level (low, medium, high, urgent)
        3. Complexity assessment (simple, moderate, complex)
        4. Confidence score (0.0 to 1.0)
        5. Recommended next steps
        
        Consider the tone, specific keywords, and context clues in the customer message.
        Extract any entities like order numbers, email addresses, or product names.
        
        Customer Query: {customer_query}
        Customer ID: {customer_id}
        """,
        expected_output="""
        A structured classification including:
        - Intent category with confidence score
        - Urgency and complexity levels
        - Key entities mentioned (product names, order numbers, etc.)
        - Recommended routing strategy
        - JSON formatted result with all classification details
        """,
        agent=agents["intent_classifier"],
    )

    # Task 2: Knowledge Base Search
    # Focus on the intent: {intent} with urgency level: {urgency}
    retrieve_knowledge_task = Task(
        description="""
        Search the knowledge base for information relevant to the customer query.
        Use the classified intent to focus your search. Find:
        1. Relevant FAQ articles
        2. Product documentation
        3. Troubleshooting guides
        4. Company policies
        5. Step-by-step procedures
        
        Rank results by relevance and include source references.
        
        Use the classified intent and urgency from previous steps to focus your search.
        
        Customer Query: {customer_query}
        """,
        expected_output="""
        A comprehensive list of relevant knowledge base articles with:
        - Article titles and summaries
        - Relevance scores and categories
        - Key excerpts that address the customer query
        - Source references for citation
        - Recommended knowledge for response generation
        """,
        agent=agents["knowledge_retriever"],
        context=[classify_intent_task],
    )

    # Task 3: Database Query and Data Retrieval
    # Intent: {intent}
    # Entities found: {entities}
    query_database_task = Task(
        description="""
        Based on the classified intent and customer information, query the database for:
        1. Customer account details (if customer ID provided)
        2. Order status and history
        3. Product information and availability
        4. Billing and payment records
        5. Previous support ticket history
        
        Only retrieve information that's relevant to addressing the customer query.
        
        Customer ID: {customer_id}
        """,
        expected_output="""
        Structured data including:
        - Customer account information (if applicable)
        - Relevant order details and status
        - Product specifications or availability
        - Historical context from previous interactions
        - Any relevant billing or payment information
        """,
        agent=agents["database_agent"],
        context=[classify_intent_task],
    )

    # Task 4: Response Generation
    #  urgency level: {urgency} and 
        # Consider the sentiment: {sentiment}
    generate_response_task = Task(
        description="""
        Synthesize information from knowledge base and database queries to create a comprehensive response.
        The response should:
        1. Directly address the customer's question or concern
        2. Provide clear, actionable steps when applicable
        3. Include relevant details from their account/orders
        4. Maintain a professional, empathetic tone
        5. Anticipate follow-up questions
        
        If information is insufficient, indicate what additional details are needed.
        
        Customer Query: {customer_query}
        """,
        expected_output="""
        A complete customer response including:
        - Direct answer to the customer query
        - Step-by-step instructions (if applicable)
        - Relevant account/order information
        - Proactive additional information
        - Clear next steps or call-to-action
        - Professional and empathetic tone throughout
        """,
        agent=agents["response_generator"],
        context=[classify_intent_task, retrieve_knowledge_task, query_database_task],
    )

    # Task 5: Escalation Assessment
    # Intent: {intent}
    # Urgency: {urgency}
    assess_escalation_task = Task(
        description="""
        Evaluate whether the generated response adequately addresses the customer query or if escalation is needed.
        Consider:
        1. Response completeness and accuracy
        2. Query complexity and customer frustration level
        3. Account status (VIP customers, enterprise accounts)
        4. Issue severity and business impact
        5. Policy exceptions or edge cases
        
        Provide escalation recommendations with reasoning.
        

        Customer ID: {customer_id}
        """,
        expected_output="""
        Escalation assessment including:
        - Escalation decision (yes/no) with confidence score
        - Detailed reasoning for the decision
        - Recommended escalation path (if applicable)
        - Priority level assignment
        - Suggested human agent specialization
        - SLA considerations and timing
        """,
        agent=agents["escalation_agent"],
        context=[classify_intent_task, generate_response_task],
    )

    # Task 6: Quality Assurance
    # Intent: {intent}
        # Response to review: {generated_response}
    quality_check_task = Task(
        description="""
        Review the generated response for quality, accuracy, and compliance:
        1. Factual accuracy against knowledge base
        2. Tone and professionalism
        3. Brand guideline compliance
        4. Completeness of information
        5. Grammar and clarity
        6. Appropriate length and structure
        
        Suggest improvements if needed or approve for delivery.
        
       
        """,
        expected_output="""
        Quality assessment including:
        - Overall quality score (1-10)
        - Specific issues identified (if any)
        - Suggested improvements or corrections
        - Compliance verification results
        - Final approval status and recommendations
        - Delivery readiness assessment
        """,
        agent=agents["quality_assurance"],
        context=[generate_response_task, assess_escalation_task],
    )

    return [
        classify_intent_task,
        retrieve_knowledge_task,
        query_database_task,
        generate_response_task,
        assess_escalation_task,
        quality_check_task,
    ]


def create_simple_tasks(agents: Dict[str, Any]) -> List[Task]:
    """
    Create a simplified task workflow for testing or lightweight processing.

    Args:
        agents: Dictionary of agent instances

    Returns:
        List of simplified CrewAI Task objects
    """

    # Simplified intent classification
    simple_classify_task = Task(
        description="Analyze customer query: {customer_query} and classify intent, urgency, and complexity.",
        expected_output="Intent classification with category, urgency level, and confidence score.",
        agent=agents["intent_classifier"],
    )

    # Simplified response generation
    simple_response_task = Task(
        description="Generate a helpful response to: {customer_query} based on the classified intent.",
        expected_output="A clear, helpful response addressing the customer's query.",
        agent=agents["response_generator"],
        context=[simple_classify_task],
    )

    return [simple_classify_task, simple_response_task]


# Task configuration templates
TASK_TEMPLATES = {
    "urgent": {
        "max_execution_time": 60,  # seconds
        "priority": "high",
        "escalation_threshold": 0.3,
    },
    "standard": {
        "max_execution_time": 120,  # seconds
        "priority": "normal",
        "escalation_threshold": 0.5,
    },
    "low_priority": {
        "max_execution_time": 300,  # seconds
        "priority": "low",
        "escalation_threshold": 0.7,
    },
}
