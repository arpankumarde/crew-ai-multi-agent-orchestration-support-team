import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from crewai import Crew, Process

# Import factory functions from agents
from src.agents.factories import (
    create_intent_classifier,
    create_knowledge_retriever,
    create_database_agent,
    create_response_generator,
    create_escalation_agent,
    create_quality_assurance,
)

# Import core modules
from src.core.config import get_settings
from src.core.utils import setup_logging, generate_request_id
from src.workflows.tasks import create_tasks

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()


class CustomerServiceCrew:
    """
    Main CustomerService Crew class that orchestrates multi-agent workflow
    for processing customer queries through intent classification, knowledge retrieval,
    database queries, response generation, escalation assessment, and quality assurance.
    """

    def __init__(self):
        """Initialize the crew with agents and tasks"""
        self.settings = get_settings()
        self.agents = self._create_agents()
        self.tasks = create_tasks(self.agents)
        self.crew = self._create_crew()

    def _create_agents(self):
        """Create all agents for the customer service workflow"""
        try:
            agents = {
                "intent_classifier": create_intent_classifier(),
                "knowledge_retriever": create_knowledge_retriever(),
                "database_agent": create_database_agent(),
                "response_generator": create_response_generator(),
                "escalation_agent": create_escalation_agent(),
                "quality_assurance": create_quality_assurance(),
            }
            print(f"✅ Successfully created {len(agents)} agents")
            return agents
        except Exception as e:
            print(f"❌ Error creating agents: {e}")
            raise

    def _create_crew(self):
        """Create the CrewAI crew with agents and tasks"""
        try:
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=self.tasks,
                process=Process.sequential,
                verbose=True,
                memory=True,
            )
            print("✅ CrewAI crew created successfully")
            return crew
        except Exception as e:
            print(f"❌ Error creating crew: {e}")
            raise

    def process_query(
        self, customer_query: str, customer_id: str = None, session_id: str = None
    ):
        """
        Process a customer query through the multi-agent workflow

        Args:
            customer_query: The customer's question or request
            customer_id: Optional customer identifier
            session_id: Optional session identifier for tracking

        Returns:
            Dictionary with processing results
        """
        # Generate request ID for tracking
        request_id = generate_request_id()

        # Prepare inputs for the crew
        inputs = {
            "customer_query": customer_query,
            "customer_id": customer_id or "unknown",
            "session_id": session_id or f"session_{request_id}",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"🔄 Processing query (ID: {request_id}): {customer_query[:100]}...")

        try:
            # Execute the crew workflow
            result = self.crew.kickoff(inputs=inputs)

            return {
                "success": True,
                "request_id": request_id,
                "response": str(result) if result else "No response generated",
                "metadata": {
                    "session_id": inputs["session_id"],
                    "customer_id": inputs["customer_id"],
                    "timestamp": inputs["timestamp"],
                    "agents_involved": [agent.role for agent in self.agents.values()],
                    "workflow_completed": True,
                },
            }

        except Exception as e:
            print(f"❌ Error processing query (ID: {request_id}): {e}")
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "fallback_response": "I apologize, but I'm experiencing technical difficulties. Please contact our support team directly for assistance.",
                "metadata": {
                    "session_id": inputs["session_id"],
                    "customer_id": inputs["customer_id"],
                    "timestamp": inputs["timestamp"],
                    "workflow_completed": False,
                },
            }

    def health_check(self):
        """Perform health check on the crew and agents"""
        try:
            health_status = {
                "crew_status": "healthy" if self.crew else "unhealthy",
                "agents_count": len(self.agents),
                "agents_status": {},
                "tasks_count": len(self.tasks) if self.tasks else 0,
            }

            # Check each agent
            for agent_name, agent in self.agents.items():
                try:
                    health_status["agents_status"][agent_name] = {
                        "status": "healthy",
                        "role": agent.role,
                        "has_tools": (
                            len(agent.tools) > 0 if hasattr(agent, "tools") else False
                        ),
                    }
                except Exception as e:
                    health_status["agents_status"][agent_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }

            return health_status

        except Exception as e:
            return {"crew_status": "unhealthy", "error": str(e)}


def main():
    """Main function to run example queries"""
    try:
        print("🚀 Initializing CrewAI Customer Service System...")

        # Initialize the customer service crew
        crew = CustomerServiceCrew()

        # Perform health check
        health = crew.health_check()
        print(f"🏥 Health Check: {health}")

        # Example customer queries
        test_queries = [
            {
                "query": "I can't log into my account and need help resetting my password",
                "customer_id": "cust_12345",
                "description": "Account Access Issue",
            },
            {
                "query": "What's the status of my order ORD-789456?",
                "customer_id": "cust_67890",
                "description": "Order Status Inquiry",
            },
            {
                "query": "I was charged twice for my subscription this month",
                "customer_id": "cust_11111",
                "description": "Billing Issue",
            },
            {
                "query": "How do I integrate your API with my application?",
                "customer_id": "cust_22222",
                "description": "Technical Documentation Request",
            },
        ]

        # Process each query
        results = []
        for i, query_data in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"🔍 Processing Query {i}: {query_data['description']}")
            print(f"Query: {query_data['query']}")
            print(f"{'='*60}")

            result = crew.process_query(
                customer_query=query_data["query"],
                customer_id=query_data["customer_id"],
                session_id=f"test_session_{i}",
            )

            results.append(result)

            if result["success"]:
                print("✅ Response Generated Successfully!")
                print(f"📋 Request ID: {result['request_id']}")
                print(f"📝 Response: {result['response'][:200]}...")
                print(f"📊 Metadata: {result['metadata']}")
            else:
                print("❌ Error Processing Query")
                print(f"📋 Request ID: {result['request_id']}")
                print(f"🚨 Error: {result['error']}")
                print(f"🔄 Fallback: {result['fallback_response']}")

            print(f"\n{'-'*60}")

        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\n📈 Processing Summary:")
        print(f"   Total Queries: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        print(f"   Success Rate: {(successful/len(results)*100):.1f}%")

    except Exception as e:
        print(f"💥 Fatal error in main: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
