# Database Agent
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from src.core.config import settings
from src.core.database import DatabaseManager
from src.tools.database_tools import (
    DatabaseQueryTool,
    CustomerLookupTool,
    OrderLookupTool,
)


class DatabaseAgent:
    """
    Agent responsible for querying the PostgreSQL database to retrieve
    customer, order, product, and support ticket information.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.ai_models.openai_model,
            temperature=0.1,
            api_key=settings.ai_models.openai_api_key,
        )

        self.db_manager = DatabaseManager()
        self.query_tool = DatabaseQueryTool()
        self.customer_lookup = CustomerLookupTool()
        self.order_lookup = OrderLookupTool()

    @tool
    def database_query_tool(
        self, query_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute database queries based on query type and parameters.

        Args:
            query_type: Type of query ('customer', 'order', 'product', 'ticket')
            parameters: Query parameters (customer_id, order_id, etc.)

        Returns:
            Dictionary containing query results
        """
        try:
            if query_type == "customer":
                return self._query_customer_data(parameters)
            elif query_type == "order":
                return self._query_order_data(parameters)
            elif query_type == "product":
                return self._query_product_data(parameters)
            elif query_type == "ticket":
                return self._query_ticket_data(parameters)
            else:
                return {"error": f"Unknown query type: {query_type}"}

        except Exception as e:
            print(f"Database query error: {e}")
            return {"error": str(e), "query_type": query_type}

    @tool
    def customer_lookup_tool(
        self, customer_id: Optional[str] = None, email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Look up customer information by ID or email.

        Args:
            customer_id: Customer ID to search for
            email: Customer email to search for

        Returns:
            Dictionary containing customer information
        """
        try:
            return self.customer_lookup.get_customer_info(
                customer_id=customer_id, email=email
            )
        except Exception as e:
            print(f"Customer lookup error: {e}")
            return {"error": str(e)}

    @tool
    def order_lookup_tool(
        self, order_id: Optional[str] = None, customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Look up order information by order ID or customer ID.

        Args:
            order_id: Order ID to search for
            customer_id: Customer ID to find orders for

        Returns:
            Dictionary containing order information
        """
        try:
            return self.order_lookup.get_order_info(
                order_id=order_id, customer_id=customer_id
            )
        except Exception as e:
            print(f"Order lookup error: {e}")
            return {"error": str(e)}

    def _query_customer_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query customer data from the database"""
        customer_id = parameters.get("customer_id")
        email = parameters.get("email")

        if not customer_id and not email:
            return {"error": "Customer ID or email required"}

        try:
            # Load customer data (simulated with CSV for now)
            customers_df = pd.read_csv("./data/customers.csv")

            if customer_id:
                customer_data = customers_df[customers_df["customer_id"] == customer_id]
            else:
                customer_data = customers_df[customers_df["email"] == email]

            if customer_data.empty:
                return {"error": "Customer not found"}

            customer_record = customer_data.iloc[0].to_dict()

            # Also get order history
            orders_df = pd.read_csv("./data/orders.csv")
            customer_orders = orders_df[
                orders_df["customer_id"] == customer_record["customer_id"]
            ]

            # Get support ticket history
            tickets_df = pd.read_csv("./data/support_tickets.csv")
            customer_tickets = tickets_df[
                tickets_df["customer_id"] == customer_record["customer_id"]
            ]

            return {
                "customer": customer_record,
                "order_count": len(customer_orders),
                "recent_orders": customer_orders.head(5).to_dict("records"),
                "support_tickets": len(customer_tickets),
                "recent_tickets": customer_tickets.head(3).to_dict("records"),
            }

        except Exception as e:
            return {"error": f"Error querying customer data: {e}"}

    def _query_order_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query order data from the database"""
        order_id = parameters.get("order_id")
        customer_id = parameters.get("customer_id")

        try:
            orders_df = pd.read_csv("./data/orders.csv")

            if order_id:
                order_data = orders_df[orders_df["order_id"] == order_id]
                if order_data.empty:
                    return {"error": f"Order {order_id} not found"}

                order_record = order_data.iloc[0].to_dict()

                # Get customer info for this order
                customers_df = pd.read_csv("./data/customers.csv")
                customer_data = customers_df[
                    customers_df["customer_id"] == order_record["customer_id"]
                ]
                customer_info = (
                    customer_data.iloc[0].to_dict() if not customer_data.empty else {}
                )

                return {"order": order_record, "customer": customer_info}

            elif customer_id:
                customer_orders = orders_df[orders_df["customer_id"] == customer_id]
                if customer_orders.empty:
                    return {"error": "No orders found for this customer"}

                return {
                    "orders": customer_orders.to_dict("records"),
                    "total_orders": len(customer_orders),
                    "order_statuses": customer_orders["status"]
                    .value_counts()
                    .to_dict(),
                }
            else:
                return {"error": "Order ID or Customer ID required"}

        except Exception as e:
            return {"error": f"Error querying order data: {e}"}

    def _query_product_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query product data from the database"""
        product_name = parameters.get("product_name")
        product_id = parameters.get("product_id")
        category = parameters.get("category")

        try:
            products_df = pd.read_csv("./data/products.csv")

            if product_id:
                product_data = products_df[products_df["product_id"] == product_id]
            elif product_name:
                product_data = products_df[
                    products_df["name"].str.contains(product_name, case=False, na=False)
                ]
            elif category:
                product_data = products_df[
                    products_df["category"].str.contains(category, case=False, na=False)
                ]
            else:
                # Return all active products
                product_data = products_df[products_df["is_active"] == True].head(10)

            if product_data.empty:
                return {"error": "No products found matching criteria"}

            products_list = product_data.to_dict("records")

            # Add availability status
            for product in products_list:
                product["availability_status"] = (
                    "In Stock" if product["stock_quantity"] > 0 else "Out of Stock"
                )
                product["stock_level"] = (
                    "High"
                    if product["stock_quantity"] > 100
                    else "Low" if product["stock_quantity"] > 0 else "None"
                )

            return {"products": products_list, "total_found": len(products_list)}

        except Exception as e:
            return {"error": f"Error querying product data: {e}"}

    def _query_ticket_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query support ticket data from the database"""
        customer_id = parameters.get("customer_id")
        ticket_id = parameters.get("ticket_id")
        status = parameters.get("status")

        try:
            tickets_df = pd.read_csv("./data/support_tickets.csv")

            if ticket_id:
                ticket_data = tickets_df[tickets_df["ticket_id"] == ticket_id]
                if ticket_data.empty:
                    return {"error": f"Ticket {ticket_id} not found"}

                ticket_record = ticket_data.iloc[0].to_dict()

                # Get customer info
                customers_df = pd.read_csv("./data/customers.csv")
                customer_data = customers_df[
                    customers_df["customer_id"] == ticket_record["customer_id"]
                ]
                customer_info = (
                    customer_data.iloc[0].to_dict() if not customer_data.empty else {}
                )

                return {"ticket": ticket_record, "customer": customer_info}

            elif customer_id:
                customer_tickets = tickets_df[tickets_df["customer_id"] == customer_id]

                if status:
                    customer_tickets = customer_tickets[
                        customer_tickets["status"] == status
                    ]

                if customer_tickets.empty:
                    return {"error": "No tickets found for this customer"}

                return {
                    "tickets": customer_tickets.to_dict("records"),
                    "total_tickets": len(customer_tickets),
                    "ticket_statuses": customer_tickets["status"]
                    .value_counts()
                    .to_dict(),
                    "categories": customer_tickets["category"].value_counts().to_dict(),
                }
            else:
                return {"error": "Customer ID or Ticket ID required"}

        except Exception as e:
            return {"error": f"Error querying ticket data: {e}"}

    def create_agent(self) -> Agent:
        """Create and configure the CrewAI agent"""
        return Agent(
            role="Data Analyst",
            goal="Retrieve specific customer, order, and product information from the database",
            backstory="""You are a skilled data analyst who can quickly query databases to find 
            specific information about customers, orders, products, and transactions. You understand 
            database relationships and can efficiently locate the information needed to help customers. 
            You always ensure data accuracy and protect customer privacy.""",
            llm=self.llm,
            tools=[
                self.database_query_tool,
                self.customer_lookup_tool,
                self.order_lookup_tool,
            ],
            max_iter=3,
            verbose=True,
            allow_delegation=False,
        )


# Role: Structured data retrieval from PostgreSQL

# Capabilities:

# Customer account lookups

# Order status and history queries

# Product information retrieval

# Support ticket history analysis

# Key Features:

# CSV-based data simulation (ready for PostgreSQL integration)

# Multi-parameter queries (customer_id, order_id, email)

# Related data aggregation (orders per customer, ticket history)

# Error handling and data validation
