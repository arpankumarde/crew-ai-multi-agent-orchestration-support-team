# Database Tools for CrewAI Agents
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import get_settings
from src.core.database import get_db_manager
from src.core.utils import sanitize_input, measure_execution_time


class DatabaseQueryInput(BaseModel):
    """Input schema for database query tool"""

    query_type: str = Field(
        ..., description="Type of query: customer, order, product, ticket"
    )
    parameters: Dict[str, Any] = Field(..., description="Query parameters")


class CustomerLookupInput(BaseModel):
    """Input schema for customer lookup tool"""

    customer_id: Optional[str] = Field(None, description="Customer ID to search for")
    email: Optional[str] = Field(None, description="Customer email to search for")
    search_term: Optional[str] = Field(
        None, description="General search term for name/email"
    )


class OrderLookupInput(BaseModel):
    """Input schema for order lookup tool"""

    order_id: Optional[str] = Field(None, description="Order ID to search for")
    customer_id: Optional[str] = Field(
        None, description="Customer ID to find orders for"
    )
    status: Optional[str] = Field(None, description="Order status filter")
    limit: int = Field(10, description="Maximum number of results")


class ProductLookupInput(BaseModel):
    """Input schema for product lookup tool"""

    product_id: Optional[str] = Field(None, description="Product ID to search for")
    product_name: Optional[str] = Field(None, description="Product name to search for")
    category: Optional[str] = Field(None, description="Product category filter")
    limit: int = Field(10, description="Maximum number of results")


class SupportTicketInput(BaseModel):
    """Input schema for support ticket tool"""

    ticket_id: Optional[str] = Field(None, description="Ticket ID to search for")
    customer_id: Optional[str] = Field(
        None, description="Customer ID to find tickets for"
    )
    status: Optional[str] = Field(None, description="Ticket status filter")
    category: Optional[str] = Field(None, description="Ticket category filter")
    limit: int = Field(10, description="Maximum number of results")


class DatabaseQueryTool(BaseTool):
    """
    Generic database query tool for executing various database operations.
    Supports customer, order, product, and support ticket queries.
    """

    name: str = "database_query_tool"
    description: str = """Execute database queries to retrieve customer, order, product, or support ticket information.
    Supports various query types with flexible parameters."""
    args_schema: type[BaseModel] = DatabaseQueryInput

    def __init__(self, **data):
        super().__init__(**data)
        # self.settings = get_settings()
        self._logger = logging.getLogger(__name__)

    @measure_execution_time
    async def _arun(self, query_type: str, parameters: Dict[str, Any]) -> str:
        """Execute database query asynchronously"""
        try:
            settings = get_settings()
            db_manager = await get_db_manager()

            # Sanitize query type
            query_type = sanitize_input(query_type.lower().strip())

            # Route to appropriate query method
            if query_type == "customer":
                result = await self._query_customers(db_manager, parameters)
            elif query_type == "order":
                result = await self._query_orders(db_manager, parameters)
            elif query_type == "product":
                result = await self._query_products(db_manager, parameters)
            elif query_type == "ticket":
                result = await self._query_tickets(db_manager, parameters)
            else:
                return f"Error: Unsupported query type '{query_type}'. Supported types: customer, order, product, ticket"

            if "error" in result:
                return f"Database query failed: {result['error']}"

            return f"Query successful. Results: {result}"

        except Exception as e:
            self.logger.error(f"Database query tool error: {e}")
            return f"Database query failed with error: {str(e)}"

    def _run(self, query_type: str, parameters: Dict[str, Any]) -> str:
        """Synchronous wrapper for async database query"""
        return asyncio.run(self._arun(query_type, parameters))

    async def _query_customers(
        self, db_manager, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query customer data"""
        try:
            customer_id = parameters.get("customer_id")
            email = parameters.get("email")
            search_term = parameters.get("search_term")

            if customer_id:
                customer = await db_manager.get_customer_by_id(customer_id)
                if not customer:
                    return {"error": f"Customer {customer_id} not found"}

                # Get additional customer context
                orders = await db_manager.get_customer_orders(customer_id, limit=5)
                tickets = await db_manager.get_customer_tickets(customer_id, limit=3)

                return {
                    "customer": dict(customer),
                    "recent_orders": [dict(order) for order in orders],
                    "recent_tickets": [dict(ticket) for ticket in tickets],
                    "total_orders": len(orders),
                    "total_tickets": len(tickets),
                }

            elif email:
                customer = await db_manager.get_customer_by_email(email)
                if not customer:
                    return {"error": f"Customer with email {email} not found"}

                return {"customer": dict(customer)}

            elif search_term:
                customers = await db_manager.search_customers(search_term, limit=10)
                return {
                    "customers": [dict(customer) for customer in customers],
                    "count": len(customers),
                }
            else:
                return {"error": "Must provide customer_id, email, or search_term"}

        except Exception as e:
            return {"error": f"Customer query failed: {str(e)}"}

    async def _query_orders(
        self, db_manager, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query order data"""
        try:
            order_id = parameters.get("order_id")
            customer_id = parameters.get("customer_id")
            status = parameters.get("status")
            limit = parameters.get("limit", 10)

            if order_id:
                order = await db_manager.get_order_by_id(order_id)
                if not order:
                    return {"error": f"Order {order_id} not found"}
                return {"order": dict(order)}

            elif customer_id:
                orders = await db_manager.get_customer_orders(customer_id, limit=limit)
                return {
                    "orders": [dict(order) for order in orders],
                    "count": len(orders),
                }

            elif status:
                orders = await db_manager.get_orders_by_status(status, limit=limit)
                return {
                    "orders": [dict(order) for order in orders],
                    "count": len(orders),
                    "status": status,
                }
            else:
                return {"error": "Must provide order_id, customer_id, or status"}

        except Exception as e:
            return {"error": f"Order query failed: {str(e)}"}

    async def _query_products(
        self, db_manager, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query product data"""
        try:
            product_id = parameters.get("product_id")
            product_name = parameters.get("product_name")
            category = parameters.get("category")
            limit = parameters.get("limit", 10)

            if product_id:
                product = await db_manager.get_product_by_id(product_id)
                if not product:
                    return {"error": f"Product {product_id} not found"}
                return {"product": dict(product)}

            elif product_name:
                products = await db_manager.search_products(
                    product_name, category=category, limit=limit
                )
                return {
                    "products": [dict(product) for product in products],
                    "count": len(products),
                }

            elif category:
                products = await db_manager.get_products_by_category(
                    category, limit=limit
                )
                return {
                    "products": [dict(product) for product in products],
                    "count": len(products),
                    "category": category,
                }
            else:
                return {"error": "Must provide product_id, product_name, or category"}

        except Exception as e:
            return {"error": f"Product query failed: {str(e)}"}

    async def _query_tickets(
        self, db_manager, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query support ticket data"""
        try:
            ticket_id = parameters.get("ticket_id")
            customer_id = parameters.get("customer_id")
            status = parameters.get("status")
            limit = parameters.get("limit", 10)

            if ticket_id:
                ticket = await db_manager.get_ticket_by_id(ticket_id)
                if not ticket:
                    return {"error": f"Ticket {ticket_id} not found"}
                return {"ticket": dict(ticket)}

            elif customer_id:
                tickets = await db_manager.get_customer_tickets(
                    customer_id, status=status, limit=limit
                )
                return {
                    "tickets": [dict(ticket) for ticket in tickets],
                    "count": len(tickets),
                }
            else:
                return {"error": "Must provide ticket_id or customer_id"}

        except Exception as e:
            return {"error": f"Ticket query failed: {str(e)}"}


class CustomerLookupTool(BaseTool):
    """
    Specialized tool for customer lookup operations.
    Provides detailed customer information with related orders and tickets.
    """

    name: str = "customer_lookup_tool"
    description: str = """Look up customer information by ID, email, or search term.
    Returns customer details along with order history and support tickets."""
    args_schema: type[BaseModel] = CustomerLookupInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logging.getLogger(__name__)

    @measure_execution_time
    async def _arun(
        self,
        customer_id: Optional[str] = None,
        email: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> str:
        """Look up customer information"""
        try:
            if not any([customer_id, email, search_term]):
                return "Error: Must provide customer_id, email, or search_term"

            db_manager = await get_db_manager()

            # Use CSV data for simulation (replace with actual DB calls in production)
            customers_df = pd.read_csv("./data/customers.csv")
            orders_df = pd.read_csv("./data/orders.csv")
            tickets_df = pd.read_csv("./data/support_tickets.csv")

            if customer_id:
                customer_data = customers_df[customers_df["customer_id"] == customer_id]
                if customer_data.empty:
                    return f"Customer {customer_id} not found"

                customer = customer_data.iloc[0].to_dict()

                # Get related orders and tickets
                orders = orders_df[orders_df["customer_id"] == customer_id].head(5)
                tickets = tickets_df[tickets_df["customer_id"] == customer_id].head(3)

                result = {
                    "customer": customer,
                    "recent_orders": (
                        orders.to_dict("records") if not orders.empty else []
                    ),
                    "recent_tickets": (
                        tickets.to_dict("records") if not tickets.empty else []
                    ),
                    "total_orders": len(
                        orders_df[orders_df["customer_id"] == customer_id]
                    ),
                    "total_tickets": len(
                        tickets_df[tickets_df["customer_id"] == customer_id]
                    ),
                }

                return f"Customer found: {result}"

            elif email:
                customer_data = customers_df[customers_df["email"] == email]
                if customer_data.empty:
                    return f"Customer with email {email} not found"

                customer = customer_data.iloc[0].to_dict()
                return f"Customer found: {customer}"

            elif search_term:
                search_term = sanitize_input(search_term)
                mask = (
                    customers_df["first_name"].str.contains(
                        search_term, case=False, na=False
                    )
                    | customers_df["last_name"].str.contains(
                        search_term, case=False, na=False
                    )
                    | customers_df["email"].str.contains(
                        search_term, case=False, na=False
                    )
                )
                customers = customers_df[mask].head(10)

                if customers.empty:
                    return f"No customers found matching '{search_term}'"

                return (
                    f"Found {len(customers)} customers: {customers.to_dict('records')}"
                )

        except Exception as e:
            self.logger.error(f"Customer lookup error: {e}")
            return f"Customer lookup failed: {str(e)}"

    def _run(
        self,
        customer_id: Optional[str] = None,
        email: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(customer_id, email, search_term))


class OrderLookupTool(BaseTool):
    """
    Specialized tool for order lookup operations.
    Provides detailed order information with customer context.
    """

    name: str = "order_lookup_tool"
    description: str = """Look up order information by order ID, customer ID, or status.
    Returns order details with customer information and tracking status."""
    args_schema: type[BaseModel] = OrderLookupInput

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)

    @measure_execution_time
    async def _arun(
        self,
        order_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Look up order information"""
        try:
            # Use CSV data for simulation
            orders_df = pd.read_csv("./data/orders.csv")
            customers_df = pd.read_csv("./data/customers.csv")

            if order_id:
                order_data = orders_df[orders_df["order_id"] == order_id]
                if order_data.empty:
                    return f"Order {order_id} not found"

                order = order_data.iloc[0].to_dict()

                # Get customer information
                customer_data = customers_df[
                    customers_df["customer_id"] == order["customer_id"]
                ]
                customer = (
                    customer_data.iloc[0].to_dict() if not customer_data.empty else {}
                )

                result = {"order": order, "customer": customer}

                return f"Order found: {result}"

            elif customer_id:
                orders = orders_df[orders_df["customer_id"] == customer_id].head(limit)
                if orders.empty:
                    return f"No orders found for customer {customer_id}"

                result = {"orders": orders.to_dict("records"), "count": len(orders)}

                return f"Found {len(orders)} orders for customer: {result}"

            elif status:
                orders = orders_df[orders_df["status"] == status].head(limit)
                if orders.empty:
                    return f"No orders found with status '{status}'"

                result = {
                    "orders": orders.to_dict("records"),
                    "count": len(orders),
                    "status": status,
                }

                return f"Found {len(orders)} orders with status '{status}': {result}"

            else:
                return "Error: Must provide order_id, customer_id, or status"

        except Exception as e:
            self.logger.error(f"Order lookup error: {e}")
            return f"Order lookup failed: {str(e)}"

    def _run(
        self,
        order_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(order_id, customer_id, status, limit))


class ProductLookupTool(BaseTool):
    """
    Specialized tool for product lookup operations.
    Provides product details, availability, and pricing information.
    """

    name: str = "product_lookup_tool"
    description: str = """Look up product information by ID, name, or category.
    Returns product details including availability, pricing, and specifications."""
    args_schema: type[BaseModel] = ProductLookupInput

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @measure_execution_time
    async def _arun(
        self,
        product_id: Optional[str] = None,
        product_name: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Look up product information"""
        try:
            # Use CSV data for simulation
            products_df = pd.read_csv("./data/products.csv")

            if product_id:
                product_data = products_df[products_df["product_id"] == product_id]
                if product_data.empty:
                    return f"Product {product_id} not found"

                product = product_data.iloc[0].to_dict()

                # Add availability status
                product["availability_status"] = (
                    "In Stock" if product["stock_quantity"] > 0 else "Out of Stock"
                )
                product["stock_level"] = (
                    "High"
                    if product["stock_quantity"] > 100
                    else "Low" if product["stock_quantity"] > 0 else "None"
                )

                return f"Product found: {product}"

            elif product_name:
                mask = products_df["name"].str.contains(
                    product_name, case=False, na=False
                )
                if category:
                    mask = mask & (products_df["category"] == category)

                products = products_df[mask & products_df["is_active"]].head(limit)

                if products.empty:
                    return f"No products found matching '{product_name}'"

                # Add availability status for each product
                result = []
                for _, product in products.iterrows():
                    product_dict = product.to_dict()
                    product_dict["availability_status"] = (
                        "In Stock" if product["stock_quantity"] > 0 else "Out of Stock"
                    )
                    product_dict["stock_level"] = (
                        "High"
                        if product["stock_quantity"] > 100
                        else "Low" if product["stock_quantity"] > 0 else "None"
                    )
                    result.append(product_dict)

                return f"Found {len(result)} products: {result}"

            elif category:
                products = products_df[
                    (products_df["category"] == category) & products_df["is_active"]
                ].head(limit)

                if products.empty:
                    return f"No products found in category '{category}'"

                result = products.to_dict("records")
                return (
                    f"Found {len(result)} products in category '{category}': {result}"
                )

            else:
                return "Error: Must provide product_id, product_name, or category"

        except Exception as e:
            self.logger.error(f"Product lookup error: {e}")
            return f"Product lookup failed: {str(e)}"

    def _run(
        self,
        product_id: Optional[str] = None,
        product_name: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(product_id, product_name, category, limit))


class SupportTicketTool(BaseTool):
    """
    Specialized tool for support ticket operations.
    Provides ticket details, history, and status information.
    """

    name: str = "support_ticket_tool"
    description: str = """Look up support ticket information by ticket ID, customer ID, status, or category.
    Returns ticket details with customer context and resolution status."""
    args_schema: type[BaseModel] = SupportTicketInput

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @measure_execution_time
    async def _arun(
        self,
        ticket_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Look up support ticket information"""
        try:
            # Use CSV data for simulation
            tickets_df = pd.read_csv("./data/support_tickets.csv")
            customers_df = pd.read_csv("./data/customers.csv")

            if ticket_id:
                ticket_data = tickets_df[tickets_df["ticket_id"] == ticket_id]
                if ticket_data.empty:
                    return f"Ticket {ticket_id} not found"

                ticket = ticket_data.iloc[0].to_dict()

                # Get customer information
                customer_data = customers_df[
                    customers_df["customer_id"] == ticket["customer_id"]
                ]
                customer = (
                    customer_data.iloc[0].to_dict() if not customer_data.empty else {}
                )

                result = {"ticket": ticket, "customer": customer}

                return f"Ticket found: {result}"

            elif customer_id:
                mask = tickets_df["customer_id"] == customer_id
                if status:
                    mask = mask & (tickets_df["status"] == status)
                if category:
                    mask = mask & (tickets_df["category"] == category)

                tickets = tickets_df[mask].head(limit)

                if tickets.empty:
                    return f"No tickets found for customer {customer_id}"

                # Get statistics
                stats = {
                    "total_tickets": len(tickets),
                    "status_breakdown": tickets["status"].value_counts().to_dict(),
                    "category_breakdown": tickets["category"].value_counts().to_dict(),
                }

                result = {"tickets": tickets.to_dict("records"), "statistics": stats}

                return f"Found {len(tickets)} tickets for customer: {result}"

            elif status or category:
                mask = pd.Series([True] * len(tickets_df))
                if status:
                    mask = mask & (tickets_df["status"] == status)
                if category:
                    mask = mask & (tickets_df["category"] == category)

                tickets = tickets_df[mask].head(limit)

                if tickets.empty:
                    filter_desc = f"status '{status}'" if status else ""
                    if category:
                        filter_desc += (
                            f"{' and ' if filter_desc else ''}category '{category}'"
                        )
                    return f"No tickets found with {filter_desc}"

                result = {"tickets": tickets.to_dict("records"), "count": len(tickets)}

                return f"Found {len(tickets)} tickets: {result}"

            else:
                return "Error: Must provide ticket_id, customer_id, status, or category"

        except Exception as e:
            self.logger.error(f"Support ticket lookup error: {e}")
            return f"Support ticket lookup failed: {str(e)}"

    def _run(
        self,
        ticket_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(ticket_id, customer_id, status, category, limit))


# Pydantic v2 Compatible Features:

# BaseModel schemas with proper Field descriptions

# Generic DatabaseQueryTool for flexible database operations

# Specialized tools for different data types:

# CustomerLookupTool: Customer information with order/ticket context

# OrderLookupTool: Order details with customer information

# ProductLookupTool: Product details with availability status

# SupportTicketTool: Support ticket management with statistics

# Key Capabilities:

# CSV-based data simulation (ready for PostgreSQL integration)

# Async/sync execution support with asyncio.run() wrappers

# Input sanitization and validation

# Error handling with detailed logging

# Performance monitoring with execution time measurement
