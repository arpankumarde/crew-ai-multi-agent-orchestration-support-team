# Database Management for PostgreSQL
import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import pandas as pd
import asyncpg
from asyncpg import Pool, Connection
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Boolean,
    Text,
    Numeric,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from .config import get_settings

# SQLAlchemy Base
Base = declarative_base()


# Database Models
class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    subscription_status = Column(String(50), default="active")
    customer_tier = Column(String(20), default="bronze")


class Product(Base):
    __tablename__ = "products"

    product_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    category = Column(String(100))
    description = Column(Text)
    price = Column(Numeric(10, 2))
    stock_quantity = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String(50), primary_key=True)
    customer_id = Column(
        UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=False
    )
    product_id = Column(
        UUID(as_uuid=True), ForeignKey("products.product_id"), nullable=False
    )
    product_name = Column(String(255))
    quantity = Column(Integer, default=1)
    unit_price = Column(Numeric(10, 2))
    total_amount = Column(Numeric(10, 2))
    status = Column(String(50), default="pending")
    order_date = Column(DateTime, default=datetime.utcnow)
    tracking_number = Column(String(100))
    shipping_address = Column(Text)


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    ticket_id = Column(String(50), primary_key=True)
    customer_id = Column(
        UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=False
    )
    subject = Column(String(500), nullable=False)
    description = Column(Text)
    status = Column(String(50), default="open")
    priority = Column(String(20), default="medium")
    category = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    agent_assigned = Column(String(100))


class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations.
    Provides both asyncpg and SQLAlchemy interfaces.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # AsyncPG connection pool
        self._pool: Optional[Pool] = None

        # SQLAlchemy async engine and session
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Initialize database connections and create tables if needed"""
        try:
            # Initialize AsyncPG pool
            await self._create_asyncpg_pool()

            # Initialize SQLAlchemy
            await self._create_sqlalchemy_engine()

            # Create tables if they don't exist
            await self._create_tables()

            self.logger.info("Database manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {e}")
            raise

    async def _create_asyncpg_pool(self) -> None:
        """Create AsyncPG connection pool"""
        try:
            self._pool = await asyncpg.create_pool(
                host=self.settings.database.host,
                port=self.settings.database.port,
                database=self.settings.database.database,
                user=self.settings.database.username,
                password=self.settings.database.password,
                min_size=1,
                max_size=self.settings.database.pool_size,
                command_timeout=60,
            )
            self.logger.info("AsyncPG pool created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create AsyncPG pool: {e}")
            raise

    async def _create_sqlalchemy_engine(self) -> None:
        """Create SQLAlchemy async engine"""
        try:
            database_url = self.settings.database.url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            self._engine = create_async_engine(
                database_url,
                echo=self.settings.database.echo,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
            )
            self._session_factory = async_sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )
            self.logger.info("SQLAlchemy engine created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create SQLAlchemy engine: {e}")
            raise

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist"""
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created/verified successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            raise

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get AsyncPG database connection from pool"""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")

        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                self.logger.error(f"Database operation failed: {e}")
                raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get SQLAlchemy async session"""
        if not self._session_factory:
            raise RuntimeError("Session factory not initialized")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Session operation failed: {e}")
                raise

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute raw SQL query with AsyncPG"""
        async with self.get_connection() as conn:
            try:
                result = await conn.fetch(query, *args)
                return [dict(record) for record in result]
            except Exception as e:
                self.logger.error(
                    f"Query execution failed: {query[:100]}... Error: {e}"
                )
                raise

    async def execute_command(self, command: str, *args) -> str:
        """Execute SQL command (INSERT, UPDATE, DELETE) with AsyncPG"""
        async with self.get_connection() as conn:
            try:
                result = await conn.execute(command, *args)
                return result
            except Exception as e:
                self.logger.error(
                    f"Command execution failed: {command[:100]}... Error: {e}"
                )
                raise

    # Customer Operations
    async def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by ID"""
        query = "SELECT * FROM customers WHERE customer_id = $1"
        results = await self.execute_query(query, customer_id)
        return results[0] if results else None

    async def get_customer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get customer by email"""
        query = "SELECT * FROM customers WHERE email = $1"
        results = await self.execute_query(query, email)
        return results[0] if results else None

    async def search_customers(
        self, search_term: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search customers by name or email"""
        query = """
        SELECT * FROM customers 
        WHERE first_name ILIKE $1 OR last_name ILIKE $1 OR email ILIKE $1
        LIMIT $2
        """
        return await self.execute_query(query, f"%{search_term}%", limit)

    # Order Operations
    async def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID with customer information"""
        query = """
        SELECT o.*, c.first_name, c.last_name, c.email, c.customer_tier
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_id = $1
        """
        results = await self.execute_query(query, order_id)
        return results[0] if results else None

    async def get_customer_orders(
        self, customer_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get orders for a specific customer"""
        query = """
        SELECT * FROM orders 
        WHERE customer_id = $1 
        ORDER BY order_date DESC 
        LIMIT $2
        """
        return await self.execute_query(query, customer_id, limit)

    async def get_orders_by_status(
        self, status: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get orders by status"""
        query = (
            "SELECT * FROM orders WHERE status = $1 ORDER BY order_date DESC LIMIT $2"
        )
        return await self.execute_query(query, status, limit)

    # Product Operations
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID"""
        query = "SELECT * FROM products WHERE product_id = $1"
        results = await self.execute_query(query, product_id)
        return results[0] if results else None

    async def search_products(
        self, search_term: str, category: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search products by name or category"""
        if category:
            query = """
            SELECT * FROM products 
            WHERE (name ILIKE $1 OR description ILIKE $1) AND category = $2 AND is_active = true
            LIMIT $3
            """
            return await self.execute_query(query, f"%{search_term}%", category, limit)
        else:
            query = """
            SELECT * FROM products 
            WHERE (name ILIKE $1 OR description ILIKE $1) AND is_active = true
            LIMIT $2
            """
            return await self.execute_query(query, f"%{search_term}%", limit)

    async def get_products_by_category(
        self, category: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get products by category"""
        query = (
            "SELECT * FROM products WHERE category = $1 AND is_active = true LIMIT $2"
        )
        return await self.execute_query(query, category, limit)

    # Support Ticket Operations
    async def get_ticket_by_id(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Get support ticket by ID"""
        query = """
        SELECT t.*, c.first_name, c.last_name, c.email, c.customer_tier
        FROM support_tickets t
        JOIN customers c ON t.customer_id = c.customer_id
        WHERE t.ticket_id = $1
        """
        results = await self.execute_query(query, ticket_id)
        return results[0] if results else None

    async def get_customer_tickets(
        self, customer_id: str, status: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get support tickets for a specific customer"""
        if status:
            query = """
            SELECT * FROM support_tickets 
            WHERE customer_id = $1 AND status = $2 
            ORDER BY created_at DESC 
            LIMIT $3
            """
            return await self.execute_query(query, customer_id, status, limit)
        else:
            query = """
            SELECT * FROM support_tickets 
            WHERE customer_id = $1 
            ORDER BY created_at DESC 
            LIMIT $2
            """
            return await self.execute_query(query, customer_id, limit)

    # Data Loading from CSV
    async def load_csv_data(
        self, csv_file_path: str, table_name: str, chunk_size: int = 1000
    ) -> None:
        """Load data from CSV file into database table"""
        try:
            # Read CSV in chunks
            df_chunks = pd.read_csv(csv_file_path, chunksize=chunk_size)

            async with self.get_connection() as conn:
                for chunk in df_chunks:
                    # Convert DataFrame to records
                    records = chunk.to_dict("records")

                    # Generate appropriate INSERT statement based on table
                    if table_name == "customers":
                        await self._insert_customers(conn, records)
                    elif table_name == "products":
                        await self._insert_products(conn, records)
                    elif table_name == "orders":
                        await self._insert_orders(conn, records)
                    elif table_name == "support_tickets":
                        await self._insert_tickets(conn, records)

            self.logger.info(
                f"Successfully loaded data from {csv_file_path} into {table_name}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load CSV data from {csv_file_path}: {e}")
            raise

    async def _insert_customers(self, conn: Connection, records: List[Dict]) -> None:
        """Insert customer records"""
        query = """
        INSERT INTO customers (customer_id, first_name, last_name, email, phone, created_at, subscription_status, customer_tier)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (email) DO UPDATE SET
        first_name = EXCLUDED.first_name,
        last_name = EXCLUDED.last_name,
        phone = EXCLUDED.phone,
        subscription_status = EXCLUDED.subscription_status,
        customer_tier = EXCLUDED.customer_tier
        """

        for record in records:
            await conn.execute(
                query,
                record["customer_id"],
                record["first_name"],
                record["last_name"],
                record["email"],
                record.get("phone"),
                pd.to_datetime(record["created_at"]).to_pydatetime(),
                record["subscription_status"],
                record["customer_tier"],
            )

    async def _insert_products(self, conn: Connection, records: List[Dict]) -> None:
        """Insert product records"""
        query = """
        INSERT INTO products (product_id, name, category, description, price, stock_quantity, is_active, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (product_id) DO UPDATE SET
        name = EXCLUDED.name,
        category = EXCLUDED.category,
        description = EXCLUDED.description,
        price = EXCLUDED.price,
        stock_quantity = EXCLUDED.stock_quantity,
        is_active = EXCLUDED.is_active
        """

        for record in records:
            await conn.execute(
                query,
                record["product_id"],
                record["name"],
                record["category"],
                record["description"],
                record["price"],
                record["stock_quantity"],
                record["is_active"],
                pd.to_datetime(record["created_at"]).to_pydatetime(),
            )

    async def _insert_orders(self, conn: Connection, records: List[Dict]) -> None:
        """Insert order records"""
        query = """
        INSERT INTO orders (order_id, customer_id, product_id, product_name, quantity, unit_price, total_amount, status, order_date, tracking_number, shipping_address)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (order_id) DO UPDATE SET
        status = EXCLUDED.status,
        tracking_number = EXCLUDED.tracking_number
        """

        for record in records:
            await conn.execute(
                query,
                record["order_id"],
                record["customer_id"],
                record["product_id"],
                record["product_name"],
                record["quantity"],
                record["unit_price"],
                record["total_amount"],
                record["status"],
                pd.to_datetime(record["order_date"]).to_pydatetime(),
                record.get("tracking_number"),
                record.get("shipping_address"),
            )

    async def _insert_tickets(self, conn: Connection, records: List[Dict]) -> None:
        """Insert support ticket records"""
        query = """
        INSERT INTO support_tickets (ticket_id, customer_id, subject, description, status, priority, category, created_at, updated_at, agent_assigned)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (ticket_id) DO UPDATE SET
        status = EXCLUDED.status,
        priority = EXCLUDED.priority,
        updated_at = EXCLUDED.updated_at,
        agent_assigned = EXCLUDED.agent_assigned
        """

        for record in records:
            await conn.execute(
                query,
                record["ticket_id"],
                record["customer_id"],
                record["subject"],
                record["description"],
                record["status"],
                record["priority"],
                record["category"],
                pd.to_datetime(record["created_at"]).to_pydatetime(),
                pd.to_datetime(record["updated_at"]).to_pydatetime(),
                record.get("agent_assigned"),
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity"""
        try:
            async with self.get_connection() as conn:
                # Simple query to test connectivity
                result = await conn.fetchval("SELECT 1")

                # Get some basic stats
                customer_count = await conn.fetchval("SELECT COUNT(*) FROM customers")
                order_count = await conn.fetchval("SELECT COUNT(*) FROM orders")
                product_count = await conn.fetchval("SELECT COUNT(*) FROM products")
                ticket_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM support_tickets"
                )

                return {
                    "status": "healthy",
                    "connectivity": "ok" if result == 1 else "failed",
                    "stats": {
                        "customers": customer_count,
                        "orders": order_count,
                        "products": product_count,
                        "tickets": ticket_count,
                    },
                    "pool_size": self._pool.get_size() if self._pool else 0,
                    "pool_available": self._pool.get_idle_size() if self._pool else 0,
                }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "connectivity": "failed"}

    async def close(self) -> None:
        """Close all database connections"""
        try:
            if self._pool:
                await self._pool.close()
                self.logger.info("AsyncPG pool closed")

            if self._engine:
                await self._engine.dispose()
                self.logger.info("SQLAlchemy engine disposed")

        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()

    return _db_manager


async def close_db_manager() -> None:
    """Close global database manager"""
    global _db_manager

    if _db_manager:
        await _db_manager.close()
        _db_manager = None


# Features:

# Dual database interface: AsyncPG (raw queries) + SQLAlchemy (ORM)

# Complete data models for customers, products, orders, support tickets

# Connection pooling with configurable pool sizes

# Async context managers for safe connection handling

# CSV data loading with upsert capabilities

# Health checking with connection statistics

# Key Operations:

# ```python
# async with db_manager.get_connection() as conn:
#     customers = await conn.fetch("SELECT * FROM customers WHERE tier = $1", "platinum")

# async with db_manager.get_session() as session:
#     customer = await session.get(Customer, customer_id)
# ```
