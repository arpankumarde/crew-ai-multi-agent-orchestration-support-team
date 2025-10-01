# Configuration Management
import os
import yaml
from typing import Dict, Any, Optional, List

# from pydantic import BaseSettings, Field, validator
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import logging


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration"""

    host: str = Field(default="localhost", env="POSTGRES_HOST")
    port: int = Field(default=5432, env="POSTGRES_PORT")
    database: str = Field(default="customer_service", env="POSTGRES_DB")
    username: str = Field(default="postgres", env="POSTGRES_USER")
    password: str = Field(default="password", env="POSTGRES_PASSWORD")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo: bool = Field(default=False)

    @property
    def url(self) -> str:
        """Generate database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class VectorStoreSettings(BaseSettings):
    """ChromaDB vector store configuration"""

    # path: str = Field(default="./storage/chroma_db", env="CHROMA_DB_PATH")
    path: str = Field(default="./storage/chroma_db", alias="CHROMA_DB_PATH")
    collection_name: str = Field(default="knowledge_base")
    distance_function: str = Field(default="cosine")
    embedding_model: str = Field(default="text-embedding-3-small")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",  # 👈 prevents auto-prepending class name
        extra="ignore",  # 👈 ignore unrelated env vars
    )

    # @validator("path")
    @field_validator("path")
    def create_path_if_not_exists(cls, v):
        """Create ChromaDB path if it doesn't exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class AIModelSettings(BaseSettings):
    """AI model configuration"""

    default_llm: str = Field(default="openai")
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=2000)
    timeout: int = Field(default=60)

    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)

    # @validator("openai_api_key")
    @field_validator("openai_api_key")
    def validate_api_key(cls, v):
        if not v or v == "your_openai_api_key_here":
            raise ValueError("OpenAI API key must be provided")
        return v


class AppSettings(BaseSettings):
    """Application configuration"""

    name: str = Field(default="CrewAI Customer Service")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=True, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_concurrent_queries: int = Field(default=10)
    request_timeout: int = Field(default=300)


class KnowledgeBaseSettings(BaseSettings):
    """Knowledge base configuration"""

    data_path: str = Field(default="./data/knowledge_base")
    supported_formats: List[str] = Field(default=["md", "txt", "pdf"])
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_chunks_per_query: int = Field(default=5)


class EscalationSettings(BaseSettings):
    """Escalation rules configuration"""

    low_confidence_threshold: float = Field(default=0.6)
    complex_query_keywords: List[str] = Field(
        default=["legal", "lawsuit", "fraud", "data breach", "emergency"]
    )
    vip_customer_tiers: List[str] = Field(default=["platinum", "enterprise"])

    default_queue: str = Field(default="general_support")
    technical_queue: str = Field(default="technical_support")
    billing_queue: str = Field(default="billing_support")
    escalation_queue: str = Field(default="tier2_support")


class QualityAssuranceSettings(BaseSettings):
    """Quality assurance configuration"""

    enabled: bool = Field(default=True)
    minimum_quality_score: float = Field(default=7.0)
    auto_approve_threshold: float = Field(default=8.5)

    grammar_check: bool = Field(default=True)
    sentiment_analysis: bool = Field(default=True)
    compliance_validation: bool = Field(default=True)
    factual_accuracy: bool = Field(default=True)

    tone: str = Field(default="professional_friendly")
    max_response_length: int = Field(default=1000)
    required_elements: List[str] = Field(default=["greeting", "solution", "next_steps"])


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""

    enabled: bool = Field(default=True)
    metrics_endpoint: str = Field(default="/metrics")
    health_check_endpoint: str = Field(default="/health")

    track_response_times: bool = Field(default=True)
    track_success_rates: bool = Field(default=True)
    track_escalation_rates: bool = Field(default=True)
    track_customer_satisfaction: bool = Field(default=False)

    high_error_rate_threshold: float = Field(default=0.1)
    slow_response_threshold: float = Field(default=15.0)


class LoggingSettings(BaseSettings):
    """Logging configuration"""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json")
    file_path: Optional[str] = Field(default="./logs/app.log", env="LOG_FILE_PATH")
    max_file_size: str = Field(default="100MB")
    backup_count: int = Field(default=5)

    include_request_id: bool = Field(default=True)
    include_customer_id: bool = Field(default=True)
    include_session_id: bool = Field(default=True)
    mask_sensitive_fields: bool = Field(default=True)


class Settings(BaseSettings):
    """Main application settings"""

    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    ai_models: AIModelSettings = AIModelSettings()
    knowledge_base: KnowledgeBaseSettings = KnowledgeBaseSettings()
    escalation: EscalationSettings = EscalationSettings()
    quality_assurance: QualityAssuranceSettings = QualityAssuranceSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    logging: LoggingSettings = LoggingSettings()

    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"
    #     case_sensitive = False

    model_config = SettingsConfigDict(  # CHANGED
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # optional, avoids "extra inputs not permitted"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str = "config/settings.yaml") -> "Settings":
        """Load settings from YAML file with environment variable overrides"""
        try:
            with open(yaml_path, "r") as file:
                yaml_data = yaml.safe_load(file)

            # Convert nested YAML to flat settings
            flat_settings = cls._flatten_yaml(yaml_data)

            # Create settings instance with YAML data as defaults
            # Environment variables will still override these values
            return cls(**flat_settings)

        except FileNotFoundError:
            logging.warning(
                f"Settings file {yaml_path} not found, using defaults and environment variables"
            )
            return cls()
        except Exception as e:
            logging.error(f"Error loading settings from {yaml_path}: {e}")
            return cls()

    @staticmethod
    def _flatten_yaml(
        data: Dict[str, Any], parent_key: str = "", sep: str = "__"
    ) -> Dict[str, Any]:
        """Flatten nested YAML structure for pydantic"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Settings._flatten_yaml(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def validate_settings(self) -> bool:
        """Validate all settings and dependencies"""
        try:
            # Validate database connection parameters
            if not all(
                [
                    self.database.host,
                    self.database.database,
                    self.database.username,
                    self.database.password,
                ]
            ):
                raise ValueError("Database configuration incomplete")

            # Validate vector store path
            if not self.vector_store.path:
                raise ValueError("Vector store path not configured")

            # Validate OpenAI API key
            if (
                not self.ai_models.openai_api_key
                or self.ai_models.openai_api_key == "your_openai_api_key_here"
            ):
                raise ValueError("OpenAI API key not configured")

            # Validate knowledge base path
            if not os.path.exists(self.knowledge_base.data_path):
                logging.warning(
                    f"Knowledge base path {self.knowledge_base.data_path} does not exist"
                )

            # Validate quality thresholds
            if not (0 <= self.quality_assurance.minimum_quality_score <= 10):
                raise ValueError("Quality score must be between 0 and 10")

            if not (0 <= self.escalation.low_confidence_threshold <= 1):
                raise ValueError("Confidence threshold must be between 0 and 1")

            return True

        except Exception as e:
            logging.error(f"Settings validation failed: {e}")
            return False

    def get_env_vars(self) -> Dict[str, str]:
        """Get environment variables that should be set"""
        return {
            "OPENAI_API_KEY": self.ai_models.openai_api_key,
            "POSTGRES_HOST": self.database.host,
            "POSTGRES_PORT": str(self.database.port),
            "POSTGRES_DB": self.database.database,
            "POSTGRES_USER": self.database.username,
            "POSTGRES_PASSWORD": self.database.password,
            "CHROMA_DB_PATH": self.vector_store.path,
            "LOG_LEVEL": self.logging.level,
            "DEBUG": str(self.app.debug).lower(),
            "ENVIRONMENT": self.app.environment,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "app": self.app.model_dump(),  # CHANGED
            "database": {
                **self.database.model_dump(),  # CHANGED
                "password": "***",
            },
            "vector_store": self.vector_store.model_dump(),  # CHANGED
            "ai_models": {
                **self.ai_models.model_dump(),  # CHANGED
                "openai_api_key": "***",
            },
            "knowledge_base": self.knowledge_base.model_dump(),  # CHANGED
            "escalation": self.escalation.model_dump(),  # CHANGED
            "quality_assurance": self.quality_assurance.model_dump(),  # CHANGED
            "monitoring": self.monitoring.model_dump(),  # CHANGED
            "logging": self.logging.model_dump(),  # CHANGED
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)"""
    global _settings

    if _settings is None:
        _settings = Settings.from_yaml()

        # Validate settings on first load
        if not _settings.validate_settings():
            raise RuntimeError("Settings validation failed")

        logging.info("Settings loaded and validated successfully")

    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()


# Create settings instance for module-level access
settings = get_settings()

# Features:

# Pydantic-based settings with environment variable support

# YAML configuration loading with env var overrides

# Nested configuration classes for different system components:

# DatabaseSettings: PostgreSQL connection parameters

# VectorStoreSettings: ChromaDB configuration with auto-path creation

# AIModelSettings: OpenAI API and embedding model setup

# AppSettings: Application-level configuration

# EscalationSettings: Agent escalation rules and routing

# QualityAssuranceSettings: Response quality thresholds

# MonitoringSettings: Performance metrics configuration

# LoggingSettings: Structured logging setup

# Key Capabilities:

# Settings validation with detailed error reporting

# Environment variable masking for security

# YAML file loading with fallback to defaults

# Singleton pattern for global settings access
