# Utility Functions and Helpers
import os
import sys
import logging
import uuid
import time
import functools
import re
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib

from .config import get_settings


def setup_logging() -> None:
    """Setup application logging configuration"""
    try:
        settings = get_settings()

        # Create logs directory if it doesn't exist
        if settings.logging.file_path:
            log_path = Path(settings.logging.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging format
        if settings.logging.format == "json":
            # JSON formatter for structured logging
            import json

            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_entry = {
                        "timestamp": datetime.fromtimestamp(
                            record.created, timezone.utc
                        ).isoformat(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                    }

                    # Add exception info if present
                    if record.exc_info:
                        log_entry["exception"] = self.formatException(record.exc_info)

                    # Add extra fields from record
                    for key, value in record.__dict__.items():
                        if key not in [
                            "name",
                            "msg",
                            "args",
                            "levelname",
                            "levelno",
                            "pathname",
                            "filename",
                            "module",
                            "exc_info",
                            "exc_text",
                            "stack_info",
                            "lineno",
                            "funcName",
                            "created",
                            "msecs",
                            "relativeCreated",
                            "thread",
                            "threadName",
                            "processName",
                            "process",
                            "message",
                        ]:
                            log_entry[key] = value

                    return json.dumps(log_entry)

            formatter = JSONFormatter()
        else:
            # Standard formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.logging.level.upper()))

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler if configured
        if settings.logging.file_path:
            from logging.handlers import RotatingFileHandler

            # Parse max file size
            max_bytes = _parse_file_size(settings.logging.max_file_size)

            file_handler = RotatingFileHandler(
                settings.logging.file_path,
                maxBytes=max_bytes,
                backupCount=settings.logging.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        logging.info("Logging configured successfully")

    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.error(f"Failed to setup advanced logging: {e}")


def _parse_file_size(size_str: str) -> int:
    """Parse file size string to bytes"""
    size_str = size_str.upper()

    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def validate_environment() -> Dict[str, Any]:
    """
    Validate that all required environment variables and dependencies are available.

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "environment_vars": {},
        "dependencies": {},
    }

    try:
        settings = get_settings()

        # Check required environment variables
        required_env_vars = {
            "OPENAI_API_KEY": settings.ai_models.openai_api_key,
            "POSTGRES_HOST": settings.database.host,
            "POSTGRES_DB": settings.database.database,
            "POSTGRES_USER": settings.database.username,
            "POSTGRES_PASSWORD": settings.database.password,
        }

        for var_name, var_value in required_env_vars.items():
            if not var_value or var_value in ["your_openai_api_key_here", "password"]:
                validation_results["errors"].append(f"Missing or invalid {var_name}")
                validation_results["valid"] = False
            else:
                validation_results["environment_vars"][var_name] = "✓"

        # Check optional environment variables
        optional_env_vars = {
            "CHROMA_DB_PATH": settings.vector_store.path,
            "LOG_LEVEL": settings.logging.level,
            "DEBUG": str(settings.app.debug),
        }

        for var_name, var_value in optional_env_vars.items():
            validation_results["environment_vars"][var_name] = (
                "✓" if var_value else "default"
            )

        # Check Python dependencies
        dependencies_to_check = [
            ("crewai", "CrewAI framework"),
            ("chromadb", "ChromaDB vector database"),
            ("asyncpg", "PostgreSQL async driver"),
            ("openai", "OpenAI API client"),
            ("pandas", "Data manipulation"),
            ("pydantic", "Data validation"),
        ]

        for dep_name, dep_description in dependencies_to_check:
            try:
                __import__(dep_name)
                validation_results["dependencies"][dep_name] = "✓"
            except ImportError:
                validation_results["dependencies"][dep_name] = "✗"
                validation_results["errors"].append(
                    f"Missing dependency: {dep_name} ({dep_description})"
                )
                validation_results["valid"] = False

        # Check directory permissions
        required_dirs = [settings.vector_store.path, settings.knowledge_base.data_path]

        if settings.logging.file_path:
            required_dirs.append(str(Path(settings.logging.file_path).parent))

        for dir_path in required_dirs:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                # Test write permission
                test_file = Path(dir_path) / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                validation_results["warnings"].append(
                    f"Directory access issue: {dir_path} - {e}"
                )

        return validation_results

    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Environment validation failed: {e}")
        return validation_results


def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"sess_{int(time.time())}_{uuid.uuid4().hex[:12]}"


def sanitize_input(input_text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input for security and processing.

    Args:
        input_text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not isinstance(input_text, str):
        input_text = str(input_text)

    # Remove potentially dangerous characters
    input_text = re.sub(r'[<>"\']', "", input_text)

    # Normalize whitespace
    input_text = re.sub(r"\s+", " ", input_text.strip())

    # Truncate if too long
    if len(input_text) > max_length:
        input_text = input_text[:max_length] + "..."

    return input_text


def mask_sensitive_data(
    data: Dict[str, Any], sensitive_keys: List[str] = None
) -> Dict[str, Any]:
    """
    Mask sensitive data in dictionaries for logging.

    Args:
        data: Dictionary containing data
        sensitive_keys: List of keys to mask

    Returns:
        Dictionary with sensitive data masked
    """
    if sensitive_keys is None:
        sensitive_keys = [
            "password",
            "api_key",
            "token",
            "secret",
            "key",
            "email",
            "phone",
            "ssn",
            "credit_card",
        ]

    masked_data = data.copy()

    for key, value in masked_data.items():
        key_lower = key.lower()

        # Check if key contains sensitive information
        if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
            if isinstance(value, str) and len(value) > 4:
                masked_data[key] = value[:2] + "***" + value[-2:]
            else:
                masked_data[key] = "***"
        elif isinstance(value, dict):
            masked_data[key] = mask_sensitive_data(value, sensitive_keys)

    return masked_data


def format_response(
    data: Any, success: bool = True, message: str = None
) -> Dict[str, Any]:
    """
    Format API response in standard structure.

    Args:
        data: Response data
        success: Success status
        message: Optional message

    Returns:
        Formatted response dictionary
    """
    response = {
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    if message:
        response["message"] = message

    return response


def format_error_response(error: Exception, request_id: str = None) -> Dict[str, Any]:
    """
    Format error response for consistent error handling.

    Args:
        error: Exception that occurred
        request_id: Optional request ID for tracking

    Returns:
        Formatted error response
    """
    response = {
        "success": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": {"type": type(error).__name__, "message": str(error)},
    }

    if request_id:
        response["request_id"] = request_id

    return response


def measure_execution_time(func: Callable = None, *, log_result: bool = True):
    """
    Decorator to measure and optionally log function execution time.

    Args:
        func: Function to decorate
        log_result: Whether to log the execution time

    Returns:
        Decorated function or decorator
    """

    def decorator(function):
        @functools.wraps(function)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await function(*args, **kwargs)
                execution_time = time.time() - start_time

                if log_result:
                    logger = logging.getLogger(function.__module__)
                    logger.info(
                        f"{function.__name__} executed in {execution_time:.3f}s"
                    )

                return result
            except Exception as e:
                execution_time = time.time() - start_time
                if log_result:
                    logger = logging.getLogger(function.__module__)
                    logger.error(
                        f"{function.__name__} failed after {execution_time:.3f}s: {e}"
                    )
                raise

        @functools.wraps(function)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = function(*args, **kwargs)
                execution_time = time.time() - start_time

                if log_result:
                    logger = logging.getLogger(function.__module__)
                    logger.info(
                        f"{function.__name__} executed in {execution_time:.3f}s"
                    )

                return result
            except Exception as e:
                execution_time = time.time() - start_time
                if log_result:
                    logger = logging.getLogger(function.__module__)
                    logger.error(
                        f"{function.__name__} failed after {execution_time:.3f}s: {e}"
                    )
                raise

        # Return appropriate wrapper based on function type
        if hasattr(function, "__code__") and function.__code__.co_flags & 0x80:
            return async_wrapper  # Async function
        else:
            return sync_wrapper  # Sync function

    # Support both @measure_execution_time and @measure_execution_time()
    if func is None:
        return decorator
    else:
        return decorator(func)


def create_file_hash(file_path: Union[str, Path]) -> str:
    """
    Create MD5 hash of file contents for change detection.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Failed to create hash for {file_path}: {e}")
        return ""


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback.

    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logging.warning(f"Failed to parse JSON: {e}")
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely dump object to JSON string with fallback.

    Args:
        obj: Object to serialize
        default: Default string if serialization fails

    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logging.warning(f"Failed to serialize to JSON: {e}")
        return default


def extract_email_addresses(text: str) -> List[str]:
    """Extract email addresses from text"""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.findall(email_pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text"""
    phone_patterns = [
        r"\b\d{3}-\d{3}-\d{4}\b",
        r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",
        r"\b\d{10}\b",
        r"\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    ]

    phone_numbers = []
    for pattern in phone_patterns:
        phone_numbers.extend(re.findall(pattern, text))

    return list(set(phone_numbers))  # Remove duplicates


def extract_order_numbers(text: str) -> List[str]:
    """Extract order numbers from text"""
    order_patterns = [
        r"(?:order|ord|order#|ord#)\s*[:\-]?\s*([a-zA-Z0-9\-]{6,15})",
        r"\b(ORD-\d{6})\b",
        r"\b([A-Z]{2,3}\d{6,10})\b",
    ]

    order_numbers = []
    for pattern in order_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        order_numbers.extend(matches)

    return list(set(order_numbers))  # Remove duplicates


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    return text


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime object to string"""
    return dt.strftime(format_str)


def is_valid_email(email: str) -> bool:
    """Validate email address format"""
    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$"
    return bool(re.match(email_pattern, email))


def is_valid_phone(phone: str) -> bool:
    """Validate phone number format"""
    phone_pattern = r"^[\+]?[1-9]?[\-\.\s]?\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4}$"
    return bool(re.match(phone_pattern, phone.replace(" ", "")))


class Timer:
    """Simple timer context manager"""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logging.info(f"{self.description} completed in {duration:.3f}s")

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0

        end_time = self.end_time or time.time()
        return end_time - self.start_time


# Features:

# Advanced logging setup with JSON formatting option

# Environment validation with dependency checking

# Request/session ID generation for tracking

# Input sanitization and security functions

# Data masking for sensitive information logging

# Response formatting with standard API structure

# Performance monitoring decorators

# Text processing utilities (email/phone/order extraction)

# Timer context manager for operation timing
