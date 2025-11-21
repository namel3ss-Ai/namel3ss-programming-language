"""
Namel3ss Python SDK - Incremental Adoption & Integration Layer

This SDK enables Python developers to integrate Namel3ss into existing projects
without migrating their entire stack. It provides both remote and in-process
execution modes for maximum flexibility.

Usage Patterns:
    Remote execution (N3 as microservice):
        >>> from namel3ss_sdk import N3Client
        >>> client = N3Client(base_url="https://api.example.com")
        >>> result = client.chains.run("summarize", text="...")
        
    In-process execution (embedded N3 runtime):
        >>> from namel3ss_sdk import N3InProcessRuntime
        >>> runtime = N3InProcessRuntime("./app.n3")
        >>> result = runtime.chains.run("summarize", text="...")
    
    Async support:
        >>> async with N3Client(base_url="...") as client:
        ...     result = await client.chains.arun("summarize", text="...")

Key Features:
    - Zero-config defaults with type-safe configuration
    - Automatic retries with exponential backoff
    - Circuit breaker for fault tolerance
    - OpenTelemetry instrumentation
    - Comprehensive exception hierarchy
    - Full async support
    - Request ID tracking

Security:
    - TLS required for remote calls
    - Token rotation support
    - No PII/secrets in logs
    - Configurable timeout/retry policies
"""

from .client import N3Client
from .runtime import N3InProcessRuntime
from .config import N3Settings, N3ClientConfig, N3RuntimeConfig
from .exceptions import (
    N3Error,
    N3ClientError,
    N3TimeoutError,
    N3AuthError,
    N3RuntimeError,
    N3SchemaError,
    N3ConnectionError,
    N3RateLimitError,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "N3Client",
    "N3InProcessRuntime",
    
    # Configuration
    "N3Settings",
    "N3ClientConfig",
    "N3RuntimeConfig",
    
    # Exceptions
    "N3Error",
    "N3ClientError",
    "N3TimeoutError",
    "N3AuthError",
    "N3RuntimeError",
    "N3SchemaError",
    "N3ConnectionError",
    "N3RateLimitError",
    
    # Version
    "__version__",
]
