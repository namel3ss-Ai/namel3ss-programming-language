"""Tool Adapter Framework for Namel3ss.

Provides first-class adapters for extending N3 to interact with external systems.
"""

from .base import (
    AdapterConfig,
    AdapterError,
    AdapterExecutionError,
    AdapterResult,
    AdapterTimeoutError,
    AdapterType,
    AdapterValidationError,
    BaseAdapter,
    RetryPolicy,
    get_adapter,
    list_adapters,
    register_adapter,
)
from .database import (
    DatabaseAdapter,
    DatabaseAdapterConfig,
    DatabaseEngine,
    QueryType,
)
from .http import HttpAdapter, HttpAdapterConfig, HttpMethod
from .model import ModelAdapter, ModelAdapterConfig, ModelProvider
from .python import PythonAdapter, PythonAdapterConfig
from .queue import (
    CeleryQueueAdapter,
    RQQueueAdapter,
    QueueAdapterConfig,
    QueueBackend,
    create_queue_adapter,
)

__all__ = [
    # Base types
    "AdapterType",
    "RetryPolicy",
    "AdapterConfig",
    "AdapterResult",
    "BaseAdapter",
    
    # Errors
    "AdapterError",
    "AdapterValidationError",
    "AdapterExecutionError",
    "AdapterTimeoutError",
    
    # Registry
    "register_adapter",
    "get_adapter",
    "list_adapters",
    
    # Python adapter
    "PythonAdapter",
    "PythonAdapterConfig",
    
    # HTTP adapter
    "HttpAdapter",
    "HttpAdapterConfig",
    "HttpMethod",
    
    # Database adapter
    "DatabaseAdapter",
    "DatabaseAdapterConfig",
    "DatabaseEngine",
    "QueryType",
    
    # Queue adapters
    "CeleryQueueAdapter",
    "RQQueueAdapter",
    "QueueAdapterConfig",
    "QueueBackend",
    "create_queue_adapter",
    
    # Model adapter
    "ModelAdapter",
    "ModelAdapterConfig",
    "ModelProvider",
]
