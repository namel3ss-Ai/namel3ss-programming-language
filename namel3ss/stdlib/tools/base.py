"""Tool interface definitions for the Namel3ss standard library."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ToolCategory(Enum):
    """Standard tool categories."""
    
    HTTP = "http"
    """HTTP API tools for external service integration."""
    
    DATABASE = "database" 
    """Database query and operation tools."""
    
    VECTOR_SEARCH = "vector_search"
    """Vector similarity search and retrieval tools."""
    
    PYTHON = "python"
    """Python function execution tools."""
    
    CUSTOM = "custom"
    """Custom or provider-specific tools."""


@dataclass(frozen=True)
class ToolInterface(ABC):
    """Base interface for tool specifications."""
    
    category: ToolCategory
    name: str
    description: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate a tool configuration against this interface."""
        pass


@dataclass(frozen=True) 
class HTTPToolSpec(ToolInterface):
    """Standard interface for HTTP API tools."""
    
    def __post_init__(self):
        object.__setattr__(self, 'category', ToolCategory.HTTP)
        object.__setattr__(self, 'required_fields', [
            'method', 'url', 'description'
        ])
        object.__setattr__(self, 'optional_fields', [
            'headers', 'params', 'body', 'auth', 'timeout', 
            'retry_config', 'response_schema'
        ])
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate HTTP tool configuration."""
        errors = {}
        
        # Check required fields
        for field in self.required_fields:
            if field not in config:
                errors[field] = f"Required field '{field}' is missing"
        
        # Validate method
        method = config.get('method', '').upper()
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if method and method not in valid_methods:
            errors['method'] = f"Invalid HTTP method. Valid methods: {', '.join(valid_methods)}"
        
        # Validate URL format
        url = config.get('url', '')
        if url and not (url.startswith('http://') or url.startswith('https://')):
            errors['url'] = "URL must start with http:// or https://"
        
        # Validate timeout
        timeout = config.get('timeout')
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors['timeout'] = "Timeout must be a positive number"
        
        # Validate headers format
        headers = config.get('headers')
        if headers is not None:
            if not isinstance(headers, dict):
                errors['headers'] = "Headers must be a dictionary"
            else:
                for key, value in headers.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        errors['headers'] = "Header keys and values must be strings"
                        break
        
        return errors


@dataclass(frozen=True)
class DatabaseToolSpec(ToolInterface):
    """Standard interface for database query tools."""
    
    def __post_init__(self):
        object.__setattr__(self, 'category', ToolCategory.DATABASE)
        object.__setattr__(self, 'required_fields', [
            'connection', 'query_type', 'description'
        ])
        object.__setattr__(self, 'optional_fields', [
            'query', 'parameters', 'timeout', 'result_limit',
            'result_schema', 'transaction_mode'
        ])
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate database tool configuration.""" 
        errors = {}
        
        # Check required fields
        for field in self.required_fields:
            if field not in config:
                errors[field] = f"Required field '{field}' is missing"
        
        # Validate query type
        query_type = config.get('query_type', '').lower()
        valid_types = ['select', 'insert', 'update', 'delete', 'exec', 'procedure']
        if query_type and query_type not in valid_types:
            errors['query_type'] = f"Invalid query type. Valid types: {', '.join(valid_types)}"
        
        # Validate connection format
        connection = config.get('connection', '')
        if connection and not isinstance(connection, str):
            errors['connection'] = "Connection must be a string identifier"
        
        # Validate result limit
        result_limit = config.get('result_limit')
        if result_limit is not None:
            if not isinstance(result_limit, int) or result_limit <= 0:
                errors['result_limit'] = "Result limit must be a positive integer"
        
        # Validate timeout
        timeout = config.get('timeout')
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors['timeout'] = "Timeout must be a positive number"
        
        return errors


@dataclass(frozen=True)
class VectorSearchToolSpec(ToolInterface):
    """Standard interface for vector similarity search tools."""
    
    def __post_init__(self):
        object.__setattr__(self, 'category', ToolCategory.VECTOR_SEARCH)
        object.__setattr__(self, 'required_fields', [
            'index_name', 'description'
        ])
        object.__setattr__(self, 'optional_fields', [
            'query_vector', 'query_text', 'top_k', 'filters',
            'similarity_threshold', 'embedding_model', 'metadata_fields'
        ])
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate vector search tool configuration."""
        errors = {}
        
        # Check required fields
        for field in self.required_fields:
            if field not in config:
                errors[field] = f"Required field '{field}' is missing"
        
        # Require either query_vector or query_text
        if 'query_vector' not in config and 'query_text' not in config:
            errors['query_input'] = "Either 'query_vector' or 'query_text' must be specified"
        
        # Validate top_k
        top_k = config.get('top_k')
        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0 or top_k > 1000:
                errors['top_k'] = "top_k must be an integer between 1 and 1000"
        
        # Validate similarity threshold
        threshold = config.get('similarity_threshold')
        if threshold is not None:
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                errors['similarity_threshold'] = "Similarity threshold must be between 0 and 1"
        
        # Validate query vector format
        query_vector = config.get('query_vector')
        if query_vector is not None:
            if not isinstance(query_vector, list):
                errors['query_vector'] = "Query vector must be a list of numbers"
            elif not all(isinstance(x, (int, float)) for x in query_vector):
                errors['query_vector'] = "Query vector must contain only numbers"
        
        # Validate filters format
        filters = config.get('filters')
        if filters is not None and not isinstance(filters, dict):
            errors['filters'] = "Filters must be a dictionary"
        
        return errors


# Standard tool specifications
STANDARD_TOOL_SPECS: Dict[ToolCategory, ToolInterface] = {
    ToolCategory.HTTP: HTTPToolSpec(
        category=ToolCategory.HTTP,
        name="http_tool",
        description="HTTP API tool for external service integration"
    ),
    ToolCategory.DATABASE: DatabaseToolSpec(
        category=ToolCategory.DATABASE, 
        name="database_tool",
        description="Database query and operation tool"
    ),
    ToolCategory.VECTOR_SEARCH: VectorSearchToolSpec(
        category=ToolCategory.VECTOR_SEARCH,
        name="vector_search_tool", 
        description="Vector similarity search and retrieval tool"
    )
}


def get_tool_spec(category: Union[str, ToolCategory]) -> ToolInterface:
    """
    Get the specification for a tool category.
    
    Args:
        category: Tool category name or enum value
        
    Returns:
        Tool interface specification
        
    Raises:
        ValueError: If category is not recognized
    """
    if isinstance(category, str):
        try:
            category = ToolCategory(category)
        except ValueError:
            valid_categories = [c.value for c in ToolCategory]
            raise ValueError(
                f"Unknown tool category '{category}'. "
                f"Valid categories: {', '.join(valid_categories)}"
            )
    
    if category not in STANDARD_TOOL_SPECS:
        raise ValueError(f"No standard specification available for category '{category.value}'")
    
    return STANDARD_TOOL_SPECS[category]


def list_tool_categories() -> List[str]:
    """List all available tool category names."""
    return [category.value for category in ToolCategory]


def get_category_description(category: Union[str, ToolCategory]) -> str:
    """Get human-readable description of a tool category."""
    spec = get_tool_spec(category)
    return spec.description