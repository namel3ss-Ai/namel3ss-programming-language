"""
Runtime capability detection for optional features.

This module provides functions to check if optional dependencies are available
and raise helpful errors when they're missing. This enables a lightweight core
installation while gracefully handling missing features.

Usage:
    from namel3ss.features import require_openai, has_redis
    
    # Check if feature is available
    if has_redis():
        # Use Redis functionality
        pass
    
    # Require feature or raise helpful error
    require_openai()  # Raises ImportError with install instructions
    from openai import OpenAI  # Safe to import now
"""

from typing import Optional


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is required but not installed."""
    pass


# ============================================================================
# AI/LLM Providers
# ============================================================================

def has_openai() -> bool:
    """Check if OpenAI SDK is available."""
    try:
        import openai
        return True
    except ImportError:
        return False


def require_openai() -> None:
    """
    Ensure OpenAI SDK is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If openai package is not installed
    """
    if not has_openai():
        raise MissingDependencyError(
            "OpenAI integration requires the 'openai' extra.\n"
            "Install with: pip install 'namel3ss[openai]'\n"
            "Or for all AI providers: pip install 'namel3ss[ai]'"
        )


def has_anthropic() -> bool:
    """Check if Anthropic SDK is available."""
    try:
        import anthropic
        return True
    except ImportError:
        return False


def require_anthropic() -> None:
    """
    Ensure Anthropic SDK is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If anthropic package is not installed
    """
    if not has_anthropic():
        raise MissingDependencyError(
            "Anthropic integration requires the 'anthropic' extra.\n"
            "Install with: pip install 'namel3ss[anthropic]'\n"
            "Or for all AI providers: pip install 'namel3ss[ai]'"
        )


def has_tiktoken() -> bool:
    """Check if tiktoken tokenizer is available."""
    try:
        import tiktoken
        return True
    except ImportError:
        return False


def require_tiktoken() -> None:
    """
    Ensure tiktoken is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If tiktoken package is not installed
    """
    if not has_tiktoken():
        raise MissingDependencyError(
            "Token counting requires the 'tiktoken' library.\n"
            "Install with: pip install 'namel3ss[ai]'"
        )


# ============================================================================
# Database Support
# ============================================================================

def has_sqlalchemy() -> bool:
    """Check if SQLAlchemy is available."""
    try:
        import sqlalchemy
        return True
    except ImportError:
        return False


def require_sqlalchemy() -> None:
    """
    Ensure SQLAlchemy is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If sqlalchemy package is not installed
    """
    if not has_sqlalchemy():
        raise MissingDependencyError(
            "SQL database support requires the 'sql' extra.\n"
            "Install with: pip install 'namel3ss[sql]'"
        )


def has_asyncpg() -> bool:
    """Check if asyncpg PostgreSQL driver is available."""
    try:
        import asyncpg
        return True
    except ImportError:
        return False


def require_asyncpg() -> None:
    """
    Ensure asyncpg is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If asyncpg package is not installed
    """
    if not has_asyncpg():
        raise MissingDependencyError(
            "PostgreSQL support requires the 'postgres' extra.\n"
            "Install with: pip install 'namel3ss[postgres]'\n"
            "Or for all SQL databases: pip install 'namel3ss[sql]'"
        )


def has_psycopg() -> bool:
    """Check if psycopg3 PostgreSQL driver is available."""
    try:
        import psycopg
        return True
    except ImportError:
        return False


def require_psycopg() -> None:
    """
    Ensure psycopg3 is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If psycopg package is not installed
    """
    if not has_psycopg():
        raise MissingDependencyError(
            "PostgreSQL support requires the 'postgres' extra.\n"
            "Install with: pip install 'namel3ss[postgres]'\n"
            "Or for all SQL databases: pip install 'namel3ss[sql]'"
        )


def has_aiomysql() -> bool:
    """Check if aiomysql MySQL driver is available."""
    try:
        import aiomysql
        return True
    except ImportError:
        return False


def require_aiomysql() -> None:
    """
    Ensure aiomysql is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If aiomysql package is not installed
    """
    if not has_aiomysql():
        raise MissingDependencyError(
            "MySQL support requires the 'mysql' extra.\n"
            "Install with: pip install 'namel3ss[mysql]'\n"
            "Or for all SQL databases: pip install 'namel3ss[sql]'"
        )


def has_mongo() -> bool:
    """Check if MongoDB drivers are available."""
    try:
        import motor
        import pymongo
        return True
    except ImportError:
        return False


def require_mongo() -> None:
    """
    Ensure MongoDB drivers are available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If motor/pymongo packages are not installed
    """
    if not has_mongo():
        raise MissingDependencyError(
            "MongoDB support requires the 'mongo' extra.\n"
            "Install with: pip install 'namel3ss[mongo]'"
        )


# ============================================================================
# Caching & Message Queues
# ============================================================================

def has_redis() -> bool:
    """Check if Redis client is available."""
    try:
        import redis
        return True
    except ImportError:
        return False


def require_redis() -> None:
    """
    Ensure Redis client is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If redis package is not installed
    """
    if not has_redis():
        raise MissingDependencyError(
            "Redis support requires the 'redis' extra.\n"
            "Install with: pip install 'namel3ss[redis]'"
        )


# ============================================================================
# Real-time Communication
# ============================================================================

def has_websockets() -> bool:
    """Check if websockets library is available."""
    try:
        import websockets
        return True
    except ImportError:
        return False


def require_websockets() -> None:
    """
    Ensure websockets library is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If websockets package is not installed
    """
    if not has_websockets():
        raise MissingDependencyError(
            "WebSocket support requires the 'websockets' extra.\n"
            "Install with: pip install 'namel3ss[websockets]'\n"
            "Or for full real-time features: pip install 'namel3ss[realtime]'"
        )


# ============================================================================
# Observability & Monitoring
# ============================================================================

def has_opentelemetry() -> bool:
    """Check if OpenTelemetry SDK is available."""
    try:
        import opentelemetry
        return True
    except ImportError:
        return False


def require_opentelemetry() -> None:
    """
    Ensure OpenTelemetry SDK is available, or raise helpful error.
    
    Raises:
        MissingDependencyError: If opentelemetry packages are not installed
    """
    if not has_opentelemetry():
        raise MissingDependencyError(
            "OpenTelemetry instrumentation requires the 'otel' extra.\n"
            "Install with: pip install 'namel3ss[otel]'"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def get_available_features() -> dict[str, bool]:
    """
    Get a dictionary of all optional features and their availability.
    
    Returns:
        Dictionary mapping feature names to availability status
    
    Example:
        >>> features = get_available_features()
        >>> if features['openai']:
        ...     print("OpenAI is available")
    """
    return {
        'openai': has_openai(),
        'anthropic': has_anthropic(),
        'tiktoken': has_tiktoken(),
        'sqlalchemy': has_sqlalchemy(),
        'asyncpg': has_asyncpg(),
        'psycopg': has_psycopg(),
        'aiomysql': has_aiomysql(),
        'mongo': has_mongo(),
        'redis': has_redis(),
        'websockets': has_websockets(),
        'opentelemetry': has_opentelemetry(),
    }


def print_feature_status() -> None:
    """Print a formatted table of all optional features and their status."""
    features = get_available_features()
    
    print("\n🔍 Namel3ss Optional Features:")
    print("─" * 40)
    
    categories = {
        'AI/LLM Providers': ['openai', 'anthropic', 'tiktoken'],
        'Databases': ['sqlalchemy', 'asyncpg', 'psycopg', 'aiomysql', 'mongo'],
        'Caching': ['redis'],
        'Real-time': ['websockets'],
        'Observability': ['opentelemetry'],
    }
    
    for category, feature_list in categories.items():
        print(f"\n{category}:")
        for feature in feature_list:
            if feature in features:
                status = "✓" if features[feature] else "✗"
                print(f"  {status} {feature}")
    
    print()


__all__ = [
    # AI/LLM
    'has_openai',
    'require_openai',
    'has_anthropic',
    'require_anthropic',
    'has_tiktoken',
    'require_tiktoken',
    # Databases
    'has_sqlalchemy',
    'require_sqlalchemy',
    'has_asyncpg',
    'require_asyncpg',
    'has_psycopg',
    'require_psycopg',
    'has_aiomysql',
    'require_aiomysql',
    'has_mongo',
    'require_mongo',
    # Caching
    'has_redis',
    'require_redis',
    # Real-time
    'has_websockets',
    'require_websockets',
    # Observability
    'has_opentelemetry',
    'require_opentelemetry',
    # Utilities
    'get_available_features',
    'print_feature_status',
    'MissingDependencyError',
]
