"""Configuration management for N3 SDK using Pydantic Settings.

Supports multiple sources (environment variables, .env files, explicit config)
with type validation and defaults.

Environment variables:
    N3_BASE_URL: N3 server URL
    N3_API_TOKEN: Authentication token
    N3_TIMEOUT: Request timeout in seconds
    N3_MAX_RETRIES: Maximum retry attempts
    N3_VERIFY_SSL: Enable TLS verification
    N3_LOG_LEVEL: Logging level
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class N3ClientConfig(BaseSettings):
    """Configuration for N3 remote client.
    
    Can be initialized from environment variables, .env files, or kwargs.
    
    Example:
        From environment:
        >>> import os
        >>> os.environ['N3_BASE_URL'] = 'https://api.example.com'
        >>> config = N3ClientConfig()
        
        From kwargs:
        >>> config = N3ClientConfig(
        ...     base_url='https://api.example.com',
        ...     api_token='secret'
        ... )
        
        From .env file:
        >>> config = N3ClientConfig(_env_file='.env')
    """
    
    model_config = SettingsConfigDict(
        env_prefix='N3_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )
    
    # Connection settings
    base_url: str = Field(
        default="http://localhost:8000",
        description="N3 server base URL"
    )
    
    api_token: Optional[str] = Field(
        default=None,
        description="Authentication token"
    )
    
    # Timeout settings
    timeout: float = Field(
        default=30.0,
        ge=0.1,
        description="Request timeout in seconds"
    )
    
    connect_timeout: float = Field(
        default=5.0,
        ge=0.1,
        description="Connection timeout in seconds"
    )
    
    # Retry settings
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )
    
    retry_backoff_factor: float = Field(
        default=1.0,
        ge=0.0,
        description="Backoff multiplier for retries"
    )
    
    retry_backoff_max: float = Field(
        default=60.0,
        ge=1.0,
        description="Maximum backoff time in seconds"
    )
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failures before opening circuit"
    )
    
    circuit_breaker_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="Seconds before testing recovery"
    )
    
    # TLS settings
    verify_ssl: bool = Field(
        default=True,
        description="Verify TLS certificates"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # OpenTelemetry
    enable_tracing: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing"
    )
    
    service_name: str = Field(
        default="n3-sdk-client",
        description="Service name for tracing"
    )


class N3RuntimeConfig(BaseSettings):
    """Configuration for N3 in-process runtime.
    
    Example:
        >>> config = N3RuntimeConfig(
        ...     source_file='./app.n3',
        ...     allow_stubs=False
        ... )
    """
    
    model_config = SettingsConfigDict(
        env_prefix='N3_RUNTIME_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )
    
    # Source settings
    source_file: Optional[Path] = Field(
        default=None,
        description="Path to .n3 source file"
    )
    
    # Runtime settings
    allow_stubs: bool = Field(
        default=False,
        description="Allow stub implementations"
    )
    
    # Cache settings
    enable_cache: bool = Field(
        default=True,
        description="Enable response caching"
    )
    
    cache_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum cache entries"
    )
    
    cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Cache TTL in seconds"
    )
    
    # Execution settings
    max_turns: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum agent turns"
    )
    
    timeout: float = Field(
        default=300.0,
        ge=1.0,
        description="Execution timeout in seconds"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # OpenTelemetry
    enable_tracing: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing"
    )
    
    service_name: str = Field(
        default="n3-sdk-runtime",
        description="Service name for tracing"
    )


class N3Settings(BaseSettings):
    """Global N3 SDK settings combining client and runtime config.
    
    Example:
        >>> settings = N3Settings(
        ...     client=N3ClientConfig(base_url='https://api.example.com'),
        ...     runtime=N3RuntimeConfig(source_file='./app.n3')
        ... )
    """
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )
    
    client: N3ClientConfig = Field(default_factory=N3ClientConfig)
    runtime: N3RuntimeConfig = Field(default_factory=N3RuntimeConfig)


# Singleton instance
_settings: Optional[N3Settings] = None


def get_settings() -> N3Settings:
    """Get global settings singleton."""
    global _settings
    if _settings is None:
        _settings = N3Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
