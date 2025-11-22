"""
Configuration management for CRUD service.

Loads settings from environment variables with validation.
No hard-coded credentials or connection strings.
"""

from typing import Literal, Optional
from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment.
    
    All sensitive values (database URLs, API keys) must come from environment
    variables, never hard-coded.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = Field(default="{{ project_name }}", description="Application name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, description="Number of worker processes")
    
    # Database
    database_url: PostgresDsn = Field(
        ...,
        description="PostgreSQL connection URL (required)"
    )
    db_pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    db_max_overflow: int = Field(default=20, ge=0, description="Max overflow connections")
    db_pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    db_echo: bool = Field(default=False, description="Echo SQL queries (dev only)")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format"
    )
    
    # Pagination
    default_page_size: int = Field(default=20, ge=1, le=100, description="Default items per page")
    max_page_size: int = Field(default=100, ge=1, le=1000, description="Maximum items per page")
    
    # Security
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    
    # Multi-tenancy (extension point)
    enable_multi_tenancy: bool = Field(default=False, description="Enable multi-tenant mode")
    tenant_header: str = Field(default="X-Tenant-ID", description="Tenant identifier header")
    
    # Authentication - JWT Configuration
    jwt_secret_key: str = Field(
        ...,
        description="Secret key for JWT signing (required, use strong random key)"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm (HS256, RS256, etc.)"
    )
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        description="Access token expiration time in minutes"
    )
    jwt_issuer: Optional[str] = Field(
        default=None,
        description="JWT issuer claim for validation (optional)"
    )
    jwt_audience: Optional[str] = Field(
        default=None,
        description="JWT audience claim for validation (optional)"
    )
    auth_disabled: bool = Field(
        default=False,
        description="Disable authentication (ONLY for local dev, never in production)"
    )
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: PostgresDsn) -> PostgresDsn:
        """Ensure database URL uses postgresql scheme."""
        if v.scheme not in ("postgresql", "postgresql+asyncpg", "postgresql+psycopg2"):
            raise ValueError("Database URL must use postgresql:// scheme")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    def validate_auth_config(self) -> None:
        """Validate authentication configuration."""
        if self.is_production and self.auth_disabled:
            raise ValueError(
                "Authentication cannot be disabled in production. "
                "Set AUTH_DISABLED=false or remove the setting."
            )
        
        if not self.auth_disabled and len(self.jwt_secret_key) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters. "
                "Use a cryptographically secure random key."
            )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings singleton.
    
    Settings are loaded once and cached. Use this function to access
    configuration throughout the application.
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
