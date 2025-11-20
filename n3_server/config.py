from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # API
    api_title: str = "N3 Graph API"
    api_version: str = "0.1.0"
    api_prefix: str = "/api"
    
    # Database
    database_url: str = "postgresql+asyncpg://n3:n3password@localhost:5432/n3_graphs"
    
    # OpenTelemetry
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "n3-server"
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]
    
    # JWT for share tokens
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    
    # Auth - secret key for JWT tokens
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_USE_STRONG_RANDOM_KEY"
    
    # RLHF training
    rlhf_model_path: str = "./models"
    rlhf_batch_size: int = 8
    rlhf_learning_rate: float = 1e-5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
