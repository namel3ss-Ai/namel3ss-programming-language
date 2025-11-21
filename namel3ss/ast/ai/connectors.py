"""
Connector definitions for external service integrations.

Connectors represent configuration handles for external services
(databases, APIs, message queues, etc.) that the N3 runtime can interact with.

This module provides production-grade connector definitions with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Connector:
    """
    Declarative connector definition for external service integration.
    
    Represents a configured connection to an external service such as:
    - Databases (PostgreSQL, MySQL, MongoDB, Redis)
    - Message queues (RabbitMQ, Kafka, AWS SQS)
    - APIs (REST, GraphQL, gRPC)
    - Cloud services (AWS S3, Azure Blob, Google Cloud Storage)
    - Vector stores (Pinecone, Weaviate, Qdrant)
    
    Connectors are declared in N3 DSL and resolved at runtime to establish
    connections to external systems. Configuration is typically loaded from
    environment variables or secure configuration stores.
    
    Attributes:
        name: Unique identifier for this connector within the N3 program
        connector_type: Type/category of connector (e.g., "postgres", "s3", "kafka")
        provider: Optional cloud provider or service (e.g., "aws", "gcp", "azure")
        config: Connection configuration (host, port, credentials, options)
        description: Human-readable description of this connector's purpose
        
    Example DSL:
        connector main_db {
            type: postgres
            provider: aws_rds
            config: {
                host: "db.example.com",
                database: "production",
                port: 5432,
                ssl_mode: "require"
            }
            description: "Primary production database"
        }
        
        connector doc_store {
            type: s3
            provider: aws
            config: {
                bucket: "company-documents",
                region: "us-east-1"
            }
        }
        
        connector cache {
            type: redis
            config: {
                host: "cache.internal",
                port: 6379,
                db: 0
            }
        }
    
    Validation:
        Use validate_connector() from .validation to ensure configuration
        is valid before attempting to establish connections.
        
    Notes:
        - Sensitive credentials should be referenced via environment variables
          or secure secret management systems, not hardcoded in config
        - The runtime will attempt to establish connections lazily when first used
        - Connection pooling and retry logic is handled by the runtime, not the AST
    """
    name: str
    connector_type: str
    provider: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    @property
    def category(self) -> str:
        """
        Alias for connector_type for backward compatibility.
        
        Returns:
            The connector type/category
            
        Deprecated:
            Use connector_type directly instead of category
        """
        return self.connector_type


__all__ = ["Connector"]
