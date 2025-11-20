"""
Connector definitions for external service integrations.

Connectors represent configuration handles for external services
(databases, APIs, message queues, etc.) that the N3 runtime can interact with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Connector:
    """
    Declarative connector definition for external service integration.
    
    Represents a configured connection to an external service such as:
    - Databases (PostgreSQL, MySQL, MongoDB)
    - Message queues (RabbitMQ, Kafka)
    - APIs (REST, GraphQL)
    - Cloud services (S3, Azure Blob)
    
    Example DSL:
        connector main_db {
            type: postgres
            provider: aws_rds
            config: {
                host: "db.example.com",
                database: "production",
                port: 5432
            }
        }
    """
    name: str
    connector_type: str
    provider: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    @property
    def category(self) -> str:
        """Alias for connector_type for backward compatibility."""
        return self.connector_type


__all__ = ["Connector"]
