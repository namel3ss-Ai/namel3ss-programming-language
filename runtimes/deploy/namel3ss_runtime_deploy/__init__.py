"""
Namel3ss Deploy Runtime

Provides Docker, Kubernetes, and cloud deployment adapters for Namel3ss applications.
"""

from .adapter import (
    generate_docker,
    generate_kubernetes,
    generate_aws_config,
    generate_gcp_config,
    generate_azure_config,
)

__all__ = [
    "generate_docker",
    "generate_kubernetes",
    "generate_aws_config",
    "generate_gcp_config",
    "generate_azure_config",
]

__version__ = "0.1.0"
