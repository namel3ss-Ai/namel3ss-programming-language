"""Repository package for {{ project_name }}."""

from .interface import {{ entity_name }}Repository
from .postgres import Postgres{{ entity_name }}Repository

__all__ = [
    "{{ entity_name }}Repository",
    "Postgres{{ entity_name }}Repository",
]
