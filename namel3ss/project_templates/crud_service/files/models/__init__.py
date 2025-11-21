"""Models package for {{ project_name }}."""

from .domain import {{ entity_name }}
from .schemas import (
    {{ entity_name }}Create,
    {{ entity_name }}Update,
    {{ entity_name }}Response,
    {{ entity_name }}List,
    ErrorResponse,
    ErrorDetail,
)

__all__ = [
    "{{ entity_name }}",
    "{{ entity_name }}Create",
    "{{ entity_name }}Update",
    "{{ entity_name }}Response",
    "{{ entity_name }}List",
    "ErrorResponse",
    "ErrorDetail",
]
