"""CRUD resource encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ....ast import CrudResource


def _encode_crud_resource(resource: "CrudResource", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a CRUD resource definition for backend state."""
    select_fields = list(dict.fromkeys(resource.select_fields or []))
    mutable_fields = list(dict.fromkeys(resource.mutable_fields or []))
    primary_key = resource.primary_key or "id"
    if not mutable_fields:
        mutable_fields = [field for field in select_fields if field and field != primary_key]
    allowed: List[str] = []
    seen: Set[str] = set()
    for operation in resource.allowed_operations:
        candidate = str(operation or "").lower()
        if candidate in {"list", "retrieve", "create", "update", "delete"} and candidate not in seen:
            seen.add(candidate)
            allowed.append(candidate)
    if str(resource.source_type or "table").lower() != "table":
        allowed = [op for op in allowed if op in {"list", "retrieve"}]
    read_only = resource.read_only or not any(op in {"create", "update", "delete"} for op in allowed)
    default_limit = int(resource.default_limit or 100)
    max_limit = int(resource.max_limit or max(default_limit, 100))
    if default_limit <= 0:
        default_limit = 100
    if max_limit < default_limit:
        max_limit = default_limit
    label = resource.label or resource.name.replace("-", " ").title()
    return {
        "slug": resource.name,
        "label": label,
        "source_type": str(resource.source_type or "table").lower(),
        "source_name": resource.source_name,
        "primary_key": primary_key,
        "select_fields": select_fields,
        "mutable_fields": mutable_fields,
        "allowed_operations": allowed,
        "tenant_column": resource.tenant_column,
        "default_limit": default_limit,
        "max_limit": max_limit,
        "read_only": read_only,
    }
