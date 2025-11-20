"""Utility functions for backend state encoding."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ....ast import Prompt
    from .classes import PageComponent


def _slugify_route(route: str) -> str:
    """Convert a route path to a slug identifier."""
    slug = route.strip("/") or "root"
    slug = slug.replace("/", "_")
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", slug)
    return slug.lower() or "page"


def _slugify_page_name(name: str) -> str:
    """Convert a page name to a slug identifier."""
    slug = name.strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug.lower() or "page"


def _slugify_identifier(value: str) -> str:
    """Convert any string to a slug identifier."""
    slug = value.strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug.lower() or "item"


def _page_api_path(route: str) -> str:
    """Generate API path for a page route."""
    cleaned = route.strip("/")
    if not cleaned:
        cleaned = "root"
    return f"/api/pages/{cleaned}"


def _coerce_column_name_list(value: Any) -> List[str]:
    """Coerce various formats into a list of column names."""
    if not value:
        return []
    if isinstance(value, str):
        parts = [segment.strip() for segment in value.split(",") if segment.strip()]
        return parts or [value.strip()]
    if isinstance(value, dict):
        if "name" in value:
            return [str(value["name"])]
        value = value.get("columns") or value.get("names")
    if isinstance(value, (list, tuple, set)):
        names: List[str] = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
        return names
    return []


def _normalize_split_mapping(raw: Dict[str, Any]) -> Dict[str, float]:
    """Normalize a split mapping to name->ratio dictionary."""
    splits: Dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number <= 0:
            continue
        splits[name] = number
    return splits


def _validate_chain_memory_options(
    options: Dict[str, Any],
    memory_names: Set[str],
    chain_name: str,
    step_kind: str,
    target: str,
) -> None:
    """Validate that chain step memory references are defined."""
    if not options:
        return
    for key in ("read_memory", "write_memory"):
        if key not in options:
            continue
        names = options[key]
        if isinstance(names, str):
            collection = [names]
        elif isinstance(names, list):
            collection = names
        else:
            continue
        missing = [name for name in collection if name not in memory_names]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Chain '{chain_name}' step '{step_kind}:{target}' references undefined memory: {missing_str}"
            )


def _validate_prompt_arguments(
    prompt_lookup: Dict[str, "Prompt"],
    prompt_name: str,
    arguments: Dict[str, Any],
) -> None:
    """Validate prompt call arguments against prompt definition."""
    prompt = prompt_lookup.get(prompt_name)
    if prompt is None:
        raise ValueError(f"Prompt '{prompt_name}' is not defined")
    valid_names = {field.name for field in prompt.input_fields}
    required_names = {field.name for field in prompt.input_fields if field.required}
    provided = set(arguments.keys())
    missing = sorted(required_names - provided)
    if missing:
        raise ValueError(
            f"Prompt '{prompt_name}' is missing required inputs: {', '.join(missing)}"
        )
    extra = sorted(provided - valid_names)
    if extra:
        raise ValueError(
            f"Prompt '{prompt_name}' does not accept inputs: {', '.join(extra)}"
        )


def _component_to_serializable(component: "PageComponent") -> Dict[str, Any]:
    """Convert a PageComponent to a serializable dictionary."""
    from .classes import PageComponent
    
    data = {"type": component.type}
    if component.index is not None:
        data["__component_index"] = component.index
    data.update(component.payload)
    return data
