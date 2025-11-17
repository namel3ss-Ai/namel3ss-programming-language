"""Utilities for tracking and validating optional dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Dict, Iterable, List, Optional

CORE_DEPENDENCIES: Dict[str, str] = {
    "fastapi": "FastAPI runtime",
    "httpx": "HTTP client support",
    "pydantic": "Pydantic models",
    "uvicorn": "Uvicorn development server",
}

OPTIONAL_DEPENDENCIES: Dict[str, Dict[str, Iterable[str]]] = {
    "cli": {
        "title": "CLI helpers",
        "modules": ("dotenv",),
    },
    "sql": {
        "title": "SQL connectors",
        "modules": ("sqlalchemy",),
    },
    "redis": {
        "title": "Redis caching & pub/sub",
        "modules": ("redis.asyncio",),
    },
    "mongo": {
        "title": "MongoDB connectors",
        "modules": ("motor", "pymongo"),
    },
}


@dataclass(frozen=True)
class DependencyReport:
    """Summary of a dependency group and the modules it requires."""

    key: str
    title: str
    modules: List[str]
    missing: List[str]
    optional: bool
    advice: Optional[str] = None


def module_available(module: str) -> bool:
    """Return ``True`` when *module* can be imported."""

    if not module:
        return False
    try:
        return find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):  # pragma: no cover - defensive
        return False


def require_dependency(module: str, group: str) -> None:
    """Raise an informative error when *module* from *group* is missing."""

    if module_available(module):
        return
    raise ImportError(
        f"{module} is required for this feature. Try: pip install 'namel3ss[{group}]'"
    )


def iter_dependency_reports() -> List[DependencyReport]:
    """Return the current availability for core and optional dependencies."""

    reports: List[DependencyReport] = []

    for module, title in CORE_DEPENDENCIES.items():
        missing = [] if module_available(module) else [module]
        advice = None
        if missing:
            advice = f"pip install {module}"
        reports.append(
            DependencyReport(
                key=module,
                title=title,
                modules=[module],
                missing=missing,
                optional=False,
                advice=advice,
            )
        )

    for extra, payload in OPTIONAL_DEPENDENCIES.items():
        modules = list(payload["modules"])
        missing = [module for module in modules if not module_available(module)]
        advice = None
        if missing:
            advice = f"pip install 'namel3ss[{extra}]'"
        reports.append(
            DependencyReport(
                key=extra,
                title=payload["title"],
                modules=modules,
                missing=missing,
                optional=True,
                advice=advice,
            )
        )

    return reports


__all__ = [
    "CORE_DEPENDENCIES",
    "OPTIONAL_DEPENDENCIES",
    "DependencyReport",
    "iter_dependency_reports",
    "module_available",
    "require_dependency",
]
