"""Slug utilities for generating deterministic filenames and identifiers."""

from __future__ import annotations

import re


def slugify_route(route: str) -> str:
    """Generate a safe filename from a page route."""
    if route == "/":
        return "index"
    slug = route.strip("/")
    slug = slug.replace("/", "_")
    return slug or "index"


def slugify_page_name(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"page_{index}"


def slugify_identifier(value: str, default: str = "insight") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or default
