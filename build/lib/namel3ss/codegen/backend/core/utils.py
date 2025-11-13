"""Shared helpers for rendering Python literals."""

from __future__ import annotations

import pprint
from typing import Any

__all__ = ["_assign_literal", "_format_literal"]


def _assign_literal(name: str, annotation: str, value: Any) -> str:
    literal = _format_literal(value)
    return f"{name}: {annotation} = {literal}"


def _format_literal(value: Any) -> str:
    return pprint.pformat(value, width=100, sort_dicts=False)
