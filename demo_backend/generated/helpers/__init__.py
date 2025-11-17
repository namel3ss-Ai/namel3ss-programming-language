"""Helper utilities for Namel3ss generated routers."""

from __future__ import annotations

from typing import Iterable

from fastapi import FastAPI

from ..routers import GENERATED_ROUTERS

__all__ = ["GENERATED_ROUTERS", "include_generated_routers"]


def include_generated_routers(app: FastAPI, routers: Iterable = GENERATED_ROUTERS) -> None:
    """Attach generated routers to ``app`` while allowing custom overrides."""

    for router in routers:
        app.include_router(router)
