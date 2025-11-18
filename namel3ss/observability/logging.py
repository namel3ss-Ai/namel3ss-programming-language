"""Centralised logging helpers for Namel3ss runtimes."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

_LOGGER_CACHE: Dict[str, logging.Logger] = {}


def get_logger(name: str = "namel3ss") -> logging.Logger:
    """Return a cached :class:`logging.Logger` instance."""

    if name not in _LOGGER_CACHE:
        _LOGGER_CACHE[name] = logging.getLogger(name)
    return _LOGGER_CACHE[name]


def log_retry_event(
    *,
    provider: str,
    model: Optional[str],
    attempt: int,
    delay: float,
    reason: Optional[str],
    logger: Optional[logging.Logger] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured retry log entry for connector calls."""

    payload: Dict[str, Any] = {
        "provider": provider or "unknown",
        "model": model or "unknown",
        "attempt": attempt,
        "next_delay": round(delay, 3),
    }
    if reason:
        payload["reason"] = reason
    if extras:
        payload.update(extras)
    target_logger = logger or get_logger("namel3ss.connectors.retry")
    target_logger.warning(
        "Retrying connector call",
        extra={"namel3ss_event": "connector_retry", "namel3ss_data": payload},
    )
