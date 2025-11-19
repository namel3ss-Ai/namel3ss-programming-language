"""Extensible metric emission helpers for Namel3ss."""

from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, Optional

MetricListener = Callable[[str, Dict[str, float], Dict[str, str]], None]

_LISTENERS: list[MetricListener] = []
_LOCK = RLock()


def register_metric_listener(callback: MetricListener) -> None:
    """Register a listener for custom metric events."""

    with _LOCK:
        if callback not in _LISTENERS:
            _LISTENERS.append(callback)


def emit_metric(name: str, values: Optional[Dict[str, float]] = None, labels: Optional[Dict[str, str]] = None) -> None:
    """Emit a metric payload to registered listeners."""

    payload = dict(values or {})
    tags = {str(key): str(value) for key, value in (labels or {}).items()}
    with _LOCK:
        listeners = list(_LISTENERS)
    for callback in listeners:
        try:
            callback(name, payload, tags)
        except Exception:
            # Metrics should never break the main execution path.
            continue


def get_metric(name: str) -> Callable[[float, Optional[Dict[str, str]]], None]:
    """Return a helper that emits a single-value metric with labels."""

    def _emitter(value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        emit_metric(name, values={"value": value}, labels=labels)

    return _emitter


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a single metric value with optional tags.
    
    This is a convenience function that wraps emit_metric for simple use cases.
    
    Args:
        name: Metric name
        value: Numeric value to record
        tags: Optional dictionary of string tags/labels
    """
    emit_metric(name, values={"value": value}, labels=tags)
