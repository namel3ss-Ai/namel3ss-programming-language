"""Test-only helper models for experiment integration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def dataset_model(payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return deterministic predictions based on the dataset payload."""

    dataset = None
    if payload and isinstance(payload, dict):
        dataset = payload.get("dataset") or payload.get("input", {}).get("dataset")
    if dataset is None:
        dataset = kwargs.get("dataset")
    rows: List[Dict[str, Any]] = list(dataset.get("rows", [])) if isinstance(dataset, dict) else []
    predictions: List[Any] = []
    for row in rows:
        tenure = row.get("tenure")
        try:
            numeric_tenure = float(tenure)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            numeric_tenure = 0.0
        predictions.append(numeric_tenure >= 10)
    return {"predictions": predictions, "status": "ok", "count": len(predictions)}
