"""Dataset encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .expressions import _encode_value, _expression_to_runtime, _expression_to_source
from .utils import _coerce_column_name_list

if TYPE_CHECKING:
    from ....ast import (
        AggregateOp,
        ComputedColumnOp,
        Dataset,
        DatasetFeature,
        DatasetOp,
        DatasetProfile,
        DatasetQualityCheck,
        DatasetSchemaField,
        DatasetTarget,
        DatasetTransformStep,
        FilterOp,
        GroupByOp,
        JoinOp,
        OrderByOp,
        WindowFrame,
        WindowOp,
    )


def _encode_dataset(dataset: "Dataset", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a dataset definition for backend state."""
    operations = [_encode_dataset_op(operation, env_keys) for operation in dataset.operations]

    connector = None
    if dataset.connector:
        connector = {
            "type": dataset.connector.connector_type,
            "name": dataset.connector.connector_name,
            "options": _encode_value(dataset.connector.options, env_keys),
        }

    payload: Dict[str, Any] = {
        "name": dataset.name,
        "source_type": dataset.source_type,
        "source": dataset.source,
        "operations": [op for op in operations if op],
        "transforms": [_encode_dataset_transform(step, env_keys) for step in dataset.transforms],
        "schema": [_encode_dataset_schema(field, env_keys) for field in dataset.schema],
        "features": [_encode_dataset_feature(feature, env_keys) for feature in dataset.features],
        "targets": [_encode_dataset_target(target, env_keys) for target in dataset.targets],
        "quality_checks": [_encode_dataset_quality_check(check, env_keys) for check in dataset.quality_checks],
        "profile": _encode_dataset_profile(dataset.profile, env_keys),
        "connector": connector,
        "reactive": dataset.reactive,
        "refresh_policy": _encode_value(dataset.refresh_policy, env_keys),
        "cache_policy": _encode_value(dataset.cache_policy, env_keys),
        "pagination": _encode_value(dataset.pagination, env_keys),
        "streaming": _encode_value(dataset.streaming, env_keys),
        "metadata": _encode_value(dataset.metadata, env_keys),
        "lineage": _encode_value(dataset.lineage, env_keys),
        "tags": list(dataset.tags or []),
        "sample_rows": [{"id": idx + 1, "value": (idx + 1) * 10} for idx in range(3)],
    }
    return payload


def _encode_dataset_op(operation: "DatasetOp", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a dataset operation (filter, group_by, etc.)."""
    from ....ast import (
        AggregateOp,
        ComputedColumnOp,
        FilterOp,
        GroupByOp,
        JoinOp,
        OrderByOp,
        WindowOp,
    )
    
    if isinstance(operation, FilterOp):
        return {
            "type": "filter",
            "condition": _expression_to_source(operation.condition),
            "condition_expr": _expression_to_runtime(operation.condition),
        }
    if isinstance(operation, GroupByOp):
        return {"type": "group_by", "columns": list(operation.columns)}
    if isinstance(operation, AggregateOp):
        return {
            "type": "aggregate",
            "function": operation.function,
            "expression": operation.expression,
        }
    if isinstance(operation, OrderByOp):
        return {"type": "order_by", "columns": list(operation.columns)}
    if isinstance(operation, ComputedColumnOp):
        return {
            "type": "computed_column",
            "name": operation.name,
            "expression": _expression_to_source(operation.expression),
            "expression_expr": _expression_to_runtime(operation.expression),
        }
    if isinstance(operation, WindowOp):
        return {
            "type": "window",
            "name": operation.name,
            "function": operation.function,
            "target": operation.target,
            "partition_by": list(operation.partition_by or []),
            "order_by": list(operation.order_by or []),
            "frame": _encode_window_frame(operation.frame),
        }
    if isinstance(operation, JoinOp):
        return {
            "type": "join",
            "target_type": operation.target_type,
            "target_name": operation.target_name,
            "join_type": operation.join_type,
            "condition": _expression_to_source(operation.condition),
            "condition_expr": _expression_to_runtime(operation.condition),
        }
    return {"type": type(operation).__name__}


def _encode_window_frame(frame: "WindowFrame") -> Dict[str, Any]:
    """Encode a window frame specification."""
    return {
        "mode": frame.mode,
        "interval_value": frame.interval_value,
        "interval_unit": frame.interval_unit,
    }


def _encode_dataset_transform(step: "DatasetTransformStep", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a dataset transformation step."""
    return {
        "name": step.name,
        "type": step.transform_type,
        "inputs": list(step.inputs or []),
        "output": step.output,
        "expression": _expression_to_source(step.expression),
        "expression_expr": _expression_to_runtime(step.expression),
        "options": _encode_value(step.options, env_keys),
    }


def _encode_dataset_schema(field: "DatasetSchemaField", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a dataset schema field definition."""
    return {
        "name": field.name,
        "dtype": field.dtype,
        "nullable": field.nullable,
        "description": field.description,
        "tags": list(field.tags or []),
        "constraints": _encode_value(field.constraints, env_keys),
        "stats": _encode_value(field.stats, env_keys),
    }


def _encode_dataset_feature(feature: "DatasetFeature", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a dataset feature definition."""
    return {
        "name": feature.name,
        "role": feature.role,
        "source": feature.source,
        "dtype": feature.dtype,
        "expression": _expression_to_source(feature.expression),
        "expression_expr": _expression_to_runtime(feature.expression),
        "description": feature.description,
        "options": _encode_value(feature.options, env_keys),
    }


def _encode_dataset_target(target: "DatasetTarget", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a dataset target (label) definition."""
    return {
        "name": target.name,
        "kind": target.kind,
        "expression": _expression_to_source(target.expression),
        "expression_expr": _expression_to_runtime(target.expression),
        "positive_class": target.positive_class,
        "horizon": target.horizon,
        "options": _encode_value(target.options, env_keys),
    }


def _encode_dataset_quality_check(check: "DatasetQualityCheck", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a dataset quality check definition."""
    return {
        "name": check.name,
        "condition": _expression_to_source(check.condition),
        "condition_expr": _expression_to_runtime(check.condition),
        "metric": check.metric,
        "threshold": check.threshold,
        "severity": check.severity,
        "message": check.message,
        "extras": _encode_value(check.extras, env_keys),
    }


def _encode_dataset_profile(profile: Optional["DatasetProfile"], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
    """Encode dataset profiling information."""
    if profile is None:
        return None
    return {
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "freshness": profile.freshness,
        "updated_at": profile.updated_at,
        "stats": _encode_value(profile.stats, env_keys),
    }
