"""Frame (table) encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .expressions import _encode_value, _expression_to_runtime, _expression_to_source

if TYPE_CHECKING:
    from ....ast import (
        Frame,
        FrameAccessPolicy,
        FrameColumn,
        FrameColumnConstraint,
        FrameConstraint,
        FrameIndex,
        FrameRelationship,
        FrameSourceDef,
    )


def _encode_frame(frame: "Frame", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a frame (table) definition for backend state."""
    metadata_value = _encode_value(frame.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    options_value = _encode_value(frame.options, env_keys)
    if not isinstance(options_value, dict):
        options_value = {"value": options_value} if options_value is not None else {}
    examples: List[Dict[str, Any]] = []
    for example in frame.examples:
        encoded = _encode_value(example, env_keys)
        if isinstance(encoded, dict):
            examples.append(encoded)
        else:
            examples.append({"value": encoded})
    return {
        "name": frame.name,
        "source_type": str(frame.source_type or "dataset").lower(),
        "source": frame.source,
        "description": frame.description,
        "columns": [_encode_frame_column(column, env_keys) for column in frame.columns],
        "indexes": [_encode_frame_index(index, env_keys) for index in frame.indexes],
        "relationships": [_encode_frame_relationship(relationship, env_keys) for relationship in frame.relationships],
        "constraints": [_encode_frame_constraint(constraint, env_keys) for constraint in frame.constraints],
        "access": _encode_frame_access(frame.access, env_keys),
        "tags": list(frame.tags or []),
        "metadata": metadata_value,
        "examples": examples,
        "options": options_value,
        "key": list(frame.key or []),
        "splits": dict(frame.splits or {}),
        "source_config": _encode_frame_source(frame.source_config),
    }


def _encode_frame_column(column: "FrameColumn", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a frame column definition."""
    metadata_value = _encode_value(column.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    return {
        "name": column.name,
        "dtype": column.dtype,
        "nullable": column.nullable,
        "description": column.description,
        "role": column.role,
        "default": _encode_value(column.default, env_keys),
        "expression": _expression_to_source(column.expression),
        "expression_expr": _expression_to_runtime(column.expression),
        "source": column.source,
        "tags": list(column.tags or []),
        "metadata": metadata_value,
        "validations": [_encode_frame_column_validation(validation, env_keys) for validation in column.validations],
    }


def _encode_frame_column_validation(validation: "FrameColumnConstraint", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a frame column validation constraint."""
    config_value = _encode_value(validation.config, env_keys)
    if not isinstance(config_value, dict):
        config_value = {"value": config_value} if config_value is not None else {}
    return {
        "name": validation.name,
        "expression": _expression_to_source(validation.expression),
        "expression_expr": _expression_to_runtime(validation.expression),
        "message": validation.message,
        "severity": validation.severity,
        "config": config_value,
    }


def _encode_frame_index(index: "FrameIndex", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a frame index definition."""
    options_value = _encode_value(index.options, env_keys)
    if not isinstance(options_value, dict):
        options_value = {"value": options_value} if options_value is not None else {}
    return {
        "name": index.name,
        "columns": list(index.columns or []),
        "unique": index.unique,
        "method": index.method,
        "options": options_value,
    }


def _encode_frame_relationship(relationship: "FrameRelationship", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a frame relationship (foreign key) definition."""
    metadata_value = _encode_value(relationship.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    return {
        "name": relationship.name,
        "target_frame": relationship.target_frame,
        "target_dataset": relationship.target_dataset,
        "local_key": relationship.local_key,
        "remote_key": relationship.remote_key,
        "cardinality": relationship.cardinality,
        "join_type": relationship.join_type,
        "description": relationship.description,
        "metadata": metadata_value,
    }


def _encode_frame_constraint(constraint: "FrameConstraint", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a frame-level constraint."""
    metadata_value = _encode_value(constraint.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    return {
        "name": constraint.name,
        "expression": _expression_to_source(constraint.expression),
        "expression_expr": _expression_to_runtime(constraint.expression),
        "message": constraint.message,
        "severity": constraint.severity,
        "metadata": metadata_value,
    }


def _encode_frame_access(policy: Optional["FrameAccessPolicy"], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
    """Encode frame access policy."""
    if policy is None:
        return None
    metadata_value = _encode_value(policy.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    return {
        "public": policy.public,
        "roles": list(policy.roles or []),
        "allow_anonymous": policy.allow_anonymous,
        "rate_limit_per_minute": policy.rate_limit_per_minute,
        "cache_seconds": policy.cache_seconds,
        "metadata": metadata_value,
    }


def _encode_frame_source(source: Optional["FrameSourceDef"]) -> Optional[Dict[str, Any]]:
    """Encode frame source configuration."""
    if source is None:
        return None
    payload: Dict[str, Any] = {"kind": source.kind}
    if source.connection:
        payload["connection"] = source.connection
    if source.table:
        payload["table"] = source.table
    if source.path:
        payload["path"] = source.path
    if source.format:
        payload["format"] = source.format
    return payload
