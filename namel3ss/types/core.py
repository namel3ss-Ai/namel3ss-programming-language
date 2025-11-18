"""Core type representations for the Namel3ss language.

The type model is intentionally lightweight for Phase 3: it allows later
phases to perform static checks without tying the implementation to any
specific parser or runtime.  The module exposes a set of reusable
classes for scalar, collection, dataset, and AI-related types as well as
utility helpers for common operations (stringification, compatibility
checks, and basic schema derivations).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from .frame import FrameColumnType, N3FrameType


class ScalarKind(str, Enum):
    """Enumeration of the scalar types surfaced at the language level."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "str"
    DATETIME = "datetime"
    ANY = "any"


@dataclass(frozen=True)
class ScalarType:
    """Represents a scalar value with optional nullability metadata."""

    kind: ScalarKind
    nullable: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        suffix = "?" if self.nullable else ""
        return f"{self.kind.value}{suffix}"


@dataclass(frozen=True)
class AnyType:
    """Top type used when no better static information is available."""

    nullable: bool = True

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        return "any?" if self.nullable else "any"


ANY_TYPE = AnyType()


@dataclass(frozen=True)
class ListType:
    """Represents a homogeneous list."""

    element_type: "N3Type"
    nullable: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        suffix = "?" if self.nullable else ""
        return f"list[{stringify_type(self.element_type)}]{suffix}"


@dataclass(frozen=True)
class MapType:
    """Represents a mapping/dictionary type."""

    key_type: "N3Type"
    value_type: "N3Type"
    nullable: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        suffix = "?" if self.nullable else ""
        return f"map[{stringify_type(self.key_type)}, {stringify_type(self.value_type)}]{suffix}"


@dataclass(frozen=True)
class FrameTypeRef:
    """Wrapper around :class:`N3FrameType` with an optional label for context."""

    schema: N3FrameType
    label: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        return self.label or "frame"


@dataclass(frozen=True)
class DatasetType:
    """Datasets surface frame-like schemas but may be backed by connectors."""

    frame: FrameTypeRef
    source: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        return self.source or self.frame.__str__()


@dataclass(frozen=True)
class ExpressionType:
    """Represents the static type of an evaluated expression."""

    value_type: "N3Type"
    is_predicate: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        prefix = "predicate " if self.is_predicate else ""
        return f"{prefix}{stringify_type(self.value_type)}"


@dataclass(frozen=True)
class PromptIOTypes:
    """Describes the input/output field schemas for prompt definitions."""

    inputs: Mapping[str, "N3Type"]
    outputs: Mapping[str, "N3Type"]


# Public alias for convenience when annotating helpers.
N3Type = ScalarType | ListType | MapType | FrameTypeRef | DatasetType | AnyType

_NUMERIC_PROMOTIONS: Dict[ScalarKind, Sequence[ScalarKind]] = {
    ScalarKind.INT: (ScalarKind.FLOAT,),
    ScalarKind.BOOL: (ScalarKind.INT, ScalarKind.FLOAT),
}


def stringify_type(type_info: N3Type) -> str:
    """Return a human readable representation of ``type_info``."""

    return str(type_info)


def _allow_nullability(source_nullable: bool, target_nullable: bool) -> bool:
    if source_nullable and not target_nullable:
        return False
    return True


def is_assignable(source: N3Type, target: N3Type) -> bool:
    """Return ``True`` when ``source`` values can be assigned to ``target`` slots."""

    if isinstance(target, AnyType):
        return True
    if isinstance(source, AnyType):
        return True
    if isinstance(source, ScalarType) and isinstance(target, ScalarType):
        if source.kind == target.kind:
            return _allow_nullability(source.nullable, target.nullable)
        promoted_targets = _NUMERIC_PROMOTIONS.get(source.kind, ())
        if target.kind in promoted_targets:
            return _allow_nullability(source.nullable, target.nullable)
        return False
    if isinstance(source, ListType) and isinstance(target, ListType):
        if not _allow_nullability(source.nullable, target.nullable):
            return False
        return is_assignable(source.element_type, target.element_type)
    if isinstance(source, MapType) and isinstance(target, MapType):
        if not _allow_nullability(source.nullable, target.nullable):
            return False
        keys_assignable = is_assignable(source.key_type, target.key_type)
        values_assignable = is_assignable(source.value_type, target.value_type)
        return keys_assignable and values_assignable
    if isinstance(source, FrameTypeRef) and isinstance(target, FrameTypeRef):
        return source.schema.describe() == target.schema.describe()
    if isinstance(source, DatasetType) and isinstance(target, DatasetType):
        return is_assignable(source.frame, target.frame)
    return False


def is_compatible(left: N3Type, right: N3Type) -> bool:
    """Return ``True`` when ``left`` and ``right`` can be compared or merged."""

    return is_assignable(left, right) or is_assignable(right, left)


_SCALAR_ALIASES: Dict[str, ScalarKind] = {
    "int": ScalarKind.INT,
    "integer": ScalarKind.INT,
    "float": ScalarKind.FLOAT,
    "double": ScalarKind.FLOAT,
    "bool": ScalarKind.BOOL,
    "boolean": ScalarKind.BOOL,
    "string": ScalarKind.STRING,
    "str": ScalarKind.STRING,
    "datetime": ScalarKind.DATETIME,
    "timestamp": ScalarKind.DATETIME,
}


def infer_scalar_type(dtype: str, *, nullable: bool = True) -> ScalarType:
    """Best-effort conversion from schema dtype strings to ``ScalarType``."""

    kind = _SCALAR_ALIASES.get(dtype.lower(), ScalarKind.STRING)
    return ScalarType(kind=kind, nullable=nullable)


def lookup_column_type(frame: FrameTypeRef, column_name: str) -> Optional[ScalarType]:
    """Return the scalar type associated with ``column_name`` if known."""

    column = frame.schema.get_column(column_name)
    if column is None:
        return None
    return infer_scalar_type(column.dtype, nullable=column.nullable)


def derive_select_schema(frame: FrameTypeRef, columns: Sequence[str]) -> FrameTypeRef:
    """Return a new ``FrameTypeRef`` projected to ``columns``."""

    return FrameTypeRef(schema=frame.schema.subset(columns), label=frame.label)


def derive_filter_schema(frame: FrameTypeRef) -> FrameTypeRef:
    """Filters never change schemas, so return the original frame reference."""

    return FrameTypeRef(schema=frame.schema, label=frame.label)


def derive_group_schema(
    frame: FrameTypeRef,
    group_columns: Sequence[str],
    aggregations: Mapping[str, ScalarType],
) -> FrameTypeRef:
    """Return a grouped schema consisting of the group columns plus aggregates."""

    projection = frame.schema.subset(group_columns)
    projected_columns: MutableMapping[str, FrameColumnType] = dict(projection.columns)
    projected_order = list(projection.order)
    for agg_name, agg_type in aggregations.items():
        projected_columns[agg_name] = FrameColumnType(
            name=agg_name,
            dtype=agg_type.kind.value,
            nullable=True,
            role=None,
        )
        projected_order.append(agg_name)
    grouped_schema = N3FrameType(
        columns=projected_columns,
        order=projected_order,
        key=list(group_columns),
        splits=dict(frame.schema.splits),
    )
    return FrameTypeRef(schema=grouped_schema, label=frame.label)


def derive_join_schema(
    left: FrameTypeRef,
    right: FrameTypeRef,
    join_on: Sequence[str],
) -> FrameTypeRef:
    """Return the schema that results from joining ``left`` with ``right``."""

    merged_schema = left.schema.merge(
        other=right.schema,
        join_on=join_on,
        frame_name=left.label or "left",
        right_name=right.label or "right",
    )
    return FrameTypeRef(schema=merged_schema, label=left.label)


__all__ = [
    "ScalarKind",
    "ScalarType",
    "AnyType",
    "ANY_TYPE",
    "ListType",
    "MapType",
    "FrameTypeRef",
    "DatasetType",
    "ExpressionType",
    "PromptIOTypes",
    "N3Type",
    "stringify_type",
    "is_assignable",
    "is_compatible",
    "infer_scalar_type",
    "lookup_column_type",
    "derive_select_schema",
    "derive_filter_schema",
    "derive_group_schema",
    "derive_join_schema",
]
