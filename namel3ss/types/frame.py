"""Type representations for N3Frame schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FrameColumnType:
    """Describes the scalar type information for a single frame column."""

    name: str
    dtype: str
    nullable: bool
    role: Optional[str] = None


@dataclass
class N3FrameType:
    """Represents the static type (schema) of an N3Frame."""

    columns: Dict[str, FrameColumnType]
    order: List[str]
    key: List[str]
    splits: Dict[str, float]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_columns(
        cls,
        columns: Sequence[object],
        *,
        key: Optional[Sequence[str]] = None,
        splits: Optional[Dict[str, float]] = None,
    ) -> N3FrameType:
        column_index: Dict[str, FrameColumnType] = {}
        ordered: List[str] = []
        for column in columns or []:
            name = getattr(column, "name", None)
            if not name:
                continue
            dtype = getattr(column, "dtype", "string") or "string"
            nullable = bool(getattr(column, "nullable", True))
            role = getattr(column, "role", None)
            column_index[name] = FrameColumnType(
                name=name,
                dtype=str(dtype),
                nullable=nullable,
                role=role,
            )
            ordered.append(name)
        return cls(
            columns=column_index,
            order=ordered,
            key=list(key or []),
            splits=dict(splits or {}),
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def ensure_columns(
        self,
        frame_name: str,
        column_names: Sequence[str],
        *,
        error_cls: type[Exception] = ValueError,
    ) -> None:
        missing = [name for name in column_names if name not in self.columns]
        if missing:
            joined = ", ".join(sorted(missing))
            raise error_cls(f"Frame '{frame_name}' does not define columns: {joined}")

    def subset(self, column_names: Sequence[str]) -> N3FrameType:
        ordered: List[str] = []
        projection: Dict[str, FrameColumnType] = {}
        for name in column_names:
            column = self.columns.get(name)
            if column is None:
                continue
            projection[name] = column
            ordered.append(name)
        filtered_key = [name for name in self.key if name in projection]
        return N3FrameType(projection, ordered, filtered_key, dict(self.splits))

    def with_aggregations(
        self,
        group_columns: Sequence[str],
        aggregations: Sequence[object],
    ) -> N3FrameType:
        projection = {name: self.columns[name] for name in group_columns if name in self.columns}
        ordered = list(group_columns)
        for agg in aggregations:
            column_name = getattr(agg, "name")
            dtype = getattr(agg, "dtype", "string")
            projection[column_name] = FrameColumnType(
                name=column_name,
                dtype=dtype,
                nullable=True,
                role=None,
            )
            ordered.append(column_name)
        return N3FrameType(projection, ordered, list(group_columns), dict(self.splits))

    def merge(
        self,
        other: N3FrameType,
        join_on: Sequence[str],
        frame_name: str,
        right_name: str,
        *,
        error_cls: type[Exception] = ValueError,
    ) -> N3FrameType:
        merged = dict(self.columns)
        ordered = list(self.order)
        join_key_set = set(join_on)
        for name in other.order:
            if name in join_key_set:
                continue
            if name in merged:
                raise error_cls(
                    f"Join between '{frame_name}' and '{right_name}' produces duplicate column '{name}'."
                )
            merged[name] = other.columns[name]
            ordered.append(name)
        return N3FrameType(merged, ordered, list(self.key), dict(self.splits))

    def to_payload(self) -> Dict[str, object]:
        columns_payload: List[Dict[str, object]] = []
        for name in self.order:
            info = self.columns[name]
            columns_payload.append(
                {
                    "name": info.name,
                    "dtype": info.dtype,
                    "nullable": info.nullable,
                    "role": info.role,
                }
            )
        return {
            "columns": columns_payload,
            "key": list(self.key),
            "splits": dict(self.splits),
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def describe(self) -> List[Tuple[str, str]]:
        """Return a list of (column, dtype) pairs preserving order."""

        return [(name, self.columns[name].dtype) for name in self.order if name in self.columns]

    def get_column(self, name: str) -> Optional[FrameColumnType]:
        return self.columns.get(name)
