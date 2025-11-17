"""Type analysis and lowering helpers for frame expressions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from namel3ss.ast import (
    Expression,
    Frame,
    FrameExpression,
    FrameFilter,
    FrameGroupBy,
    FrameJoin,
    FrameOrderBy,
    FrameRef,
    FrameSelect,
    FrameSummarise,
    NameRef,
)


class FrameTypeError(ValueError):
    """Raised when a frame expression fails static validation."""


@dataclass(frozen=True)
class FrameColumnInfo:
    name: str
    dtype: str
    nullable: bool
    role: Optional[str] = None


@dataclass
class FrameSchemaInfo:
    columns: Dict[str, FrameColumnInfo]
    order: List[str]
    key: List[str]
    splits: Dict[str, float]

    def ensure_columns(self, frame_name: str, column_names: Sequence[str]) -> None:
        missing = [name for name in column_names if name not in self.columns]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FrameTypeError(f"Frame '{frame_name}' does not define columns: {joined}")

    def subset(self, column_names: Sequence[str]) -> FrameSchemaInfo:
        ordered: List[str] = []
        projection: Dict[str, FrameColumnInfo] = {}
        for name in column_names:
            if name not in self.columns:
                continue
            projection[name] = self.columns[name]
            ordered.append(name)
        filtered_key = [name for name in self.key if name in projection]
        return FrameSchemaInfo(projection, ordered, filtered_key, dict(self.splits))

    def with_aggregations(
        self,
        group_columns: Sequence[str],
        aggregations: Sequence["AggregationSpec"],
    ) -> FrameSchemaInfo:
        projection = {name: self.columns[name] for name in group_columns if name in self.columns}
        ordered = list(group_columns)
        for agg in aggregations:
            projection[agg.name] = FrameColumnInfo(
                name=agg.name,
                dtype=agg.dtype,
                nullable=True,
                role=None,
            )
            ordered.append(agg.name)
        return FrameSchemaInfo(projection, ordered, list(group_columns), dict(self.splits))

    def merge(
        self,
        other: FrameSchemaInfo,
        join_on: Sequence[str],
        frame_name: str,
        right_name: str,
    ) -> FrameSchemaInfo:
        merged = dict(self.columns)
        ordered = list(self.order)
        join_key_set = set(join_on)
        for name in other.order:
            if name in join_key_set:
                continue
            if name in merged:
                raise FrameTypeError(
                    f"Join between '{frame_name}' and '{right_name}' produces duplicate column '{name}'."
                )
            merged[name] = other.columns[name]
            ordered.append(name)
        return FrameSchemaInfo(merged, ordered, list(self.key), dict(self.splits))

    def to_payload(self) -> Dict[str, Any]:
        columns_payload = []
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


@dataclass
class AggregationSpec:
    name: str
    function: str
    expression: Optional[Expression]
    dtype: str

    def to_payload(
        self,
        encode_expression: Callable[[Optional[Expression]], Any],
        encode_expression_source: Callable[[Optional[Expression]], Optional[str]],
    ) -> Dict[str, Any]:
        return {
            "name": self.name,
            "function": self.function,
            "dtype": self.dtype,
            "expression": encode_expression(self.expression),
            "expression_source": encode_expression_source(self.expression),
        }


@dataclass
class FrameOperationSpec:
    op: str
    columns: List[str] = field(default_factory=list)
    predicate: Optional[Expression] = None
    descending: bool = False
    aggregations: List[AggregationSpec] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    join_target: Optional[str] = None
    join_on: List[str] = field(default_factory=list)
    join_how: str = "inner"

    def to_payload(
        self,
        encode_expression: Callable[[Optional[Expression]], Any],
        encode_expression_source: Callable[[Optional[Expression]], Optional[str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"op": self.op}
        if self.columns:
            payload["columns"] = list(self.columns)
        if self.group_by:
            payload["group_by"] = list(self.group_by)
        if self.predicate is not None:
            payload["predicate"] = encode_expression(self.predicate)
            payload["predicate_source"] = encode_expression_source(self.predicate)
        if self.aggregations:
            payload["aggregations"] = [
                aggregation.to_payload(encode_expression, encode_expression_source)
                for aggregation in self.aggregations
            ]
        if self.join_target:
            payload["join_target"] = self.join_target
            payload["join_on"] = list(self.join_on)
            payload["join_how"] = self.join_how
        if self.op == "order_by":
            payload["descending"] = self.descending
        return payload


@dataclass
class FramePipelinePlan:
    root: str
    schema: FrameSchemaInfo
    operations: List[FrameOperationSpec]

    def to_payload(
        self,
        encode_expression: Callable[[Optional[Expression]], Any],
        encode_expression_source: Callable[[Optional[Expression]], Optional[str]],
    ) -> Dict[str, Any]:
        return {
            "root": self.root,
            "schema": self.schema.to_payload(),
            "operations": [
                op.to_payload(encode_expression, encode_expression_source) for op in self.operations
            ],
        }


@dataclass
class FramePipelineState:
    root: str
    schema: FrameSchemaInfo
    operations: List[FrameOperationSpec]
    group_by: List[str] = field(default_factory=list)


class FrameExpressionAnalyzer:
    """Infer schemas for frame expressions and surface validation errors."""

    _RESERVED_IDENTIFIERS = {
        "true",
        "false",
        "none",
        "null",
        "len",
        "sum",
        "avg",
        "mean",
        "min",
        "max",
        "count",
        "abs",
        "round",
        "int",
        "float",
        "str",
        "bool",
    }

    _AGGREGATION_TYPES = {
        "sum": "number",
        "avg": "number",
        "mean": "number",
        "min": "number",
        "max": "number",
        "count": "int",
    }

    def __init__(self, frames: Sequence[Frame]):
        self._frames = {frame.name: frame for frame in frames}
        self._schemas = {name: self._build_schema(frame) for name, frame in self._frames.items()}

    def analyze(self, expression: FrameExpression) -> FramePipelinePlan:
        state = self._analyze_expression(expression)
        return FramePipelinePlan(root=state.root, schema=state.schema, operations=state.operations)

    def _analyze_expression(self, expression: FrameExpression) -> FramePipelineState:
        if isinstance(expression, FrameRef):
            schema = self._schema_for_frame(expression.name)
            return FramePipelineState(root=expression.name, schema=schema, operations=[])

        if isinstance(expression, FrameFilter):
            state = self._analyze_expression(expression.source)
            self._validate_expression_columns(expression.predicate, state.schema, state.root)
            operations = list(state.operations)
            operations.append(FrameOperationSpec(op="filter", predicate=expression.predicate))
            return FramePipelineState(state.root, state.schema, operations, state.group_by)

        if isinstance(expression, FrameSelect):
            state = self._analyze_expression(expression.source)
            state.schema.ensure_columns(state.root, expression.columns)
            new_schema = state.schema.subset(expression.columns)
            operations = list(state.operations)
            operations.append(FrameOperationSpec(op="select", columns=list(expression.columns)))
            return FramePipelineState(state.root, new_schema, operations)

        if isinstance(expression, FrameOrderBy):
            state = self._analyze_expression(expression.source)
            state.schema.ensure_columns(state.root, expression.columns)
            operations = list(state.operations)
            operations.append(
                FrameOperationSpec(
                    op="order_by",
                    columns=list(expression.columns),
                    descending=bool(expression.descending),
                )
            )
            return FramePipelineState(state.root, state.schema, operations, state.group_by)

        if isinstance(expression, FrameGroupBy):
            state = self._analyze_expression(expression.source)
            state.schema.ensure_columns(state.root, expression.columns)
            operations = list(state.operations)
            group_columns = list(expression.columns)
            operations.append(FrameOperationSpec(op="group_by", columns=group_columns))
            return FramePipelineState(state.root, state.schema, operations, group_columns)

        if isinstance(expression, FrameSummarise):
            state = self._analyze_expression(expression.source)
            aggregations: List[AggregationSpec] = []
            for name, agg_expr in expression.aggregations.items():
                aggregations.append(self._parse_aggregation(name, agg_expr, state.schema, state.root))
            new_schema = state.schema.with_aggregations(state.group_by, aggregations)
            operations = list(state.operations)
            operations.append(
                FrameOperationSpec(op="summarise", aggregations=aggregations, group_by=list(state.group_by))
            )
            return FramePipelineState(state.root, new_schema, operations)

        if isinstance(expression, FrameJoin):
            state = self._analyze_expression(expression.left)
            right_schema = self._schema_for_frame(expression.right)
            state.schema.ensure_columns(state.root, expression.on)
            right_schema.ensure_columns(expression.right, expression.on)
            merged_schema = state.schema.merge(right_schema, expression.on, state.root, expression.right)
            operations = list(state.operations)
            operations.append(
                FrameOperationSpec(
                    op="join",
                    join_target=expression.right,
                    join_on=list(expression.on),
                    join_how=expression.how,
                )
            )
            return FramePipelineState(state.root, merged_schema, operations)

        raise FrameTypeError(f"Unsupported frame expression '{type(expression).__name__}'")

    def _schema_for_frame(self, frame_name: str) -> FrameSchemaInfo:
        schema = self._schemas.get(frame_name)
        if schema is None:
            raise FrameTypeError(f"Frame '{frame_name}' is not defined")
        return schema

    def _build_schema(self, frame: Frame) -> FrameSchemaInfo:
        columns: Dict[str, FrameColumnInfo] = {}
        order: List[str] = []
        for column in frame.columns:
            if not column.name:
                continue
            columns[column.name] = FrameColumnInfo(
                name=column.name,
                dtype=str(column.dtype or "string"),
                nullable=bool(column.nullable),
                role=column.role,
            )
            order.append(column.name)
        return FrameSchemaInfo(columns, order, list(frame.key or []), dict(frame.splits or {}))

    def _parse_aggregation(
        self,
        name: str,
        expression: Expression,
        schema: FrameSchemaInfo,
        frame_name: str,
    ) -> AggregationSpec:
        from namel3ss.ast import CallExpression

        if not isinstance(expression, CallExpression) or not isinstance(expression.function, NameRef):
            raise FrameTypeError(f"Aggregation '{name}' must call a function (e.g. sum(metric))")
        func_name = expression.function.name.lower()
        canonical = "avg" if func_name == "mean" else func_name
        if canonical not in self._AGGREGATION_TYPES:
            allowed = ", ".join(sorted(self._AGGREGATION_TYPES))
            raise FrameTypeError(
                f"Aggregation '{name}' uses unsupported function '{func_name}'. Allowed: {allowed}."
            )
        argument = expression.arguments[0] if expression.arguments else None
        if argument is not None:
            self._validate_expression_columns(argument, schema, frame_name)
        dtype = self._AGGREGATION_TYPES[canonical]
        return AggregationSpec(name=name, function=canonical, expression=argument, dtype=dtype)

    def _validate_expression_columns(
        self,
        expression: Optional[Expression],
        schema: FrameSchemaInfo,
        frame_name: str,
    ) -> None:
        if expression is None:
            return
        referenced = self._collect_name_refs(expression)
        missing = [
            name
            for name in referenced
            if name not in schema.columns and name.lower() not in self._RESERVED_IDENTIFIERS
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FrameTypeError(
                f"Frame '{frame_name}' does not define columns referenced in expression: {joined}"
            )

    def _collect_name_refs(self, expression: Expression) -> Set[str]:
        names: Set[str] = set()

        def visit(node: Optional[Expression]) -> None:
            if node is None:
                return
            if isinstance(node, NameRef):
                names.add(node.name)
                return
            if isinstance(node, FrameExpression):
                return
            if is_dataclass(node):
                for field_obj in fields(node):
                    value = getattr(node, field_obj.name)
                    self._gather_child_expressions(value, visit)

        visit(expression)
        return names

    def _gather_child_expressions(self, value: object, visit: Callable[[Expression], None]) -> None:
        if isinstance(value, Expression):
            visit(value)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._gather_child_expressions(item, visit)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._gather_child_expressions(item, visit)
            return