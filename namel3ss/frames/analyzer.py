"""Type analysis and lowering helpers for frame expressions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from namel3ss.ast import (
    BinaryOp,
    CallExpression,
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
    Literal,
    NameRef,
    UnaryOp,
)
from namel3ss.types import N3FrameType


class FrameTypeError(ValueError):
    """Raised when a frame expression fails static validation."""


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
    schema: N3FrameType
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
    schema: N3FrameType
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

    _NUMERIC_DTYPES = {"int", "integer", "number", "float", "decimal"}
    _STRING_DTYPES = {"string", "text"}
    _BOOL_DTYPES = {"bool", "boolean"}
    _DATETIME_DTYPES = {"datetime", "timestamp", "date", "time"}
    _NULL_DTYPES = {"none", "null"}

    _AGGREGATION_SIGNATURES = {
        "sum": {"result": "number", "inputs": {"number"}},
        "avg": {"result": "number", "inputs": {"number"}},
        "min": {"result": "same", "inputs": {"number", "string", "datetime"}},
        "max": {"result": "same", "inputs": {"number", "string", "datetime"}},
        "count": {"result": "int", "inputs": {"any"}},
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
            self._ensure_boolean_expression(expression.predicate, state.schema, state.root, "filter")
            operations = list(state.operations)
            operations.append(FrameOperationSpec(op="filter", predicate=expression.predicate))
            return FramePipelineState(state.root, state.schema, operations, state.group_by)

        if isinstance(expression, FrameSelect):
            state = self._analyze_expression(expression.source)
            state.schema.ensure_columns(state.root, expression.columns, error_cls=FrameTypeError)
            new_schema = state.schema.subset(expression.columns)
            operations = list(state.operations)
            operations.append(FrameOperationSpec(op="select", columns=list(expression.columns)))
            return FramePipelineState(state.root, new_schema, operations)

        if isinstance(expression, FrameOrderBy):
            state = self._analyze_expression(expression.source)
            state.schema.ensure_columns(state.root, expression.columns, error_cls=FrameTypeError)
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
            state.schema.ensure_columns(state.root, expression.columns, error_cls=FrameTypeError)
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
            state.schema.ensure_columns(state.root, expression.on, error_cls=FrameTypeError)
            right_schema.ensure_columns(expression.right, expression.on, error_cls=FrameTypeError)
            self._validate_join_key_types(state.schema, right_schema, expression.on, state.root, expression.right)
            merged_schema = state.schema.merge(
                right_schema,
                expression.on,
                state.root,
                expression.right,
                error_cls=FrameTypeError,
            )
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

    def _schema_for_frame(self, frame_name: str) -> N3FrameType:
        schema = self._schemas.get(frame_name)
        if schema is None:
            raise FrameTypeError(f"Frame '{frame_name}' is not defined")
        return schema

    def _build_schema(self, frame: Frame) -> N3FrameType:
        schema = N3FrameType.from_columns(
            frame.columns,
            key=frame.key,
            splits=frame.splits,
        )
        frame.type_info = schema
        return schema

    def _parse_aggregation(
        self,
        name: str,
        expression: Expression,
        schema: N3FrameType,
        frame_name: str,
    ) -> AggregationSpec:
        if not isinstance(expression, CallExpression) or not isinstance(expression.function, NameRef):
            raise FrameTypeError(f"Aggregation '{name}' must call a function (e.g. sum(metric))")
        func_name = expression.function.name.lower()
        canonical = "avg" if func_name == "mean" else func_name
        if canonical not in self._AGGREGATION_SIGNATURES:
            allowed = ", ".join(sorted(self._AGGREGATION_SIGNATURES))
            raise FrameTypeError(
                f"Aggregation '{name}' uses unsupported function '{func_name}'. Allowed: {allowed}."
            )
        argument = expression.arguments[0] if expression.arguments else None
        if argument is None and canonical != "count":
            raise FrameTypeError(
                f"Aggregation '{name}' in frame '{frame_name}' must reference a column (e.g. sum(metric))."
            )
        argument_type = "unknown"
        if argument is not None:
            self._validate_expression_columns(argument, schema, frame_name)
            argument_type = self._infer_expression_type(
                argument,
                schema,
                frame_name,
                f"aggregation '{name}'",
            )
        result_dtype = self._resolve_aggregation_dtype(
            canonical,
            argument_type,
            name,
            frame_name,
            argument,
        )
        return AggregationSpec(name=name, function=canonical, expression=argument, dtype=result_dtype)

    def _ensure_boolean_expression(
        self,
        expression: Optional[Expression],
        schema: N3FrameType,
        frame_name: str,
        operation: str,
    ) -> None:
        expr_type = self._infer_expression_type(expression, schema, frame_name, operation)
        if expr_type not in {"bool", "unknown"}:
            raise FrameTypeError(
                f"{operation} on frame '{frame_name}' must evaluate to a boolean expression."
            )

    def _infer_expression_type(
        self,
        expression: Optional[Expression],
        schema: N3FrameType,
        frame_name: str,
        operation: str,
    ) -> str:
        if expression is None:
            return "unknown"
        if isinstance(expression, Literal):
            return self._infer_literal_type(expression.value)
        if isinstance(expression, NameRef):
            column = schema.get_column(expression.name)
            if column is not None:
                return self._canonical_dtype(column.dtype)
            lowered = expression.name.lower()
            if lowered in {"true", "false"}:
                return "bool"
            if lowered in self._NULL_DTYPES:
                return "null"
            if lowered in self._RESERVED_IDENTIFIERS:
                return "unknown"
            raise FrameTypeError(
                f"{operation} on frame '{frame_name}' references unknown column '{expression.name}'."
            )
        if isinstance(expression, UnaryOp):
            op = expression.op.lower()
            operand_type = self._infer_expression_type(expression.operand, schema, frame_name, operation)
            if op in {"not", "!"}:
                self._assert_type(
                    operand_type,
                    {"bool"},
                    operation,
                    frame_name,
                    self._expression_label(expression.operand),
                )
                return "bool"
            if op in {"-", "neg", "+", "pos"}:
                self._assert_type(
                    operand_type,
                    {"number"},
                    operation,
                    frame_name,
                    self._expression_label(expression.operand),
                )
                return "number"
            return "unknown"
        if isinstance(expression, BinaryOp):
            op = str(expression.op or "").lower()
            left_type = self._infer_expression_type(expression.left, schema, frame_name, operation)
            right_type = self._infer_expression_type(expression.right, schema, frame_name, operation)
            if op in {"and", "or", "&&", "||"}:
                self._assert_type(left_type, {"bool"}, operation, frame_name, self._expression_label(expression.left))
                self._assert_type(
                    right_type, {"bool"}, operation, frame_name, self._expression_label(expression.right)
                )
                return "bool"
            if op in {"==", "=", "!=", "<>"}:
                if not self._types_are_equalish(left_type, right_type):
                    raise FrameTypeError(
                        f"{operation} on frame '{frame_name}' cannot compare "
                        f"{self._expression_label(expression.left)} ({left_type}) "
                        f"with {self._expression_label(expression.right)} ({right_type})."
                    )
                return "bool"
            if op in {"<", "<=", ">", ">="}:
                allowed = {"number", "string", "datetime"}
                self._assert_type(left_type, allowed, operation, frame_name, self._expression_label(expression.left))
                self._assert_type(right_type, allowed, operation, frame_name, self._expression_label(expression.right))
                if left_type != "unknown" and right_type != "unknown" and left_type != right_type:
                    raise FrameTypeError(
                        f"{operation} on frame '{frame_name}' requires comparable operands but "
                        f"{self._expression_label(expression.left)} is '{left_type}' and "
                        f"{self._expression_label(expression.right)} is '{right_type}'."
                    )
                return "bool"
            if op == "+":
                if left_type == "string" or right_type == "string":
                    self._assert_type(left_type, {"string"}, operation, frame_name, self._expression_label(expression.left))
                    self._assert_type(
                        right_type, {"string"}, operation, frame_name, self._expression_label(expression.right)
                    )
                    return "string"
                self._assert_type(left_type, {"number"}, operation, frame_name, self._expression_label(expression.left))
                self._assert_type(right_type, {"number"}, operation, frame_name, self._expression_label(expression.right))
                return "number"
            if op in {"-", "*", "/", "%"}:
                self._assert_type(left_type, {"number"}, operation, frame_name, self._expression_label(expression.left))
                self._assert_type(right_type, {"number"}, operation, frame_name, self._expression_label(expression.right))
                return "number"
            if op == "in":
                self._validate_in_operator(
                    left_type,
                    right_type,
                    expression.left,
                    expression.right,
                    operation,
                    frame_name,
                )
                return "bool"
            if op in {"not in", "notin"}:
                self._validate_in_operator(
                    left_type,
                    right_type,
                    expression.left,
                    expression.right,
                    operation,
                    frame_name,
                )
                return "bool"
        if isinstance(expression, CallExpression):
            for arg in expression.arguments:
                self._infer_expression_type(arg, schema, frame_name, operation)
            return "unknown"
        return "unknown"

    def _infer_literal_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if value is None:
            return "null"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, (list, tuple, set)):
            element_types = {self._infer_literal_type(item) for item in value if item is not None}
            element_types.discard("null")
            if len(element_types) == 1:
                return f"list[{element_types.pop()}]"
            return "list"
        return "unknown"

    @classmethod
    def _canonical_dtype(cls, dtype: Optional[str]) -> str:
        if dtype is None:
            return "unknown"
        normalized = str(dtype).lower()
        if normalized in cls._NUMERIC_DTYPES:
            return "number"
        if normalized in cls._STRING_DTYPES:
            return "string"
        if normalized in cls._BOOL_DTYPES:
            return "bool"
        if normalized in cls._DATETIME_DTYPES:
            return "datetime"
        if normalized in cls._NULL_DTYPES:
            return "null"
        return normalized or "unknown"

    def _expression_label(self, expression: Expression) -> str:
        if isinstance(expression, NameRef):
            return f"column '{expression.name}'"
        if isinstance(expression, Literal):
            return f"literal {expression.value!r}"
        return "expression"

    def _assert_type(
        self,
        actual: str,
        allowed: Set[str],
        operation: str,
        frame_name: str,
        label: str,
    ) -> None:
        if actual == "unknown":
            return
        if actual not in allowed:
            allowed_fmt = self._format_allowed_types(allowed)
            raise FrameTypeError(
                f"{operation} on frame '{frame_name}' expects {allowed_fmt} but {label} is '{actual}'."
            )

    def _types_are_equalish(self, left: str, right: str) -> bool:
        if left == "unknown" or right == "unknown":
            return True
        if left == "null" or right == "null":
            return True
        return left == right

    def _validate_in_operator(
        self,
        left_type: str,
        right_type: str,
        left_expr: Expression,
        right_expr: Expression,
        operation: str,
        frame_name: str,
    ) -> None:
        if right_type.startswith("list["):
            element_type = right_type[5:-1]
            if left_type != "unknown" and element_type not in {"unknown", "list"} and left_type != element_type:
                raise FrameTypeError(
                    f"{operation} on frame '{frame_name}' cannot compare "
                    f"{self._expression_label(left_expr)} ({left_type}) with values of type '{element_type}'."
                )
            return
        if right_type in {"list", "unknown"}:
            return
        raise FrameTypeError(
            f"{operation} on frame '{frame_name}' requires a list of values on the right side of 'in'."
        )

    def _resolve_aggregation_dtype(
        self,
        function_name: str,
        argument_type: str,
        aggregation_name: str,
        frame_name: str,
        argument_expression: Optional[Expression],
    ) -> str:
        signature = self._AGGREGATION_SIGNATURES[function_name]
        allowed = signature["inputs"]
        if argument_expression is not None and allowed != {"any"} and argument_type not in {"unknown"}:
            if argument_type not in allowed:
                raise FrameTypeError(
                    f"Aggregation '{aggregation_name}' on frame '{frame_name}' requires "
                    f"{self._format_allowed_types(allowed)} input but "
                    f"{self._expression_label(argument_expression)} is '{argument_type}'."
                )
        result = signature["result"]
        if result == "same":
            if argument_type == "unknown":
                return "string"
            return argument_type
        return result

    def _format_allowed_types(self, allowed: Set[str]) -> str:
        ordered = sorted(allowed)
        if len(ordered) == 1:
            return ordered[0]
        return f"{', '.join(ordered[:-1])} or {ordered[-1]}"

    def _validate_join_key_types(
        self,
        left_schema: N3FrameType,
        right_schema: N3FrameType,
        join_columns: Sequence[str],
        left_name: str,
        right_name: str,
    ) -> None:
        for column in join_columns:
            left_info = left_schema.get_column(column)
            right_info = right_schema.get_column(column)
            if left_info is None or right_info is None:
                continue
            left_type = self._canonical_dtype(left_info.dtype)
            right_type = self._canonical_dtype(right_info.dtype)
            if left_type != "unknown" and right_type != "unknown" and left_type != right_type:
                raise FrameTypeError(
                    f"join between '{left_name}' and '{right_name}' expects matching dtypes "
                    f"for column '{column}' but found '{left_type}' and '{right_type}'."
                )

    def _validate_expression_columns(
        self,
        expression: Optional[Expression],
        schema: N3FrameType,
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
