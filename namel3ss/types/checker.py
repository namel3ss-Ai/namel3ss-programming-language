"""Static type checker for Namel3ss programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

from namel3ss.ast import (
    Action,
    App,
    AttributeRef,
    BinaryOp,
    CallExpression,
    ContextValue,
    Dataset,
    ForLoop,
    Frame,
    IfBlock,
    Literal,
    Module,
    NameRef,
    Page,
    PageStatement,
    Prompt,
    PromptField,
    RunPromptOperation,
    ShowChart,
    ShowForm,
    ShowTable,
    UnaryOp,
    VariableAssignment,
)
from namel3ss.ast.datasets import ComputedColumnOp, FilterOp, GroupByOp, JoinOp, OrderByOp
from namel3ss.errors import N3TypeError
from namel3ss.types import (
    ANY_TYPE,
    AnyType,
    DatasetType,
    FrameColumnType,
    FrameTypeRef,
    N3FrameType,
    N3Type,
    PromptIOTypes,
    ScalarKind,
    ScalarType,
    is_assignable,
    is_compatible,
    lookup_column_type,
)

@dataclass
class TypeEnvironment:
    path: Optional[str]
    datasets: Dict[str, DatasetType] = field(default_factory=dict)
    frames: Dict[str, FrameTypeRef] = field(default_factory=dict)
    prompts: Dict[str, PromptIOTypes] = field(default_factory=dict)
    variables: Dict[str, N3Type] = field(default_factory=dict)


class TypeScope:
    """Simple lexical scope stack for page-level statements."""

    def __init__(self, base: Optional[Dict[str, N3Type]] = None) -> None:
        self._stack: list[Dict[str, N3Type]] = [dict(base or {})]

    def push(self) -> None:
        self._stack.append({})

    def pop(self) -> None:
        if len(self._stack) == 1:  # pragma: no cover - defensive
            raise RuntimeError("Cannot pop root scope")
        self._stack.pop()

    def assign(self, name: str, value_type: N3Type) -> None:
        self._stack[-1][name] = value_type

    def lookup(self, name: str) -> Optional[N3Type]:
        for scope in reversed(self._stack):
            if name in scope:
                return scope[name]
        return None

    @property
    def globals(self) -> Dict[str, N3Type]:
        return self._stack[0]


class AppTypeChecker:
    """Encapsulates the logic required to type-check a single app."""

    def __init__(self, *, path: Optional[str] = None) -> None:
        self.env = TypeEnvironment(path=path)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def check_module(self, module: Module) -> None:
        if not module.body:
            return
        for node in module.body:
            if isinstance(node, App):
                self.check_app(node, path=module.path or module.name)

    def check_app(self, app: App, *, path: Optional[str] = None) -> None:
        if path:
            self.env.path = path
        self.env.datasets.clear()
        self.env.frames.clear()
        self.env.prompts.clear()
        self.env.variables = {}
        self._register_datasets(app.datasets)
        self._register_frames(app.frames)
        self._register_prompts(app.prompts)
        self._check_datasets(app.datasets)
        self._check_frames(app.frames)
        self._check_app_variables(app)
        self._check_pages(app.pages)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def _register_datasets(self, datasets: Sequence[Dataset]) -> None:
        for dataset in datasets:
            frame = self._frame_from_dataset(dataset)
            self.env.datasets[dataset.name] = DatasetType(frame=frame, source=dataset.name)

    def _register_frames(self, frames: Sequence[Frame]) -> None:
        for frame in frames:
            schema = N3FrameType.from_columns(frame.columns)
            self.env.frames[frame.name] = FrameTypeRef(schema=schema, label=frame.name)

    def _register_prompts(self, prompts: Sequence[Prompt]) -> None:
        for prompt in prompts:
            inputs = {field.name: self._type_from_prompt_field(field) for field in prompt.input_fields}
            outputs = {field.name: self._type_from_prompt_field(field) for field in prompt.output_fields}
            self.env.prompts[prompt.name] = PromptIOTypes(inputs=inputs, outputs=outputs)

    def _frame_from_dataset(self, dataset: Dataset) -> FrameTypeRef:
        columns = dataset.schema or []
        if columns:
            schema = N3FrameType.from_columns(columns)
        else:
            schema = N3FrameType(columns={}, order=[], key=[], splits={})
        return FrameTypeRef(schema=schema, label=dataset.name)

    # ------------------------------------------------------------------
    # Dataset checks
    # ------------------------------------------------------------------
    def _check_datasets(self, datasets: Sequence[Dataset]) -> None:
        for dataset in datasets:
            dataset_type = self.env.datasets.get(dataset.name)
            if dataset_type is None:
                continue
            updated_frame = self._check_dataset_pipeline(dataset, dataset_type.frame)
            self.env.datasets[dataset.name] = DatasetType(frame=updated_frame, source=dataset_type.source)

    def _check_dataset_pipeline(self, dataset: Dataset, starting_frame: FrameTypeRef) -> FrameTypeRef:
        frame = starting_frame
        for op in dataset.operations:
            resolver = self._column_resolver(frame)
            if isinstance(op, FilterOp):
                expr_type = self._infer_expression(op.condition, scope=None, column_resolver=resolver)
                self._ensure_boolean(expr_type, f"Filter for dataset '{dataset.name}' must be boolean.")
                continue
            if isinstance(op, GroupByOp):
                self._ensure_columns(dataset.name, frame, op.columns)
                continue
            if isinstance(op, OrderByOp):
                self._ensure_columns(dataset.name, frame, op.columns)
                continue
            if isinstance(op, ComputedColumnOp):
                frame = self._apply_computed_column(dataset, frame, op)
                continue
            if isinstance(op, JoinOp):
                frame = self._apply_dataset_join(dataset, frame, op)
                continue
        return frame

    def _column_resolver(self, frame: FrameTypeRef) -> Callable[[str], Optional[ScalarType]]:
        return lambda name: lookup_column_type(frame, name)

    def _ensure_columns(self, dataset_name: str, frame: FrameTypeRef, columns: Sequence[str]) -> None:
        for column in columns or []:
            if lookup_column_type(frame, column) is None:
                self._raise(f"Dataset '{dataset_name}' does not define column '{column}'.")

    def _apply_computed_column(self, dataset: Dataset, frame: FrameTypeRef, op: ComputedColumnOp) -> FrameTypeRef:
        resolver = self._column_resolver(frame)
        expr_type = self._infer_expression(op.expression, scope=None, column_resolver=resolver)
        scalar_type = self._expect_scalar(expr_type, f"Computed column '{op.name}' on dataset '{dataset.name}' must be scalar.")
        if op.name in frame.schema.columns:
            self._raise(f"Dataset '{dataset.name}' already defines column '{op.name}'.")
        return self._extend_frame(frame, op.name, scalar_type)

    def _apply_dataset_join(self, dataset: Dataset, frame: FrameTypeRef, op: JoinOp) -> FrameTypeRef:
        target_frame: Optional[FrameTypeRef] = None
        resolver = self._column_resolver(frame)
        if op.target_type == "dataset":
            target_dataset = self.env.datasets.get(op.target_name)
            if target_dataset is None:
                self._raise(f"Dataset '{dataset.name}' joins unknown dataset '{op.target_name}'.")
            target_frame = target_dataset.frame
            resolver = self._combined_resolver(frame, target_frame)
        if op.condition is not None:
            condition_type = self._infer_expression(op.condition, scope=None, column_resolver=resolver)
            self._ensure_boolean(condition_type, f"Join condition on dataset '{dataset.name}' must be boolean.")
        if target_frame is not None:
            return self._merge_frames(frame, target_frame, dataset.name, op.target_name)
        return frame

    def _combined_resolver(
        self,
        left: FrameTypeRef,
        right: FrameTypeRef,
    ) -> Callable[[str], Optional[ScalarType]]:
        def resolver(name: str) -> Optional[ScalarType]:
            column = lookup_column_type(left, name)
            if column is not None:
                return column
            return lookup_column_type(right, name)

        return resolver

    def _extend_frame(self, frame: FrameTypeRef, column_name: str, column_type: ScalarType) -> FrameTypeRef:
        columns = dict(frame.schema.columns)
        order = list(frame.schema.order)
        if column_name not in columns:
            order.append(column_name)
        columns[column_name] = FrameColumnType(
            name=column_name,
            dtype=column_type.kind.value,
            nullable=column_type.nullable,
        )
        new_schema = N3FrameType(
            columns=columns,
            order=order,
            key=list(frame.schema.key),
            splits=dict(frame.schema.splits),
        )
        return FrameTypeRef(schema=new_schema, label=frame.label)

    def _merge_frames(
        self,
        left: FrameTypeRef,
        right: FrameTypeRef,
        left_name: str,
        right_name: str,
    ) -> FrameTypeRef:
        columns = dict(left.schema.columns)
        order = list(left.schema.order)
        for name in right.schema.order:
            if name in columns:
                continue
            columns[name] = right.schema.columns[name]
            order.append(name)
        merged_schema = N3FrameType(
            columns=columns,
            order=order,
            key=list(left.schema.key),
            splits=dict(left.schema.splits),
        )
        return FrameTypeRef(schema=merged_schema, label=left.label)

    def _check_frames(self, frames: Sequence[Frame]) -> None:
        for frame in frames:
            if frame.source_type == "dataset" and frame.source:
                if frame.source not in self.env.datasets:
                    self._raise(f"Frame '{frame.name}' references unknown dataset '{frame.source}'.")

    # ------------------------------------------------------------------
    # Page-level checks
    # ------------------------------------------------------------------
    def _check_app_variables(self, app: App) -> None:
        scope = TypeScope(base=self.env.variables)
        for assignment in app.variables:
            value_type = self._infer_expression(assignment.value, scope)
            scope.assign(assignment.name, value_type)
        self.env.variables.update(scope.globals)

    def _check_pages(self, pages: Sequence[Page]) -> None:
        for page in pages:
            scope = TypeScope(base=self.env.variables)
            for statement in page.statements:
                self._check_statement(statement, scope, page)

    def _check_statement(self, statement: PageStatement, scope: TypeScope, page: Page) -> None:
        if isinstance(statement, VariableAssignment):
            value_type = self._infer_expression(statement.value, scope)
            scope.assign(statement.name, value_type)
            return
        if isinstance(statement, IfBlock):
            self._check_if_block(statement, scope, page)
            return
        if isinstance(statement, ForLoop):
            loop_type = self._resolve_loop_source(statement, page)
            scope.push()
            scope.assign(statement.loop_var, loop_type)
            for inner in statement.body:
                self._check_statement(inner, scope, page)
            scope.pop()
            return
        if isinstance(statement, ShowForm):
            for op in statement.on_submit_ops:
                self._check_action_operation(op, scope, page)
            return
        if isinstance(statement, Action):
            for op in statement.operations:
                self._check_action_operation(op, scope, page)
            return
        if isinstance(statement, ShowTable):
            self._ensure_data_source(statement.source_type, statement.source, page)
            return
        if isinstance(statement, ShowChart):
            self._ensure_data_source(statement.source_type, statement.source, page)
            return

    def _check_if_block(self, block: IfBlock, scope: TypeScope, page: Page) -> None:
        condition_type = self._infer_expression(block.condition, scope)
        self._ensure_boolean(condition_type, f"Condition on page '{page.name}' must be boolean.")
        self._check_block(block.body, scope, page)
        for elif_block in block.elifs:
            cond_type = self._infer_expression(elif_block.condition, scope)
            self._ensure_boolean(cond_type, f"Condition on page '{page.name}' must be boolean.")
            self._check_block(elif_block.body, scope, page)
        if block.else_body:
            self._check_block(block.else_body, scope, page)

    def _check_block(self, statements: Sequence[PageStatement], scope: TypeScope, page: Page) -> None:
        if not statements:
            return
        scope.push()
        for inner in statements:
            self._check_statement(inner, scope, page)
        scope.pop()

    def _check_action_operation(self, operation: Any, scope: TypeScope, page: Page) -> None:
        if isinstance(operation, RunPromptOperation):
            self._check_run_prompt(operation, scope, page)

    def _check_run_prompt(self, operation: RunPromptOperation, scope: TypeScope, page: Page) -> None:
        prompt = self.env.prompts.get(operation.prompt_name)
        if prompt is None:
            self._raise(f"Prompt '{operation.prompt_name}' is not defined but is used on page '{page.name}'.")
        missing = [name for name in prompt.inputs if name not in operation.arguments]
        if missing:
            joined = ", ".join(sorted(missing))
            self._raise(
                f"Prompt '{operation.prompt_name}' is missing required inputs ({joined}) on page '{page.name}'."
            )
        extra = [name for name in operation.arguments if name not in prompt.inputs]
        if extra:
            joined = ", ".join(sorted(extra))
            self._raise(f"Prompt '{operation.prompt_name}' does not accept inputs ({joined}).")
        for name, expected in prompt.inputs.items():
            arg_expr = operation.arguments[name]
            actual = self._infer_expression(arg_expr, scope)
            if not is_assignable(actual, expected):
                self._raise(
                    f"Prompt '{operation.prompt_name}' expects input '{name}' of type {expected} but received {actual}."
                )

    def _ensure_data_source(self, source_kind: str, source_name: Optional[str], page: Page) -> None:
        if not source_name:
            return
        if source_kind == "dataset":
            if source_name not in self.env.datasets:
                self._raise(f"Page '{page.name}' references unknown dataset '{source_name}'.")
        elif source_kind == "frame":
            if source_name not in self.env.frames:
                self._raise(f"Page '{page.name}' references unknown frame '{source_name}'.")

    def _resolve_loop_source(self, loop: ForLoop, page: Page) -> FrameTypeRef:
        if loop.source_kind == "dataset":
            dataset = self.env.datasets.get(loop.source_name)
            if dataset is None:
                self._raise(f"Loop on page '{page.name}' references unknown dataset '{loop.source_name}'.")
            return dataset.frame
        if loop.source_kind == "frame":
            frame = self.env.frames.get(loop.source_name)
            if frame is None:
                self._raise(f"Loop on page '{page.name}' references unknown frame '{loop.source_name}'.")
            return frame
        empty_schema = N3FrameType(columns={}, order=[], key=[], splits={})
        return FrameTypeRef(schema=empty_schema, label=loop.source_name or loop.loop_var)

    # ------------------------------------------------------------------
    # Expression inference
    # ------------------------------------------------------------------
    def _infer_expression(
        self,
        expr: Any,
        scope: Optional[TypeScope],
        column_resolver: Optional[Callable[[str], Optional[ScalarType]]] = None,
    ) -> N3Type:
        if isinstance(expr, Literal):
            return self._literal_type(expr.value)
        if isinstance(expr, NameRef):
            type_info = scope.lookup(expr.name) if scope else None
            if type_info is not None:
                return type_info
            if column_resolver:
                column_type = column_resolver(expr.name)
                if column_type is not None:
                    return column_type
            dataset = self.env.datasets.get(expr.name)
            if dataset is not None:
                return dataset
            frame = self.env.frames.get(expr.name)
            if frame is not None:
                return frame
            self._raise(f"Unknown reference '{expr.name}'.")
        if isinstance(expr, AttributeRef):
            base_type = self._infer_expression(NameRef(expr.base), scope, column_resolver)
            frame = None
            if isinstance(base_type, DatasetType):
                frame = base_type.frame
            elif isinstance(base_type, FrameTypeRef):
                frame = base_type
            if frame is None:
                self._raise(f"Cannot access attribute '{expr.attr}' on non-frame reference '{expr.base}'.")
            column_type = lookup_column_type(frame, expr.attr)
            if column_type is None:
                self._raise(f"Frame '{frame.label or expr.base}' does not define column '{expr.attr}'.")
            return column_type
        if isinstance(expr, UnaryOp):
            operand = self._infer_expression(expr.operand, scope, column_resolver)
            if expr.op == "not":
                self._ensure_boolean(operand, "Operand for 'not' must be boolean.")
                return ScalarType(ScalarKind.BOOL)
            if expr.op in {"-", "+"}:
                if not self._is_numeric(operand):
                    self._raise(f"Unary operator '{expr.op}' requires a numeric operand.")
                return operand
            return ANY_TYPE
        if isinstance(expr, BinaryOp):
            left = self._infer_expression(expr.left, scope, column_resolver)
            right = self._infer_expression(expr.right, scope, column_resolver)
            op = expr.op
            if op in {"==", "!="}:
                if not is_compatible(left, right):
                    self._raise(f"Operands for '{op}' are incompatible: {left} vs {right}.")
                return ScalarType(ScalarKind.BOOL)
            if op in {"and", "or"}:
                self._ensure_boolean(left, "Logical operands must be boolean.")
                self._ensure_boolean(right, "Logical operands must be boolean.")
                return ScalarType(ScalarKind.BOOL)
            if op in {">", ">=", "<", "<="}:
                if not self._is_numeric_or_datetime(left) or not self._is_numeric_or_datetime(right):
                    self._raise(f"Relational operator '{op}' requires numeric or datetime operands.")
                return ScalarType(ScalarKind.BOOL)
            if op in {"+", "-", "*", "/"}:
                if not self._is_numeric(left) or not self._is_numeric(right):
                    self._raise(f"Arithmetic operator '{op}' requires numeric operands.")
                if self._uses_float(left, right) or op == "/":
                    return ScalarType(ScalarKind.FLOAT)
                return ScalarType(ScalarKind.INT)
            return ANY_TYPE
        if isinstance(expr, ContextValue):
            return ANY_TYPE
        if isinstance(expr, CallExpression):
            return ANY_TYPE
        return ANY_TYPE

    def _literal_type(self, value: Any) -> ScalarType:
        if isinstance(value, bool):
            return ScalarType(ScalarKind.BOOL)
        if isinstance(value, int):
            return ScalarType(ScalarKind.INT)
        if isinstance(value, float):
            return ScalarType(ScalarKind.FLOAT)
        if isinstance(value, str):
            return ScalarType(ScalarKind.STRING)
        return ScalarType(ScalarKind.ANY, nullable=True)

    def _ensure_boolean(self, value_type: N3Type, message: str) -> None:
        if not isinstance(value_type, ScalarType) or value_type.kind != ScalarKind.BOOL:
            self._raise(message)

    def _expect_scalar(self, value_type: N3Type, message: str) -> ScalarType:
        if isinstance(value_type, ScalarType):
            return value_type
        if isinstance(value_type, AnyType):
            return ScalarType(ScalarKind.ANY, nullable=value_type.nullable)
        self._raise(message)

    def _is_numeric(self, value_type: N3Type) -> bool:
        return isinstance(value_type, ScalarType) and value_type.kind in {ScalarKind.INT, ScalarKind.FLOAT}

    def _is_numeric_or_datetime(self, value_type: N3Type) -> bool:
        return isinstance(value_type, ScalarType) and value_type.kind in {
            ScalarKind.INT,
            ScalarKind.FLOAT,
            ScalarKind.DATETIME,
        }

    def _uses_float(self, left: N3Type, right: N3Type) -> bool:
        return any(
            isinstance(value, ScalarType) and value.kind == ScalarKind.FLOAT for value in (left, right)
        )

    def _type_from_prompt_field(self, field: PromptField) -> ScalarType:
        mapping = {
            "text": ScalarKind.STRING,
            "string": ScalarKind.STRING,
            "str": ScalarKind.STRING,
            "number": ScalarKind.FLOAT,
            "float": ScalarKind.FLOAT,
            "integer": ScalarKind.INT,
            "int": ScalarKind.INT,
            "bool": ScalarKind.BOOL,
            "boolean": ScalarKind.BOOL,
        }
        kind = mapping.get(field.field_type.lower(), ScalarKind.STRING)
        return ScalarType(kind, nullable=not field.required)

    def _raise(self, message: str, *, code: str = "TYPE_ERROR", hint: Optional[str] = None) -> None:
        raise N3TypeError(message, path=self.env.path, code=code, hint=hint)


def check_module(module: Module) -> None:
    """Run the type checker against ``module``."""

    checker = AppTypeChecker(path=module.path or module.name)
    checker.check_module(module)


def check_app(app: App, *, path: Optional[str] = None) -> None:
    """Run the type checker against a standalone ``app`` instance."""

    checker = AppTypeChecker(path=path)
    checker.check_app(app, path=path)


__all__ = ["check_module", "check_app"]
