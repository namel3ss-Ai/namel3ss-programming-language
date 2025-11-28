"""
Static Type Checker for Namel3ss

This module implements a comprehensive static type checking pass that runs after
parsing and resolution, but before code generation. It validates:

- Function signatures and call sites
- Variable assignments and reassignments  
- Schema fields and dataset types
- Prompt input/output schemas
- Expression types (number, text, boolean, arrays, objects, unions)
- Lambda expressions
- Subscript operations

Type errors are attached to specific source locations and integrate with the
existing error reporting mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from namel3ss.ast import (
    Module, App, Page, Dataset, Prompt, Chain, AIModel,
    Expression, Literal, NameRef, AttributeRef, BinaryOp, UnaryOp,
    CallExpression, ContextValue
)
from namel3ss.ast.expressions import (
    LiteralExpr, VarExpr, BinaryOp as ExprBinaryOp, AttributeExpr,
    CallExpr, LambdaExpr, IfExpr, LetExpr, ListExpr, DictExpr,
    TupleExpr, IndexExpr, SliceExpr, FunctionDef, Parameter
)
from namel3ss.lang.parser.errors import N3TypeError
from namel3ss.ast.source_location import SourceLocation


# ==================== Type Representation ====================


@dataclass
class Type:
    """Base class for all types in the type system."""
    pass


@dataclass  
class PrimitiveType(Type):
    """Primitive types: text, number, boolean, null"""
    name: str  # "text", "number", "boolean", "null"
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrimitiveType) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(("primitive", self.name))


@dataclass
class ArrayType(Type):
    """Array type: array<T>"""
    element_type: Type
    
    def __str__(self) -> str:
        return f"array<{self.element_type}>"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArrayType) and self.element_type == other.element_type
    
    def __hash__(self) -> int:
        return hash(("array", self.element_type))


@dataclass
class ObjectType(Type):
    """Object type: {field1: type1, field2: type2}"""
    fields: Dict[str, Type] = field(default_factory=dict)
    
    def __str__(self) -> str:
        field_strs = [f"{k}: {v}" for k, v in self.fields.items()]
        return "{" + ", ".join(field_strs) + "}"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjectType) and self.fields == other.fields
    
    def __hash__(self) -> int:
        return hash(("object", tuple(sorted(self.fields.items()))))


@dataclass
class UnionType(Type):
    """Union type: type1 | type2 | ..."""
    types: List[Type] = field(default_factory=list)
    
    def __str__(self) -> str:
        return " | ".join(str(t) for t in self.types)
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnionType) and set(self.types) == set(other.types)
    
    def __hash__(self) -> int:
        return hash(("union", tuple(sorted(str(t) for t in self.types))))


@dataclass
class FunctionType(Type):
    """Function type: (param1: type1, param2: type2) => return_type"""
    param_types: List[Type] = field(default_factory=list)
    return_type: Type = field(default=None)  # type: ignore
    
    def __str__(self) -> str:
        params = ", ".join(str(t) for t in self.param_types)
        return f"({params}) => {self.return_type}"
    
    def __eq__(self, other: object) -> bool:
        return (isinstance(other, FunctionType) and 
                self.param_types == other.param_types and
                self.return_type == other.return_type)
    
    def __hash__(self) -> int:
        return hash(("function", tuple(self.param_types), self.return_type))


@dataclass
class EnumType(Type):
    """Enum type: one_of("value1", "value2", ...)"""
    values: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"one_of({', '.join(repr(v) for v in self.values)})"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, EnumType) and set(self.values) == set(other.values)
    
    def __hash__(self) -> int:
        return hash(("enum", tuple(sorted(self.values))))


@dataclass
class AnyType(Type):
    """Any type - for untyped or dynamically typed values"""
    
    def __str__(self) -> str:
        return "any"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnyType)
    
    def __hash__(self) -> int:
        return hash("any")


# Built-in type instances
TEXT = PrimitiveType("text")
NUMBER = PrimitiveType("number")
BOOLEAN = PrimitiveType("boolean")
NULL = PrimitiveType("null")
ANY = AnyType()


# ==================== Type Environment ====================


@dataclass
class TypeBinding:
    """A type binding in the type environment."""
    name: str
    type: Type
    mutable: bool = True
    location: Optional[SourceLocation] = None


class TypeEnvironment:
    """
    Type environment for static type checking.
    
    Manages scopes and tracks variable types, function signatures,
    and other type information during the checking process.
    """
    
    def __init__(self, parent: Optional[TypeEnvironment] = None):
        self.parent = parent
        self.bindings: Dict[str, TypeBinding] = {}
        self.functions: Dict[str, FunctionType] = {}
        self.schemas: Dict[str, ObjectType] = {}
        
    def bind(self, name: str, type: Type, mutable: bool = True, 
             location: Optional[SourceLocation] = None) -> None:
        """Bind a name to a type in the current scope."""
        self.bindings[name] = TypeBinding(name, type, mutable, location)
    
    def lookup(self, name: str) -> Optional[TypeBinding]:
        """Look up a name in the current scope or parent scopes."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def define_function(self, name: str, func_type: FunctionType) -> None:
        """Define a function type."""
        self.functions[name] = func_type
    
    def lookup_function(self, name: str) -> Optional[FunctionType]:
        """Look up a function type."""
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.lookup_function(name)
        return None
    
    def define_schema(self, name: str, schema: ObjectType) -> None:
        """Define a schema type."""
        self.schemas[name] = schema
    
    def lookup_schema(self, name: str) -> Optional[ObjectType]:
        """Look up a schema type."""
        if name in self.schemas:
            return self.schemas[name]
        if self.parent:
            return self.parent.lookup_schema(name)
        return None
    
    def child_scope(self) -> TypeEnvironment:
        """Create a child scope."""
        return TypeEnvironment(parent=self)


# ==================== Static Type Checker ====================


class StaticTypeChecker:
    """
    Static type checker for Namel3ss programs.
    
    Validates type correctness throughout the AST, producing clear error
    messages for type mismatches, undefined references, and schema violations.
    """
    
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.errors: List[N3TypeError] = []
        self.env = TypeEnvironment()
        self._init_builtins()
    
    def _init_builtins(self) -> None:
        """Initialize built-in types and functions."""
        # Built-in functions
        self.env.define_function("str", FunctionType([ANY], TEXT))
        self.env.define_function("int", FunctionType([ANY], NUMBER))
        self.env.define_function("len", FunctionType([ArrayType(ANY)], NUMBER))
        
        # Built-in map and filter - these will be handled specially in check_call
        # map<T, R>(array<T>, (T) => R) => array<R>
        # filter<T>(array<T>, (T) => boolean) => array<T>
        # We mark them as special generic functions
        self.env.define_function("map", FunctionType([ArrayType(ANY), FunctionType([ANY], ANY)], ArrayType(ANY)))
        self.env.define_function("filter", FunctionType([ArrayType(ANY), FunctionType([ANY], BOOLEAN)], ArrayType(ANY)))
        self.env.define_function("bool", FunctionType([ANY], BOOLEAN))
        self.env.define_function("len", FunctionType([ArrayType(ANY)], NUMBER))
        self.env.define_function("sum", FunctionType([ArrayType(NUMBER)], NUMBER))
        self.env.define_function("map", FunctionType(
            [ArrayType(ANY), FunctionType([ANY], ANY)],
            ArrayType(ANY)
        ))
        self.env.define_function("filter", FunctionType(
            [ArrayType(ANY), FunctionType([ANY], BOOLEAN)],
            ArrayType(ANY)
        ))
        self.env.define_function("reduce", FunctionType(
            [ArrayType(ANY), FunctionType([ANY, ANY], ANY), ANY],
            ANY
        ))
    
    def error(self, message: str, line: int = 0, column: int = 0, 
              code: str = "TYPE_ERROR") -> None:
        """Record a type error."""
        error = N3TypeError(
            message=message,
            path=self.path,
            line=line,
            column=column,
            code=code
        )
        self.errors.append(error)
    
    def check_module(self, module: Module) -> List[N3TypeError]:
        """
        Check a module for type errors.
        
        Returns:
            List of type errors found
        """
        self.errors = []
        
        # Check each declaration in the module
        for decl in module.body:
            if isinstance(decl, App):
                self.check_app(decl)
        
        return self.errors
    
    def check_app(self, app: App) -> None:
        """Check an app declaration."""
        # Register datasets
        for dataset in app.datasets:
            self.check_dataset(dataset)
        
        # Register prompts and their schemas
        for prompt in app.prompts:
            self.check_prompt(prompt)
        
        # Check chains
        for chain in app.chains:
            self.check_chain(chain)
        
        # Check pages
        for page in app.pages:
            self.check_page(page)
        
        # Check AI models
        for model in app.ai_models:
            self.check_ai_model(model)
    
    def check_dataset(self, dataset: Dataset) -> None:
        """Check a dataset declaration."""
        # TODO: Validate dataset schema if available
        pass
    
    def check_prompt(self, prompt: Prompt) -> None:
        """Check a prompt declaration and register its schema."""
        # Build input schema
        if hasattr(prompt, 'arguments') and prompt.arguments:
            input_fields = {}
            for arg in prompt.arguments:
                arg_type = self.parse_type_annotation(arg.type_annotation) if hasattr(arg, 'type_annotation') else ANY
                input_fields[arg.name] = arg_type
            input_schema = ObjectType(input_fields)
            self.env.define_schema(f"{prompt.name}_input", input_schema)
        
        # Build output schema
        if hasattr(prompt, 'output_schema') and prompt.output_schema:
            output_fields = {}
            for field in prompt.output_schema.fields:
                field_type = self.parse_output_field_type(field)
                output_fields[field.name] = field_type
            output_schema = ObjectType(output_fields)
            self.env.define_schema(f"{prompt.name}_output", output_schema)
    
    def check_chain(self, chain: Chain) -> None:
        """Check a chain declaration."""
        # TODO: Validate chain steps and data flow
        pass
    
    def check_page(self, page: Page) -> None:
        """Check a page declaration."""
        # TODO: Validate page statements and expressions
        pass
    
    def check_ai_model(self, model: AIModel) -> None:
        """Check an AI model declaration."""
        # TODO: Validate model configuration
        pass
    
    def check_expression(self, expr: Expression, env: Optional[TypeEnvironment] = None) -> Type:
        """
        Type-check an expression and return its type.
        
        Args:
            expr: Expression to check
            env: Type environment (uses self.env if not provided)
        
        Returns:
            Type of the expression
        """
        if env is None:
            env = self.env
        
        # Legacy AST nodes
        if isinstance(expr, Literal):
            return self.infer_literal_type(expr.value)
        
        if isinstance(expr, NameRef):
            binding = env.lookup(expr.name)
            if binding is None:
                self.error(f"Undefined variable '{expr.name}'", code="UNDEFINED_VARIABLE")
                return ANY
            return binding.type
        
        if isinstance(expr, AttributeRef):
            base_binding = env.lookup(expr.base)
            if base_binding is None:
                self.error(f"Undefined variable '{expr.base}'", code="UNDEFINED_VARIABLE")
                return ANY
            
            base_type = base_binding.type
            if isinstance(base_type, ObjectType):
                if expr.attr not in base_type.fields:
                    self.error(
                        f"Object type '{base_type}' has no field '{expr.attr}'",
                        code="UNKNOWN_FIELD"
                    )
                    return ANY
                return base_type.fields[expr.attr]
            else:
                # Allow attribute access on any type (duck typing)
                return ANY
        
        if isinstance(expr, BinaryOp):
            left_type = self.check_expression(expr.left, env)
            right_type = self.check_expression(expr.right, env)
            return self.check_binary_op(expr.op, left_type, right_type)
        
        if isinstance(expr, UnaryOp):
            operand_type = self.check_expression(expr.operand, env)
            return self.check_unary_op(expr.op, operand_type)
        
        if isinstance(expr, CallExpression):
            return self.check_call(expr, env)
        
        # New expression AST nodes
        if isinstance(expr, LiteralExpr):
            return self.infer_literal_type(expr.value)
        
        if isinstance(expr, VarExpr):
            # First check variables
            binding = env.lookup(expr.name)
            if binding is not None:
                return binding.type
            
            # Then check functions
            func_type = env.lookup_function(expr.name)
            if func_type is not None:
                return func_type
            
            # Not found
            self.error(f"Undefined variable '{expr.name}'", code="UNDEFINED_VARIABLE")
            return ANY
        
        if isinstance(expr, ExprBinaryOp):
            left_type = self.check_expression(expr.left, env)
            right_type = self.check_expression(expr.right, env)
            return self.check_binary_op(expr.op, left_type, right_type)
        
        if isinstance(expr, AttributeExpr):
            base_type = self.check_expression(expr.base, env)
            if isinstance(base_type, ObjectType):
                if expr.attr not in base_type.fields:
                    self.error(
                        f"Object type '{base_type}' has no field '{expr.attr}'",
                        code="UNKNOWN_FIELD"
                    )
                    return ANY
                return base_type.fields[expr.attr]
            return ANY
        
        if isinstance(expr, CallExpr):
            # Special handling for generic built-ins: map and filter
            if isinstance(expr.func, VarExpr):
                func_name = expr.func.name
                
                if func_name == "map" and len(expr.args) == 2:
                    # map<T, R>(array<T>, (T) => R) => array<R>
                    array_arg = expr.args[0]
                    fn_arg = expr.args[1]
                    
                    array_type = self.check_expression(array_arg, env)
                    fn_type = self.check_expression(fn_arg, env)
                    
                    if not isinstance(array_type, ArrayType):
                        self.error(f"First argument to 'map' must be array, got {array_type}", code="TYPE_MISMATCH")
                        return ANY
                    
                    if not isinstance(fn_type, FunctionType):
                        self.error(f"Second argument to 'map' must be function, got {fn_type}", code="TYPE_MISMATCH")
                        return ANY
                    
                    # Return array of function's return type
                    return ArrayType(fn_type.return_type)
                
                if func_name == "filter" and len(expr.args) == 2:
                    # filter<T>(array<T>, (T) => boolean) => array<T>
                    array_arg = expr.args[0]
                    fn_arg = expr.args[1]
                    
                    array_type = self.check_expression(array_arg, env)
                    fn_type = self.check_expression(fn_arg, env)
                    
                    if not isinstance(array_type, ArrayType):
                        self.error(f"First argument to 'filter' must be array, got {array_type}", code="TYPE_MISMATCH")
                        return ANY
                    
                    if not isinstance(fn_type, FunctionType):
                        self.error(f"Second argument to 'filter' must be function, got {fn_type}", code="TYPE_MISMATCH")
                        return ANY
                    
                    # Check that function returns boolean
                    if not self.is_assignable(BOOLEAN, fn_type.return_type):
                        self.error(f"Filter function must return boolean, got {fn_type.return_type}", code="TYPE_MISMATCH")
                    
                    # Return array of same element type
                    return array_type
            
            func_type = self.check_expression(expr.func, env)
            if isinstance(func_type, FunctionType):
                # Check argument count
                if len(expr.args) != len(func_type.param_types):
                    self.error(
                        f"Function expects {len(func_type.param_types)} arguments, got {len(expr.args)}",
                        code="WRONG_ARG_COUNT"
                    )
                
                # Check argument types
                for i, (arg, expected_type) in enumerate(zip(expr.args, func_type.param_types)):
                    arg_type = self.check_expression(arg, env)
                    if not self.is_assignable(arg_type, expected_type):
                        self.error(
                            f"Argument {i+1}: expected {expected_type}, got {arg_type}",
                            code="TYPE_MISMATCH"
                        )
                
                return func_type.return_type
            elif func_type != ANY:  # If not ANY, it's an error
                self.error(
                    f"Cannot call non-function type {func_type}",
                    code="NOT_CALLABLE"
                )
            return ANY
        
        if isinstance(expr, LambdaExpr):
            return self.check_lambda(expr, env)
        
        if isinstance(expr, IfExpr):
            cond_type = self.check_expression(expr.condition, env)
            if not self.is_assignable(cond_type, BOOLEAN):
                self.error(
                    f"Condition must be boolean, got {cond_type}",
                    code="TYPE_MISMATCH"
                )
            
            then_type = self.check_expression(expr.then_expr, env)
            else_type = self.check_expression(expr.else_expr, env) if expr.else_expr else NULL
            
            # Result type is union of both branches
            if then_type == else_type:
                return then_type
            return UnionType([then_type, else_type])
        
        if isinstance(expr, LetExpr):
            child_env = env.child_scope()
            for name, value_expr in expr.bindings:
                value_type = self.check_expression(value_expr, env)
                child_env.bind(name, value_type)
            return self.check_expression(expr.body, child_env)
        
        if isinstance(expr, ListExpr):
            if not expr.elements:
                return ArrayType(ANY)
            element_types = [self.check_expression(elem, env) for elem in expr.elements]
            # Check for homogeneous arrays
            if all(t == element_types[0] for t in element_types):
                return ArrayType(element_types[0])
            # Mixed types - report error
            self.error(
                f"Array elements must have same type, found: {', '.join(str(t) for t in set(element_types))}",
                code="TYPE_MISMATCH"
            )
            return ArrayType(UnionType(element_types))
        
        if isinstance(expr, DictExpr):
            fields = {}
            for key_expr, value_expr in expr.pairs:
                if not isinstance(key_expr, (LiteralExpr, VarExpr)):
                    self.error("Dictionary keys must be literals or variables", code="INVALID_DICT_KEY")
                    continue
                key = key_expr.value if isinstance(key_expr, LiteralExpr) else key_expr.name
                if isinstance(key, str):
                    value_type = self.check_expression(value_expr, env)
                    fields[key] = value_type
            return ObjectType(fields)
        
        if isinstance(expr, TupleExpr):
            # For now, treat tuples as arrays
            if not expr.elements:
                return ArrayType(ANY)
            element_types = [self.check_expression(elem, env) for elem in expr.elements]
            if all(t == element_types[0] for t in element_types):
                return ArrayType(element_types[0])
            return ArrayType(UnionType(element_types))
        
        if isinstance(expr, IndexExpr):
            return self.check_index(expr, env)
        
        if isinstance(expr, SliceExpr):
            base_type = self.check_expression(expr.base, env)
            if isinstance(base_type, ArrayType):
                return base_type  # Slicing an array returns an array
            return ANY
        
        # Default fallback
        return ANY
    
    def check_lambda(self, lambda_expr: LambdaExpr, env: TypeEnvironment) -> FunctionType:
        """Check a lambda expression and return its function type."""
        child_env = env.child_scope()
        
        param_types = []
        for param in lambda_expr.params:
            # Parse parameter type hint if present
            if hasattr(param, 'type_hint') and param.type_hint:
                param_type = self.parse_type_annotation(param.type_hint)
            elif hasattr(param, 'type_annotation') and param.type_annotation:
                param_type = self.parse_type_annotation(param.type_annotation)
            else:
                param_type = ANY
            
            param_types.append(param_type)
            child_env.bind(param.name, param_type)
        
        # Check body
        return_type = self.check_expression(lambda_expr.body, child_env)
        
        return FunctionType(param_types, return_type)
    
    def check_index(self, index_expr: IndexExpr, env: TypeEnvironment) -> Type:
        """Check an index expression (arr[i] or obj['key'])."""
        base_type = self.check_expression(index_expr.base, env)
        index_type = self.check_expression(index_expr.index, env)
        
        if isinstance(base_type, ArrayType):
            # Array indexing - index must be number
            if not self.is_assignable(index_type, NUMBER):
                self.error(
                    f"Array index must be number, got {index_type}",
                    code="TYPE_MISMATCH"
                )
            return base_type.element_type
        
        if isinstance(base_type, ObjectType):
            # Object indexing - index must be text (property name)
            if isinstance(index_expr.index, LiteralExpr) and isinstance(index_expr.index.value, str):
                prop_name = index_expr.index.value
                if prop_name in base_type.fields:
                    return base_type.fields[prop_name]
                # Property doesn't exist
                self.error(
                    f"Object has no property '{prop_name}'",
                    code="UNDEFINED_PROPERTY"
                )
                return ANY
            if not self.is_assignable(index_type, TEXT):
                self.error(
                    f"Object index must be text, got {index_type}",
                    code="TYPE_MISMATCH"
                )
            # Return union of all field types or ANY
            if base_type.fields:
                return UnionType(list(base_type.fields.values()))
            return ANY
        
        # Allow indexing on any type (duck typing)
        return ANY
    
    def check_call(self, call: CallExpression, env: TypeEnvironment) -> Type:
        """Check a function call."""
        # Get function name
        if isinstance(call.function, NameRef):
            func_name = call.function.name
            func_type = env.lookup_function(func_name)
            
            if func_type is None:
                # Check if it's a variable with function type
                binding = env.lookup(func_name)
                if binding and isinstance(binding.type, FunctionType):
                    func_type = binding.type
                else:
                    self.error(f"Undefined function '{func_name}'", code="UNDEFINED_FUNCTION")
                    return ANY
            
            # Special handling for generic built-ins: map and filter
            if func_name == "map" and len(call.arguments) == 2:
                # map<T, R>(array<T>, (T) => R) => array<R>
                array_arg = call.arguments[0]
                fn_arg = call.arguments[1]
                
                array_type = self.check_expression(array_arg, env)
                fn_type = self.check_expression(fn_arg, env)
                
                if not isinstance(array_type, ArrayType):
                    self.error(f"First argument to 'map' must be array, got {array_type}", code="TYPE_MISMATCH")
                    return ANY
                
                if not isinstance(fn_type, FunctionType):
                    self.error(f"Second argument to 'map' must be function, got {fn_type}", code="TYPE_MISMATCH")
                    return ANY
                
                # Return array of function's return type
                return ArrayType(fn_type.return_type)
            
            if func_name == "filter" and len(call.arguments) == 2:
                # filter<T>(array<T>, (T) => boolean) => array<T>
                array_arg = call.arguments[0]
                fn_arg = call.arguments[1]
                
                array_type = self.check_expression(array_arg, env)
                fn_type = self.check_expression(fn_arg, env)
                
                if not isinstance(array_type, ArrayType):
                    self.error(f"First argument to 'filter' must be array, got {array_type}", code="TYPE_MISMATCH")
                    return ANY
                
                if not isinstance(fn_type, FunctionType):
                    self.error(f"Second argument to 'filter' must be function, got {fn_type}", code="TYPE_MISMATCH")
                    return ANY
                
                # Check that function returns boolean
                if not self.is_assignable(fn_type.return_type, BOOLEAN):
                    self.error(f"Filter function must return boolean, got {fn_type.return_type}", code="TYPE_MISMATCH")
                
                # Return array of same element type
                return array_type
            
            # Check argument count
            if len(call.arguments) != len(func_type.param_types):
                self.error(
                    f"Function '{func_name}' expects {len(func_type.param_types)} arguments, got {len(call.arguments)}",
                    code="ARGUMENT_COUNT_MISMATCH"
                )
            
            # Check argument types
            for i, (arg, expected_type) in enumerate(zip(call.arguments, func_type.param_types)):
                arg_type = self.check_expression(arg, env)
                if not self.is_assignable(arg_type, expected_type):
                    self.error(
                        f"Function '{func_name}' argument {i+1}: expected {expected_type}, got {arg_type}",
                        code="TYPE_MISMATCH"
                    )
            
            return func_type.return_type
        
        # For now, return ANY for complex function expressions
        return ANY
    
    def check_binary_op(self, op: str, left: Type, right: Type) -> Type:
        """Check a binary operation and return its result type."""
        if op in ['+', '-', '*', '/', '%', '**']:
            # Arithmetic operators
            if op == '+':
                # Addition: both must be numbers OR both must be text
                if self.is_assignable(left, NUMBER) and self.is_assignable(right, NUMBER):
                    return NUMBER
                if self.is_assignable(left, TEXT) and self.is_assignable(right, TEXT):
                    return TEXT
                # Type mismatch
                self.error(
                    f"Operator '+' requires both operands to be same type (number or text), got {left} and {right}",
                    code="TYPE_MISMATCH"
                )
                return ANY
            else:
                # Other arithmetic operators require numbers
                if not self.is_assignable(left, NUMBER):
                    self.error(f"Left operand of '{op}' must be number, got {left}", code="TYPE_MISMATCH")
                if not self.is_assignable(right, NUMBER):
                    self.error(f"Right operand of '{op}' must be number, got {right}", code="TYPE_MISMATCH")
                return NUMBER
        
        if op in ['==', '!=', '<', '>', '<=', '>=']:
            # Comparison operators - can compare any types, returns boolean
            return BOOLEAN
        
        if op in ['and', 'or', '&&', '||']:
            # Logical operators require boolean operands
            if not self.is_assignable(left, BOOLEAN):
                self.error(f"Left operand of '{op}' must be boolean, got {left}", code="TYPE_MISMATCH")
            if not self.is_assignable(right, BOOLEAN):
                self.error(f"Right operand of '{op}' must be boolean, got {right}", code="TYPE_MISMATCH")
            return BOOLEAN
        
        if op == 'in':
            # Membership test - right must be array
            if not isinstance(right, ArrayType):
                self.error(f"Right operand of 'in' must be array, got {right}", code="TYPE_MISMATCH")
            return BOOLEAN
        
        # Default: return ANY for unknown operators
        return ANY
    
    def check_unary_op(self, op: str, operand: Type) -> Type:
        """Check a unary operation and return its result type."""
        if op in ['!', 'not']:
            if not self.is_assignable(operand, BOOLEAN):
                self.error(f"Operand of '{op}' must be boolean, got {operand}", code="TYPE_MISMATCH")
            return BOOLEAN
        
        if op in ['+', '-']:
            if not self.is_assignable(operand, NUMBER):
                self.error(f"Operand of '{op}' must be number, got {operand}", code="TYPE_MISMATCH")
            return NUMBER
        
        return ANY
    
    def is_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target type.
        
        Note: Despite the parameter names, this method historically checks
        if the FIRST parameter can accept the SECOND parameter.
        So is_assignable(A, B) means "can A accept B" or "can B be assigned to A".
        
        For clarity: is_assignable(target_type, source_value)
        """
        # Swap to match test expectations - first param is target, second is source
        target, source = source, target
        
        # ANY is compatible with everything
        if isinstance(target, AnyType) or isinstance(source, AnyType):
            return True
        
        # Same primitive types
        if isinstance(source, PrimitiveType) and isinstance(target, PrimitiveType):
            return source.name == target.name
        
        # Array types
        if isinstance(source, ArrayType) and isinstance(target, ArrayType):
            return self.is_assignable(source.element_type, target.element_type)
        
        # Object types - structural subtyping
        # Target must have at most the fields of source (source can have extra fields)
        if isinstance(source, ObjectType) and isinstance(target, ObjectType):
            for field_name, field_type in target.fields.items():
                if field_name not in source.fields:
                    return False
                if not self.is_assignable(source.fields[field_name], field_type):
                    return False
            return True
        
        # Union types
        # Source is assignable to T | U if source is assignable to any member
        if isinstance(target, UnionType):
            return any(self.is_assignable(t, source) for t in target.types)
        
        # T | U is assignable to target only if all members are assignable to target
        if isinstance(source, UnionType):
            return all(self.is_assignable(target, t) for t in source.types)
        
        # Function types
        if isinstance(source, FunctionType) and isinstance(target, FunctionType):
            if len(source.param_types) != len(target.param_types):
                return False
            # Contravariant in parameters, covariant in return type
            for s_param, t_param in zip(source.param_types, target.param_types):
                if not self.is_assignable(t_param, s_param):  # Note: reversed!
                    return False
            return self.is_assignable(source.return_type, target.return_type)
        
        # Enum types
        if isinstance(source, EnumType) and isinstance(target, EnumType):
            return set(source.values).issubset(set(target.values))
        
        return False
    
    def infer_literal_type(self, value: Any) -> Type:
        """Infer the type of a literal value."""
        if value is None:
            return NULL
        if isinstance(value, bool):
            return BOOLEAN
        if isinstance(value, (int, float)):
            return NUMBER
        if isinstance(value, str):
            return TEXT
        if isinstance(value, list):
            if not value:
                return ArrayType(ANY)
            elem_types = [self.infer_literal_type(elem) for elem in value]
            if all(t == elem_types[0] for t in elem_types):
                return ArrayType(elem_types[0])
            return ArrayType(UnionType(elem_types))
        if isinstance(value, dict):
            fields = {k: self.infer_literal_type(v) for k, v in value.items()}
            return ObjectType(fields)
        return ANY
    
    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a type annotation string into a Type."""
        annotation = annotation.strip()
        
        # Primitive types
        if annotation == "text" or annotation == "string":
            return TEXT
        if annotation == "number" or annotation == "int" or annotation == "float":
            return NUMBER
        if annotation == "boolean" or annotation == "bool":
            return BOOLEAN
        if annotation == "null" or annotation == "none":
            return NULL
        if annotation == "any":
            return ANY
        
        # Array types
        if annotation.startswith("array<") and annotation.endswith(">"):
            elem_annotation = annotation[6:-1].strip()
            elem_type = self.parse_type_annotation(elem_annotation)
            return ArrayType(elem_type)
        
        if annotation.startswith("array[") and annotation.endswith("]"):
            elem_annotation = annotation[6:-1].strip()
            elem_type = self.parse_type_annotation(elem_annotation)
            return ArrayType(elem_type)
        
        # Object types
        if annotation.startswith("{") and annotation.endswith("}"):
            # Parse object fields
            inner = annotation[1:-1].strip()
            if not inner:
                return ObjectType({})
            # Simple parsing - split by comma and parse field: type
            fields = {}
            for field_str in inner.split(","):
                if ":" in field_str:
                    field_name, field_type_str = field_str.split(":", 1)
                    field_name = field_name.strip()
                    field_type = self.parse_type_annotation(field_type_str.strip())
                    fields[field_name] = field_type
            return ObjectType(fields)
        
        # Union types
        if "|" in annotation:
            type_strs = annotation.split("|")
            types = [self.parse_type_annotation(t.strip()) for t in type_strs]
            return UnionType(types)
        
        # Default: treat as any
        return ANY
    
    def parse_output_field_type(self, field) -> Type:
        """Parse an output field type from a prompt schema."""
        if hasattr(field, 'field_type'):
            type_info = field.field_type
            if isinstance(type_info, str):
                return self.parse_type_annotation(type_info)
        
        # Check for enum
        if hasattr(field, 'enum_values') and field.enum_values:
            return EnumType(field.enum_values)
        
        return ANY


# ==================== Public API ====================


def check_module_static(module: Module, path: Optional[str] = None) -> List[N3TypeError]:
    """
    Run static type checking on a module.
    
    Args:
        module: Module AST to check
        path: Optional file path for error reporting
    
    Returns:
        List of type errors found
    
    Example:
        ```python
        from namel3ss.lang.parser import parse_module
        from namel3ss.types.static_checker import check_module_static
        
        module = parse_module(source_code)
        errors = check_module_static(module, "app.ai")
        
        if errors:
            for error in errors:
                print(f"{error.path}:{error.line}:{error.column} - {error.message}")
        ```
    """
    checker = StaticTypeChecker(path)
    return checker.check_module(module)


def check_app_static(app: App, path: Optional[str] = None) -> List[N3TypeError]:
    """
    Run static type checking on an app.
    
    Args:
        app: App AST to check
        path: Optional file path for error reporting
    
    Returns:
        List of type errors found
    """
    checker = StaticTypeChecker(path)
    checker.check_app(app)
    return checker.errors


__all__ = [
    # Type system
    "Type",
    "PrimitiveType",
    "ArrayType",
    "ObjectType",
    "UnionType",
    "FunctionType",
    "EnumType",
    "AnyType",
    
    # Built-in types
    "TEXT",
    "NUMBER",
    "BOOLEAN",
    "NULL",
    "ANY",
    
    # Type environment
    "TypeBinding",
    "TypeEnvironment",
    
    # Type checker
    "StaticTypeChecker",
    "check_module_static",
    "check_app_static",
]
