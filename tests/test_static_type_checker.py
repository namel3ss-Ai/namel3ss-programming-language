"""
Tests for the static type checker.

Tests type validation, inference, subtyping, and error reporting.
"""

import pytest
from namel3ss.types.static_checker import (
    StaticTypeChecker,
    TypeEnvironment,
    PrimitiveType,
    ArrayType,
    ObjectType,
    UnionType,
    FunctionType,
    EnumType,
    AnyType,
    TEXT,
    NUMBER,
    BOOLEAN,
    NULL,
    ANY,
)
from namel3ss.ast.expressions import (
    LiteralExpr,
    VarExpr,
    BinaryOp,
    CallExpr,
    LambdaExpr,
    IndexExpr,
    ListExpr,
    DictExpr,
    Parameter,
)


class TestPrimitiveTypes:
    """Test primitive type operations."""
    
    def test_primitive_type_equality(self):
        assert TEXT == PrimitiveType("text")
        assert NUMBER == PrimitiveType("number")
        assert BOOLEAN == PrimitiveType("boolean")
        assert NULL == PrimitiveType("null")
    
    def test_primitive_type_string_representation(self):
        assert str(TEXT) == "text"
        assert str(NUMBER) == "number"
        assert str(BOOLEAN) == "boolean"
        assert str(NULL) == "null"


class TestArrayTypes:
    """Test array type operations."""
    
    def test_array_type_creation(self):
        arr_type = ArrayType(NUMBER)
        assert arr_type.element_type == NUMBER
        assert str(arr_type) == "array<number>"
    
    def test_array_type_equality(self):
        arr1 = ArrayType(NUMBER)
        arr2 = ArrayType(NUMBER)
        arr3 = ArrayType(TEXT)
        
        assert arr1 == arr2
        assert arr1 != arr3
    
    def test_nested_array_types(self):
        nested = ArrayType(ArrayType(NUMBER))
        assert str(nested) == "array<array<number>>"


class TestObjectTypes:
    """Test object type operations."""
    
    def test_object_type_creation(self):
        obj_type = ObjectType({"name": TEXT, "age": NUMBER})
        assert obj_type.fields["name"] == TEXT
        assert obj_type.fields["age"] == NUMBER
        assert str(obj_type) == "{name: text, age: number}"
    
    def test_object_type_equality(self):
        obj1 = ObjectType({"name": TEXT, "age": NUMBER})
        obj2 = ObjectType({"name": TEXT, "age": NUMBER})
        obj3 = ObjectType({"name": TEXT})
        
        assert obj1 == obj2
        assert obj1 != obj3


class TestFunctionTypes:
    """Test function type operations."""
    
    def test_function_type_creation(self):
        fn_type = FunctionType([NUMBER, NUMBER], NUMBER)
        assert len(fn_type.param_types) == 2
        assert fn_type.return_type == NUMBER
        assert str(fn_type) == "(number, number) => number"
    
    def test_function_type_no_params(self):
        fn_type = FunctionType([], TEXT)
        assert str(fn_type) == "() => text"


class TestUnionTypes:
    """Test union type operations."""
    
    def test_union_type_creation(self):
        union = UnionType([TEXT, NUMBER, NULL])
        assert TEXT in union.types
        assert NUMBER in union.types
        assert NULL in union.types
        assert str(union) == "text | number | null"
    
    def test_union_type_equality(self):
        union1 = UnionType([TEXT, NUMBER])
        union2 = UnionType([NUMBER, TEXT])  # Order shouldn't matter
        assert union1 == union2


class TestEnumTypes:
    """Test enum type operations."""
    
    def test_enum_type_creation(self):
        enum = EnumType(["admin", "user", "guest"])
        assert "admin" in enum.values
        assert "user" in enum.values
        assert str(enum) == "one_of('admin', 'user', 'guest')"


class TestTypeEnvironment:
    """Test type environment scoping and symbol management."""
    
    def test_variable_binding(self):
        env = TypeEnvironment()
        env.bind("x", NUMBER)
        
        binding = env.lookup("x")
        assert binding is not None
        assert binding.type == NUMBER
    
    def test_variable_lookup_not_found(self):
        env = TypeEnvironment()
        assert env.lookup("unknown") is None
    
    def test_nested_scopes(self):
        env = TypeEnvironment()
        env.bind("x", NUMBER)
        
        # Create child scope
        child_env = env.child_scope()
        child_env.bind("y", TEXT)
        
        assert child_env.lookup("x").type == NUMBER  # Outer scope
        assert child_env.lookup("y").type == TEXT    # Inner scope
        
        # Back to parent scope
        assert env.lookup("y") is None    # Not in parent scope
        assert env.lookup("x").type == NUMBER  # Still accessible
    
    def test_variable_shadowing(self):
        env = TypeEnvironment()
        env.bind("x", NUMBER)
        
        # Create child scope and shadow
        child_env = env.child_scope()
        child_env.bind("x", TEXT)  # Shadow outer x
        
        assert child_env.lookup("x").type == TEXT
        
        # Back to parent
        assert env.lookup("x").type == NUMBER
    
    def test_function_binding(self):
        env = TypeEnvironment()
        fn_type = FunctionType([NUMBER, NUMBER], NUMBER)
        env.define_function("add", fn_type)
        
        assert env.lookup_function("add") == fn_type


class TestTypeChecker:
    """Test the main type checker functionality."""
    
    def setup_method(self):
        """Set up a fresh type checker for each test."""
        self.checker = StaticTypeChecker()
    
    def test_literal_type_inference(self):
        """Test that literals are correctly typed."""
        # Text literal
        text_lit = LiteralExpr(value="hello")
        assert self.checker.check_expression(text_lit) == TEXT
        
        # Number literal
        num_lit = LiteralExpr(value=42)
        assert self.checker.check_expression(num_lit) == NUMBER
        
        # Boolean literal
        bool_lit = LiteralExpr(value=True)
        assert self.checker.check_expression(bool_lit) == BOOLEAN
        
        # Null literal
        null_lit = LiteralExpr(value=None)
        assert self.checker.check_expression(null_lit) == NULL
    
    def test_variable_reference(self):
        """Test variable reference type checking."""
        self.checker.env.bind("x", NUMBER)
        
        var_expr = VarExpr(name="x")
        assert self.checker.check_expression(var_expr) == NUMBER
    
    def test_undefined_variable(self):
        """Test error on undefined variable reference."""
        var_expr = VarExpr(name="undefined")
        
        result_type = self.checker.check_expression(var_expr)
        assert result_type == ANY
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "UNDEFINED_VARIABLE"
    
    def test_binary_op_number_addition(self):
        """Test type checking for numeric addition."""
        left = LiteralExpr(value=10)
        right = LiteralExpr(value=20)
        expr = BinaryOp(op="+", left=left, right=right)
        
        assert self.checker.check_expression(expr) == NUMBER
        assert len(self.checker.errors) == 0
    
    def test_binary_op_string_concatenation(self):
        """Test type checking for string concatenation."""
        left = LiteralExpr(value="Hello ")
        right = LiteralExpr(value="World")
        expr = BinaryOp(op="+", left=left, right=right)
        
        assert self.checker.check_expression(expr) == TEXT
        assert len(self.checker.errors) == 0
    
    def test_binary_op_type_mismatch(self):
        """Test error on incompatible operand types."""
        left = LiteralExpr(value=10)
        right = LiteralExpr(value="hello")
        expr = BinaryOp(op="+", left=left, right=right)
        
        self.checker.check_expression(expr)
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "TYPE_MISMATCH"
    
    def test_comparison_operators(self):
        """Test type checking for comparison operators."""
        left = LiteralExpr(value=10)
        right = LiteralExpr(value=20)
        
        for op in ["==", "!=", "<", ">", "<=", ">="]:
            self.checker = StaticTypeChecker()  # Reset
            expr = BinaryOp(op=op, left=left, right=right)
            assert self.checker.check_expression(expr) == BOOLEAN
    
    def test_logical_operators(self):
        """Test type checking for logical operators."""
        left = LiteralExpr(value=True)
        right = LiteralExpr(value=False)
        
        for op in ["and", "or"]:
            self.checker = StaticTypeChecker()  # Reset
            expr = BinaryOp(op=op, left=left, right=right)
            assert self.checker.check_expression(expr) == BOOLEAN
    
    def test_array_literal(self):
        """Test type inference for array literals."""
        elements = [
            LiteralExpr(value=1),
            LiteralExpr(value=2),
            LiteralExpr(value=3)
        ]
        arr_expr = ListExpr(elements=elements)
        
        result_type = self.checker.check_expression(arr_expr)
        assert isinstance(result_type, ArrayType)
        assert result_type.element_type == NUMBER
    
    def test_array_literal_mixed_types(self):
        """Test error on array with mixed element types."""
        elements = [
            LiteralExpr(value=1),
            LiteralExpr(value="two"),
            LiteralExpr(value=3)
        ]
        arr_expr = ListExpr(elements=elements)
        
        self.checker.check_expression(arr_expr)
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "TYPE_MISMATCH"
    
    def test_array_indexing(self):
        """Test type checking for array indexing."""
        # Setup: array of numbers
        self.checker.env.bind("numbers", ArrayType(NUMBER))
        
        base = VarExpr(name="numbers")
        index = LiteralExpr(value=0)
        index_expr = IndexExpr(base=base, index=index)
        
        result_type = self.checker.check_expression(index_expr)
        assert result_type == NUMBER
    
    def test_array_indexing_non_number_index(self):
        """Test error on non-numeric array index."""
        self.checker.env.bind("numbers", ArrayType(NUMBER))
        
        base = VarExpr(name="numbers")
        index = LiteralExpr(value="zero")  # String index on array
        index_expr = IndexExpr(base=base, index=index)
        
        self.checker.check_expression(index_expr)
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "TYPE_MISMATCH"
    
    def test_object_property_access(self):
        """Test type checking for object property access."""
        obj_type = ObjectType({"name": TEXT, "age": NUMBER})
        self.checker.env.bind("user", obj_type)
        
        base = VarExpr(name="user")
        index = LiteralExpr(value="name")
        index_expr = IndexExpr(base=base, index=index)
        
        result_type = self.checker.check_expression(index_expr)
        assert result_type == TEXT
    
    def test_object_undefined_property(self):
        """Test error on accessing undefined object property."""
        obj_type = ObjectType({"name": TEXT})
        self.checker.env.bind("user", obj_type)
        
        base = VarExpr(name="user")
        index = LiteralExpr(value="age")  # Undefined property
        index_expr = IndexExpr(base=base, index=index)
        
        self.checker.check_expression(index_expr)
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "UNDEFINED_PROPERTY"
    
    def test_lambda_expression(self):
        """Test type checking for lambda expressions."""
        params = [Parameter(name="x", type_hint="number")]
        body = BinaryOp(
            op="*",
            left=VarExpr(name="x"),
            right=LiteralExpr(value=2)
        )
        lambda_expr = LambdaExpr(params=params, body=body)
        
        result_type = self.checker.check_expression(lambda_expr)
        assert isinstance(result_type, FunctionType)
        assert len(result_type.param_types) == 1
        assert result_type.param_types[0] == NUMBER
        assert result_type.return_type == NUMBER
    
    def test_lambda_without_type_annotations(self):
        """Test lambda with inferred parameter types."""
        params = [Parameter(name="x")]
        body = VarExpr(name="x")
        lambda_expr = LambdaExpr(params=params, body=body)
        
        result_type = self.checker.check_expression(lambda_expr)
        assert isinstance(result_type, FunctionType)
        # Without annotations, parameters default to ANY
        assert result_type.param_types[0] == ANY
    
    def test_function_call(self):
        """Test type checking for function calls."""
        # Register a function
        fn_type = FunctionType([NUMBER, NUMBER], NUMBER)
        self.checker.env.define_function("add", fn_type)
        
        func = VarExpr(name="add")
        args = [LiteralExpr(value=10), LiteralExpr(value=20)]
        call_expr = CallExpr(func=func, args=args)
        
        result_type = self.checker.check_expression(call_expr)
        assert result_type == NUMBER
    
    def test_function_call_wrong_arg_types(self):
        """Test error on function call with wrong argument types."""
        fn_type = FunctionType([NUMBER, NUMBER], NUMBER)
        self.checker.env.define_function("add", fn_type)
        
        func = VarExpr(name="add")
        args = [LiteralExpr(value=10), LiteralExpr(value="twenty")]  # Wrong type
        call_expr = CallExpr(func=func, args=args)
        
        self.checker.check_expression(call_expr)
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "TYPE_MISMATCH"
    
    def test_function_call_wrong_arg_count(self):
        """Test error on function call with wrong number of arguments."""
        fn_type = FunctionType([NUMBER, NUMBER], NUMBER)
        self.checker.env.define_function("add", fn_type)
        
        func = VarExpr(name="add")
        args = [LiteralExpr(value=10)]  # Too few args
        call_expr = CallExpr(func=func, args=args)
        
        self.checker.check_expression(call_expr)
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "WRONG_ARG_COUNT"
    
    def test_calling_non_function(self):
        """Test error on calling a non-function value."""
        self.checker.env.bind("x", NUMBER)
        
        func = VarExpr(name="x")
        args = []
        call_expr = CallExpr(func=func, args=args)
        
        self.checker.check_expression(call_expr)
        assert len(self.checker.errors) == 1
        assert self.checker.errors[0].code == "NOT_CALLABLE"


class TestTypeAssignability:
    """Test type compatibility and subtyping rules."""
    
    def setup_method(self):
        self.checker = StaticTypeChecker()
    
    def test_identical_types_assignable(self):
        """Test that identical types are assignable."""
        assert self.checker.is_assignable(NUMBER, NUMBER)
        assert self.checker.is_assignable(TEXT, TEXT)
        assert self.checker.is_assignable(BOOLEAN, BOOLEAN)
    
    def test_any_type_accepts_all(self):
        """Test that ANY type accepts any value."""
        assert self.checker.is_assignable(ANY, NUMBER)
        assert self.checker.is_assignable(ANY, TEXT)
        assert self.checker.is_assignable(ANY, BOOLEAN)
    
    def test_any_type_assignable_to_all(self):
        """Test that ANY can be assigned to any type."""
        assert self.checker.is_assignable(NUMBER, ANY)
        assert self.checker.is_assignable(TEXT, ANY)
    
    def test_array_type_assignability(self):
        """Test array type assignability."""
        arr_num1 = ArrayType(NUMBER)
        arr_num2 = ArrayType(NUMBER)
        arr_text = ArrayType(TEXT)
        
        assert self.checker.is_assignable(arr_num1, arr_num2)
        assert not self.checker.is_assignable(arr_num1, arr_text)
    
    def test_object_type_structural_subtyping(self):
        """Test structural subtyping for object types."""
        # Subtype has extra fields
        supertype = ObjectType({"name": TEXT})
        subtype = ObjectType({"name": TEXT, "age": NUMBER})
        
        assert self.checker.is_assignable(supertype, subtype)
        assert not self.checker.is_assignable(subtype, supertype)
    
    def test_object_type_field_type_compatibility(self):
        """Test that field types must be compatible."""
        obj1 = ObjectType({"name": TEXT})
        obj2 = ObjectType({"name": NUMBER})
        
        assert not self.checker.is_assignable(obj1, obj2)
    
    def test_union_type_assignability(self):
        """Test union type assignability."""
        union = UnionType([TEXT, NUMBER])
        
        # Can assign members to union
        assert self.checker.is_assignable(union, TEXT)
        assert self.checker.is_assignable(union, NUMBER)
        
        # Cannot assign non-members
        assert not self.checker.is_assignable(union, BOOLEAN)
    
    def test_function_type_assignability(self):
        """Test function type assignability."""
        fn1 = FunctionType([NUMBER], TEXT)
        fn2 = FunctionType([NUMBER], TEXT)
        fn3 = FunctionType([TEXT], TEXT)
        
        assert self.checker.is_assignable(fn1, fn2)
        assert not self.checker.is_assignable(fn1, fn3)


class TestBuiltInFunctions:
    """Test type checking for built-in functions."""
    
    def setup_method(self):
        self.checker = StaticTypeChecker()
    
    def test_str_function(self):
        """Test str() function type checking."""
        func = VarExpr(name="str")
        args = [LiteralExpr(value=42)]
        call_expr = CallExpr(func=func, args=args)
        
        result_type = self.checker.check_expression(call_expr)
        assert result_type == TEXT
    
    def test_int_function(self):
        """Test int() function type checking."""
        func = VarExpr(name="int")
        args = [LiteralExpr(value="42")]
        call_expr = CallExpr(func=func, args=args)
        
        result_type = self.checker.check_expression(call_expr)
        assert result_type == NUMBER
    
    def test_len_function(self):
        """Test len() function type checking."""
        self.checker.env.bind("items", ArrayType(NUMBER))
        
        func = VarExpr(name="len")
        args = [VarExpr(name="items")]
        call_expr = CallExpr(func=func, args=args)
        
        result_type = self.checker.check_expression(call_expr)
        assert result_type == NUMBER
    
    def test_map_function(self):
        """Test map() function type checking."""
        # map(array<T>, (T) => R) => array<R>
        self.checker.env.bind("numbers", ArrayType(NUMBER))
        
        # Lambda: fn(x: number) => x * 2
        lambda_params = [Parameter(name="x", type_hint="number")]
        lambda_body = BinaryOp(
            op="*",
            left=VarExpr(name="x"),
            right=LiteralExpr(value=2)
        )
        lambda_expr = LambdaExpr(params=lambda_params, body=lambda_body)
        
        func = VarExpr(name="map")
        args = [VarExpr(name="numbers"), lambda_expr]
        call_expr = CallExpr(func=func, args=args)
        
        result_type = self.checker.check_expression(call_expr)
        assert isinstance(result_type, ArrayType)
        assert result_type.element_type == NUMBER
    
    def test_filter_function(self):
        """Test filter() function type checking."""
        # filter(array<T>, (T) => boolean) => array<T>
        self.checker.env.bind("numbers", ArrayType(NUMBER))
        
        # Lambda: fn(x: number) => x > 0
        lambda_params = [Parameter(name="x", type_hint="number")]
        lambda_body = BinaryOp(
            op=">",
            left=VarExpr(name="x"),
            right=LiteralExpr(value=0)
        )
        lambda_expr = LambdaExpr(params=lambda_params, body=lambda_body)
        
        func = VarExpr(name="filter")
        args = [VarExpr(name="numbers"), lambda_expr]
        call_expr = CallExpr(func=func, args=args)
        
        result_type = self.checker.check_expression(call_expr)
        assert isinstance(result_type, ArrayType)
        assert result_type.element_type == NUMBER


class TestErrorReporting:
    """Test error reporting and error messages."""
    
    def test_error_has_message(self):
        """Test that errors have descriptive messages."""
        checker = StaticTypeChecker()
        var_expr = VarExpr(name="undefined")
        
        checker.check_expression(var_expr)
        assert len(checker.errors) == 1
        assert "undefined" in checker.errors[0].message.lower()
    
    def test_error_has_code(self):
        """Test that errors have error codes."""
        checker = StaticTypeChecker()
        var_expr = VarExpr(name="undefined")
        
        checker.check_expression(var_expr)
        assert checker.errors[0].code is not None
    
    def test_multiple_errors_collected(self):
        """Test that multiple errors are collected."""
        checker = StaticTypeChecker()
        
        # Error 1: Undefined variable
        checker.check_expression(VarExpr(name="x"))
        
        # Error 2: Another undefined variable
        checker.check_expression(VarExpr(name="y"))
        
        assert len(checker.errors) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



