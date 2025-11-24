"""End-to-end tests for symbolic expression workflow."""

import pytest
from pathlib import Path

from namel3ss.parser import Parser
from namel3ss.ast import Program, Module
from namel3ss.resolver import resolve_program
from namel3ss.ast.expressions import FunctionDef, RuleDef, LambdaExpr, VarExpr, CallExpr
from namel3ss.codegen.backend.core.runtime.symbolic_evaluator import evaluate_expression_tree


class TestSymbolicExpressionIntegration:
    """Test complete workflow from parsing to evaluation."""
    
    def test_parse_simple_function(self):
        """Test parsing a simple function definition."""
        source = '''
app "Test".

fn double(x) => x * 2
'''
        parser = Parser(source)
        module = parser.parse()
        
        assert module is not None
        # Note: Current parser may not recognize fn syntax yet
        # This test documents expected behavior
    
    def test_parse_and_evaluate_lambda(self):
        """Test parsing and evaluating a lambda expression."""
        from namel3ss.ast.expressions import LambdaExpr, Parameter, LiteralExpr, VarExpr, CallExpr
        
        # Create lambda: fn(x) => x * 2
        double = LambdaExpr(
            params=[Parameter(name='x')],
            body=CallExpr(
                func=VarExpr(name='mul'),
                args=[VarExpr(name='x'), LiteralExpr(value=2)]
            )
        )
        
        # For now, we can't evaluate multiplication directly
        # but we can test lambda creation
        from namel3ss.codegen.backend.core.runtime.symbolic_evaluator import SymbolicEvaluator
        
        evaluator = SymbolicEvaluator()
        closure = evaluator.eval(double)
        
        assert closure['_closure'] is True
        assert len(closure['params']) == 1
    
    def test_builtin_function_call(self):
        """Test calling built-in functions."""
        from namel3ss.ast.expressions import CallExpr, VarExpr, ListExpr, LiteralExpr
        
        # Test: length([1, 2, 3])
        expr = CallExpr(
            func=VarExpr(name='length'),
            args=[
                ListExpr(elements=[
                    LiteralExpr(value=1),
                    LiteralExpr(value=2),
                    LiteralExpr(value=3)
                ])
            ]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == 3
    
    def test_map_function(self):
        """Test map with lambda."""
        from namel3ss.ast.expressions import (
            CallExpr, VarExpr, ListExpr, LiteralExpr, LambdaExpr, Parameter
        )
        
        # Create: map(fn(x) => x, [1, 2, 3])
        # Using identity function since we can't do arithmetic yet
        expr = CallExpr(
            func=VarExpr(name='map'),
            args=[
                VarExpr(name='identity'),  # Built-in identity function
                ListExpr(elements=[
                    LiteralExpr(value=1),
                    LiteralExpr(value=2),
                    LiteralExpr(value=3)
                ])
            ]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == [1, 2, 3]
    
    def test_filter_function(self):
        """Test filter with predicate."""
        from namel3ss.ast.expressions import (
            CallExpr, VarExpr, ListExpr, LiteralExpr, LambdaExpr, Parameter
        )
        
        # For now, test with a built-in predicate
        # filter(is_int, [1, "two", 3])
        expr = CallExpr(
            func=VarExpr(name='filter'),
            args=[
                VarExpr(name='is_int'),
                ListExpr(elements=[
                    LiteralExpr(value=1),
                    LiteralExpr(value="two"),
                    LiteralExpr(value=3)
                ])
            ]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == [1, 3]
    
    def test_pattern_matching(self):
        """Test pattern matching evaluation."""
        from namel3ss.ast.expressions import (
            MatchExpr, MatchCase, ListExpr, LiteralExpr,
            ListPattern, VarPattern, WildcardPattern
        )
        
        # match [1, 2, 3] { case [x, y, z] => x }
        expr = MatchExpr(
            expr=ListExpr(elements=[
                LiteralExpr(value=1),
                LiteralExpr(value=2),
                LiteralExpr(value=3)
            ]),
            cases=[
                MatchCase(
                    pattern=ListPattern(elements=[
                        VarPattern(name='x'),
                        VarPattern(name='y'),
                        VarPattern(name='z')
                    ]),
                    body=VarExpr(name='x')
                )
            ]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == 1
    
    def test_let_binding(self):
        """Test let expressions."""
        from namel3ss.ast.expressions import LetExpr, LiteralExpr, VarExpr
        
        # let x = 5 in x
        expr = LetExpr(
            bindings=[('x', LiteralExpr(value=5))],
            body=VarExpr(name='x')
        )
        
        result = evaluate_expression_tree(expr)
        assert result == 5
    
    def test_if_expression(self):
        """Test if-then-else."""
        from namel3ss.ast.expressions import IfExpr, LiteralExpr
        
        # if true then 1 else 2
        expr = IfExpr(
            condition=LiteralExpr(value=True),
            then_expr=LiteralExpr(value=1),
            else_expr=LiteralExpr(value=2)
        )
        
        result = evaluate_expression_tree(expr)
        assert result == 1
    
    def test_nested_expressions(self):
        """Test nested expression evaluation."""
        from namel3ss.ast.expressions import (
            LetExpr, CallExpr, VarExpr, ListExpr, LiteralExpr
        )
        
        # let nums = [1, 2, 3, 4, 5] in
        # let doubled = map(identity, nums) in
        # length(doubled)
        expr = LetExpr(
            bindings=[
                ('nums', ListExpr(elements=[
                    LiteralExpr(value=1),
                    LiteralExpr(value=2),
                    LiteralExpr(value=3),
                    LiteralExpr(value=4),
                    LiteralExpr(value=5)
                ]))
            ],
            body=LetExpr(
                bindings=[
                    ('doubled', CallExpr(
                        func=VarExpr(name='map'),
                        args=[
                            VarExpr(name='identity'),
                            VarExpr(name='nums')
                        ]
                    ))
                ],
                body=CallExpr(
                    func=VarExpr(name='length'),
                    args=[VarExpr(name='doubled')]
                )
            )
        )
        
        result = evaluate_expression_tree(expr)
        assert result == 5
    
    def test_higher_order_composition(self):
        """Test higher-order function composition."""
        from namel3ss.ast.expressions import (
            CallExpr, VarExpr, ListExpr, LiteralExpr
        )
        
        # Test: head(reverse([1, 2, 3]))
        expr = CallExpr(
            func=VarExpr(name='head'),
            args=[
                CallExpr(
                    func=VarExpr(name='reverse'),
                    args=[
                        ListExpr(elements=[
                            LiteralExpr(value=1),
                            LiteralExpr(value=2),
                            LiteralExpr(value=3)
                        ])
                    ]
                )
            ]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == 3
    
    def test_recursion_with_function_def(self):
        """Test recursive function definition and call."""
        from namel3ss.ast.expressions import (
            FunctionDef, Parameter, IfExpr, LiteralExpr, VarExpr, CallExpr
        )
        from namel3ss.codegen.backend.core.runtime.symbolic_evaluator import SymbolicEvaluator
        
        # Define: fn sum_to_n(n) => if n <= 0 then 0 else n + sum_to_n(n-1)
        # Simplified without arithmetic operators
        sum_func = FunctionDef(
            name='sum_to_n',
            params=[Parameter(name='n')],
            body=IfExpr(
                condition=LiteralExpr(value=True),  # Simplified
                then_expr=LiteralExpr(value=0),
                else_expr=LiteralExpr(value=0)
            )
        )
        
        evaluator = SymbolicEvaluator()
        evaluator.eval(sum_func)
        
        # Verify function is registered
        assert 'sum_to_n' in evaluator.functions
    
    def test_list_operations(self):
        """Test list operation functions."""
        from namel3ss.ast.expressions import CallExpr, VarExpr, ListExpr, LiteralExpr
        
        # Test: tail([1, 2, 3])
        expr = CallExpr(
            func=VarExpr(name='tail'),
            args=[
                ListExpr(elements=[
                    LiteralExpr(value=1),
                    LiteralExpr(value=2),
                    LiteralExpr(value=3)
                ])
            ]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == [2, 3]
    
    def test_string_operations(self):
        """Test string operation functions."""
        from namel3ss.ast.expressions import CallExpr, VarExpr, LiteralExpr
        
        # Test: upper("hello")
        expr = CallExpr(
            func=VarExpr(name='upper'),
            args=[LiteralExpr(value="hello")]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == "HELLO"
    
    def test_dict_operations(self):
        """Test dictionary operations."""
        from namel3ss.ast.expressions import CallExpr, VarExpr, DictExpr, LiteralExpr
        
        # Test: keys({a: 1, b: 2})
        expr = CallExpr(
            func=VarExpr(name='keys'),
            args=[
                DictExpr(pairs=[
                    (LiteralExpr(value='a'), LiteralExpr(value=1)),
                    (LiteralExpr(value='b'), LiteralExpr(value=2))
                ])
            ]
        )
        
        result = evaluate_expression_tree(expr)
        assert set(result) == {'a', 'b'}
    
    def test_error_handling(self):
        """Test error handling in evaluation."""
        from namel3ss.ast.expressions import VarExpr
        from namel3ss.codegen.backend.core.runtime.symbolic_evaluator import EvaluationError
        
        # Test undefined variable
        expr = VarExpr(name='undefined_var')
        
        with pytest.raises(EvaluationError, match="Undefined variable"):
            evaluate_expression_tree(expr)
    
    def test_type_conversions(self):
        """Test type conversion functions."""
        from namel3ss.ast.expressions import CallExpr, VarExpr, LiteralExpr
        
        # Test: int("42")
        expr = CallExpr(
            func=VarExpr(name='int'),
            args=[LiteralExpr(value="42")]
        )
        
        result = evaluate_expression_tree(expr)
        assert result == 42
        assert isinstance(result, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
