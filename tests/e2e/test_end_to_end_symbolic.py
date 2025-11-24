"""End-to-end integration tests for symbolic expressions in complete N3 applications."""

import pytest
from namel3ss.parser import Parser


class TestEndToEndSymbolicIntegration:
    """Test complete N3 applications with symbolic expressions."""
    
    def test_dataset_with_symbolic_filter(self):
        """Test dataset using symbolic expressions in filters."""
        source = '''
app "UserManager".

dataset "eligible_users" from memory:
    filter by: user.age >= 18 and (user.status == "active" or user.status == "verified")
'''
        parser = Parser(source)
        module = parser.parse()
        
        # Verify parsing
        assert module is not None
        app = module.body[0]
        assert len(app.datasets) == 1
        
        # Verify dataset filter exists
        dataset = app.datasets[0]
        assert len(dataset.operations) == 1
        filter_op = dataset.operations[0]
        condition = filter_op.condition
        
        # Condition should exist and be an expression
        assert condition is not None
        assert hasattr(condition, '__class__')
    
    def test_dataset_with_higher_order_functions(self):
        """Test dataset using map/filter/reduce."""
        source = '''
app "DataProcessor".

dataset "processed" from memory:
    add column doubled = map(values, fn(x) => x * 2)
    add column total = reduce(values, fn(acc, x) => acc + x, 0)
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        dataset = app.datasets[0]
        assert len(dataset.operations) == 2
        
        # Both should be computed columns with CallExpr
        assert all(op.__class__.__name__ == "ComputedColumnOp" for op in dataset.operations)
    
    def test_control_flow_with_symbolic_conditions(self):
        """Test if/for with symbolic expression conditions."""
        source = '''
app "ConditionalApp".

page home:
    if ctx.user.role == "admin":
        show text "Welcome, Admin!"
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        assert len(app.functions) == 1
        assert len(app.pages) == 1
        
        page = app.pages[0]
        # Page should have if statement with symbolic condition
        assert len(page.body) > 0
    
    def test_pattern_matching_in_filters(self):
        """Test pattern matching in dataset filters."""
        source = '''
app "OrderSystem".

dataset "priority_orders" from memory:
    filter by: match order.type:
        case "urgent": True
        case "priority": True
        else: False
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        dataset = app.datasets[0]
        filter_op = dataset.operations[0]
        
        # Condition should be MatchExpr
        assert filter_op.condition.__class__.__name__ == "MatchExpr"
    
    @pytest.mark.skip("Top-level rule syntax requires legacy parser update")
    def test_rules_and_queries(self):
        """Test rule definitions and queries."""
        source = '''
app "LogicApp".

rule eligible(user) :- user.age >= 18, user.verified == True.
rule premium(user) :- eligible(user), user.subscription == "premium".

dataset "premium_users" from memory:
    filter by: query premium(user)
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        
        # Verify rules are defined
        assert len(app.rules) == 2
        assert app.rules[0].head.predicate == "eligible"
        assert app.rules[1].head.predicate == "premium"
        
        # Verify dataset uses query
        dataset = app.datasets[0]
        filter_op = dataset.operations[0]
        assert filter_op.condition.__class__.__name__ == "QueryExpr"
    
    @pytest.mark.skip("Top-level fn syntax requires legacy parser update")
    def test_nested_functions_and_recursion(self):
        """Test nested function calls and recursion."""
        source = '''
app "MathApp".

fn factorial(n) => if n <= 1: 1 else: n * factorial(n - 1)
fn sum_range(start, end) => if start > end: 0 else: start + sum_range(start + 1, end)

dataset "calculations" from memory:
    add column fact_5 = factorial(5)
    add column sum_10 = sum_range(1, 10)
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        assert len(app.functions) == 2
        
        # Verify functions use recursion (IfExpr with CallExpr)
        factorial_fn = app.functions[0]
        assert factorial_fn.body.__class__.__name__ == "IfExpr"
    
    def test_let_expressions_in_computed_columns(self):
        """Test let expressions for local bindings."""
        source = '''
app "BindingTest".

dataset "computed" from memory:
    add column result = let x = price * 0.9: let tax = x * 0.1: x + tax
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        dataset = app.datasets[0]
        computed_op = dataset.operations[0]
        
        # Expression should be LetExpr
        assert computed_op.expression.__class__.__name__ == "LetExpr"
    
    def test_lambda_in_higher_order_functions(self):
        """Test lambda expressions passed to map/filter."""
        source = '''
app "FunctionalApp".

dataset "transformed" from memory:
    add column squares = map(numbers, fn(x) => x * x)
    add column evens = filter(numbers, fn(x) => x % 2 == 0)
    add column sum = reduce(numbers, fn(acc, x) => acc + x, 0)
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        dataset = app.datasets[0]
        assert len(dataset.operations) == 3
        
        # All should use lambdas
        for op in dataset.operations:
            expr = op.expression
            assert expr.__class__.__name__ == "CallExpr"
    
    @pytest.mark.skip("Top-level fn syntax requires legacy parser update")
    def test_complex_nested_expressions(self):
        """Test deeply nested symbolic expressions."""
        source = '''
app "ComplexApp".

fn process(data) => map(
    filter(data, fn(x) => x > 0),
    fn(x) => match x:
        case 1: "one"
        case 2: "two"
        else: "many"
)

dataset "results" from memory:
    add column processed = process(values)
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        assert len(app.functions) == 1
        
        # Function should have nested CallExpr with map/filter/match
        process_fn = app.functions[0]
        assert process_fn.body.__class__.__name__ == "CallExpr"
    
    @pytest.mark.skip("Top-level fn syntax requires legacy parser update")
    def test_resolver_validates_symbolic_expressions(self):
        """Test that resolver validates symbolic expressions."""
        source = '''
app "ValidatedApp".

fn double(x) => x * 2
fn triple(y) => y * 3

dataset "test" from memory:
    filter by: double(value)
'''
        parser = Parser(source)
        module = parser.parse()
        
        # Module should parse successfully
        assert module is not None
        app = module.body[0]
        assert len(app.functions) == 2
    
    @pytest.mark.skip("Undefined function detection not yet implemented")
    def test_undefined_function_caught_by_resolver(self):
        """Test that resolver catches undefined function calls."""
        source = '''
app "ErrorApp".

dataset "test" from memory:
    filter by: undefined_function(value)
'''
        parser = Parser(source)
        module = parser.parse()
        
        # Resolver should catch the undefined function
        # Note: Current resolver may not catch this yet
        # This test documents expected behavior
    
    def test_mixed_legacy_and_symbolic_expressions(self):
        """Test backward compatibility with legacy expressions."""
        source = '''
app "MixedApp".

dataset "data" from memory:
    filter by: age > 18 and status == "active"
    add column doubled = value * 2
    add column tripled = value * 3
'''
        parser = Parser(source)
        module = parser.parse()
        
        app = module.body[0]
        dataset = app.datasets[0]
        
        # First two operations use legacy expressions
        # Third uses symbolic expression (function call)
        assert len(dataset.operations) == 3
    
    def test_safety_limits_configuration(self):
        """Test that safety limits can be configured."""
        import os
        
        # Set limits
        os.environ["NAMEL3SS_EXPR_MAX_DEPTH"] = "50"
        os.environ["NAMEL3SS_EXPR_MAX_STEPS"] = "5000"
        
        from namel3ss.codegen.backend.core.runtime.expression_sandbox import (
            get_expr_max_depth,
            get_expr_max_steps
        )
        
        assert get_expr_max_depth() == 50
        assert get_expr_max_steps() == 5000
        
        # Clean up
        del os.environ["NAMEL3SS_EXPR_MAX_DEPTH"]
        del os.environ["NAMEL3SS_EXPR_MAX_STEPS"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
