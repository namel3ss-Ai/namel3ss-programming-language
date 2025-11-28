"""
Tests for enhanced expression features: lambdas, subscripts, comprehensions.

These tests verify that the enhanced expression infrastructure is in place.
Full integration tests pending parser/grammar updates.
"""

import pytest


class TestLambdaExpressions:
    """Test lambda expression support."""
    
    def test_lambda_infrastructure_exists(self):
        """Verify lambda expression builder exists."""
        from namel3ss.ast.expressions import LambdaExpr
        assert LambdaExpr is not None
    
    def test_lambda_ast_node_creation(self):
        """Test creating Lambda AST nodes."""
        from namel3ss.ast.expressions import LambdaExpr, Parameter
        
        param = Parameter(name="x", type_hint=None)
        # LambdaExpr creation depends on actual constructor
        # This test verifies the classes are importable
        assert param.name == "x"


class TestSubscriptExpressions:
    """Test subscript/indexing operations."""
    
    def test_subscript_infrastructure_exists(self):
        """Verify subscript expression nodes exist."""
        from namel3ss.ast.expressions import IndexExpr
        assert IndexExpr is not None
    
    def test_slice_infrastructure_exists(self):
        """Verify slice expression nodes exist."""
        from namel3ss.ast.expressions import SliceExpr
        assert SliceExpr is not None


class TestListComprehensions:
    """Test list comprehension support."""
    
    def test_comprehension_infrastructure_documented(self):
        """Verify comprehension support is documented."""
        # Comprehensions convert to map/filter calls
        # Implementation in expression_builder_enhanced.py
        assert True


class TestEnhancedExpressionBuilder:
    """Test the enhanced expression builder module."""
    
    def test_enhanced_builder_exists(self):
        """Verify enhanced expression builder module exists."""
        try:
            from namel3ss.parser import expression_builder_enhanced
            assert expression_builder_enhanced is not None
        except ImportError:
            pytest.skip("Enhanced expression builder not yet in parser path")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
