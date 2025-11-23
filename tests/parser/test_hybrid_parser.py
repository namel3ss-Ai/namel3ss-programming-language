"""
Test suite for the hybrid parser architecture.

Tests both modern brace syntax and legacy dot/colon syntax,
along with the fallback mechanism and tracking system.
"""

import pytest
from namel3ss.parser import Parser, _FallbackTracker
from namel3ss.lang.parser import parse_module as new_parse_module, N3SyntaxError
from namel3ss.ast import App, Page


class TestHybridParserArchitecture:
    """Test the hybrid parser system with unified and legacy parsers."""
    
    def setup_method(self):
        """Reset the fallback tracker before each test."""
        # Reset tracker state manually since there's no reset method
        _FallbackTracker._count = 0
        _FallbackTracker._last_path = None
        _FallbackTracker._last_error_code = None
        _FallbackTracker._last_message = None

    def test_modern_syntax_uses_unified_parser(self):
        """Modern brace syntax should parse with unified parser (no fallback)."""
        source = '''
app "Modern Test" {
    description: "Uses modern brace syntax"
}

page "Home" at "/" {
    show text "Hello World"
}
'''
        parser = Parser(source, path="test.n3")
        module = parser.parse()
        
        # Verify parse success
        assert module is not None
        assert len(module.body) == 1
        app = module.body[0]
        assert isinstance(app, App)
        assert app.name == "Modern Test"
        assert len(app.pages) == 1
        assert app.pages[0].name == "Home"
        
        # Verify no fallback was used
        snapshot = _FallbackTracker.snapshot()
        assert snapshot["count"] == 0
        assert snapshot["last_path"] is None

    def test_legacy_dot_syntax_uses_fallback(self):
        """Legacy dot syntax should trigger fallback to legacy parser."""
        source = '''app "Legacy Test".

page "Home" at "/":
  show text "Hello World"
'''
        parser = Parser(source, path="legacy_test.n3")
        module = parser.parse()
        
        # Verify parse success via fallback
        assert module is not None
        assert len(module.body) == 1
        app = module.body[0]
        assert isinstance(app, App)
        assert app.name == "Legacy Test"
        assert len(app.pages) == 1
        assert app.pages[0].name == "Home"
        
        # Verify fallback was used
        snapshot = _FallbackTracker.snapshot()
        assert snapshot["count"] == 1
        assert snapshot["last_path"] == "legacy_test.n3"
        assert snapshot["last_error_code"] == "SYNTAX_ERROR"

    def test_legacy_frame_with_columns_shorthand(self):
        """Legacy frame syntax with columns: shorthand should work via fallback."""
        source = '''app "Dashboard Test".

frame "orders_summary" from dataset "orders":
  columns: id, customer_id, status, total
'''
        parser = Parser(source, path="dashboard.n3")
        module = parser.parse()
        
        # Verify parse success
        assert module is not None
        app = module.body[0]
        assert len(app.frames) == 1
        
        frame = app.frames[0]
        assert frame.name == "orders_summary"
        assert len(frame.columns) == 4
        assert [col.name for col in frame.columns] == ["id", "customer_id", "status", "total"]
        
        # Verify all columns have default string type
        for col in frame.columns:
            assert col.dtype == "string"
            assert col.nullable is True

    def test_fallback_tracker_counts_multiple_files(self):
        """Fallback tracker should accumulate counts across multiple parse attempts."""
        # First legacy file
        source1 = 'app "Test1".'
        parser1 = Parser(source1, path="test1.n3")
        parser1.parse()
        
        # Second legacy file
        source2 = 'app "Test2".'
        parser2 = Parser(source2, path="test2.n3")
        parser2.parse()
        
        # Check tracker
        snapshot = _FallbackTracker.snapshot()
        assert snapshot["count"] == 2
        assert snapshot["last_path"] == "test2.n3"

    def test_mixed_syntax_in_session(self):
        """Should handle mix of modern and legacy syntax in same session."""
        # Modern syntax first (no fallback)
        modern = '''app "Modern" {
    description: "Modern syntax"
}'''
        parser_modern = Parser(modern, path="modern.n3")
        parser_modern.parse()
        
        snapshot1 = _FallbackTracker.snapshot()
        assert snapshot1["count"] == 0
        
        # Legacy syntax second (should fallback)
        legacy = 'app "Legacy".'
        parser_legacy = Parser(legacy, path="legacy.n3")
        parser_legacy.parse()
        
        snapshot2 = _FallbackTracker.snapshot()
        assert snapshot2["count"] == 1
        assert snapshot2["last_path"] == "legacy.n3"

    def test_unified_parser_direct_call_on_legacy_fails(self):
        """Direct unified parser call on legacy syntax should fail cleanly."""
        legacy_source = 'app "Legacy".'
        
        with pytest.raises(N3SyntaxError) as exc_info:
            new_parse_module(legacy_source, path="test.n3")
        
        # Should get a clear error about unexpected declaration
        assert "Unexpected top-level declaration" in str(exc_info.value)

    def test_complex_legacy_syntax_patterns(self):
        """Test various complex legacy syntax patterns."""
        source = '''app "Complex Legacy" connects to postgres "testdb".

dataset "users" from table user_data:
  filter by: active == true

frame "user_stats" from dataset "users":
  columns: id, name, email, created_at
  
page "Dashboard" at "/dashboard":
  show text "User Dashboard"
  show table "Active Users" from dataset users
'''
        parser = Parser(source, path="complex.n3")
        module = parser.parse()
        
        # Verify successful parsing
        assert module is not None
        app = module.body[0]
        assert app.name == "Complex Legacy"
        assert len(app.datasets) == 1
        assert len(app.frames) == 1
        assert len(app.pages) == 1
        
        # Verify dataset
        dataset = app.datasets[0]
        assert dataset.name == "users"
        
        # Verify frame with columns
        frame = app.frames[0]
        assert frame.name == "user_stats"
        assert len(frame.columns) == 4
        
        # Verify page
        page = app.pages[0]
        assert page.name == "Dashboard"
        assert page.route == "/dashboard"
        
        # Verify fallback was used
        snapshot = _FallbackTracker.snapshot()
        assert snapshot["count"] == 1


class TestEnhancedErrorMessages:
    """Test the enhanced error messages for better developer experience."""
    
    def test_show_command_error_message(self):
        """Test error message for misplaced show command."""
        source = 'show text "Hello"'
        
        with pytest.raises(N3SyntaxError) as exc_info:
            new_parse_module(source, path="test.n3")
        
        error_msg = str(exc_info.value)
        assert "This looks like content that should be inside a declaration block" in error_msg
        assert "Modern syntax requires braces" in error_msg

    def test_component_error_message(self):
        """Test error message for misplaced page components."""
        source = 'text'
        
        with pytest.raises(N3SyntaxError) as exc_info:
            new_parse_module(source, path="test.n3")
        
        error_msg = str(exc_info.value)
        assert "This looks like a page component" in error_msg
        assert "Components must be inside a page declaration" in error_msg

    def test_table_component_error_message(self):
        """Test error message for table component outside page."""
        source = 'table'
        
        with pytest.raises(N3SyntaxError) as exc_info:
            new_parse_module(source, path="test.n3")
        
        error_msg = str(exc_info.value)
        assert "page component" in error_msg

    def test_filter_command_error_message(self):
        """Test error message for filter command outside proper context."""
        source = 'filter by: status == "active"'
        
        with pytest.raises(N3SyntaxError) as exc_info:
            new_parse_module(source, path="test.n3")
        
        error_msg = str(exc_info.value)
        assert "declaration block" in error_msg


class TestBackwardCompatibility:
    """Test that our hybrid system maintains 100% backward compatibility."""
    
    def test_all_legacy_examples_still_work(self):
        """Test that common legacy syntax patterns still parse correctly."""
        legacy_examples = [
            # Simple app with dot
            'app "Simple".',
            
            # App with page and colon
            '''app "WithPage".
page "Home" at "/":
  show text "Hello"''',
            
            # App with database connection
            'app "Database" connects to postgres "db".',
            
            # Complex frame syntax
            '''app "FrameTest".
frame "data" from dataset "source":
  columns: id, name, value''',
        ]
        
        for i, source in enumerate(legacy_examples):
            parser = Parser(source, path=f"legacy_{i}.n3")
            module = parser.parse()
            
            # Each should parse successfully
            assert module is not None
            assert len(module.body) == 1
            assert isinstance(module.body[0], App)

    def test_no_breaking_changes_in_modern_syntax(self):
        """Ensure modern syntax still works exactly as before."""
        source = '''
app "Modern App" {
    description: "Test application"
    theme: {
        primary_color: "#007bff"
    }
}

llm "gpt4" {
    provider: "openai"
    model: "gpt-4o-mini"
}

page "Home" at "/" {
    show text "Welcome to the app"
    show form "Contact" {
        field "name" type="text" required=true
        field "email" type="email" required=true
    }
}
'''
        parser = Parser(source, path="modern.n3")
        module = parser.parse()
        
        # Should parse without fallback
        assert module is not None
        app = module.body[0]
        assert app.name == "Modern App"
        assert len(app.llms) == 1
        assert len(app.pages) == 1
        
        # No fallback should be used
        snapshot = _FallbackTracker.snapshot()
        assert snapshot["count"] == 0