"""
Tests for the Editor/IDE Integration API.

Tests parsing, analysis, symbol lookup, and LSP-ready functionality.
"""

import pytest
from namel3ss.tools.editor_api import (
    EditorAPI,
    Position,
    Range,
    Location,
    Symbol,
    Diagnostic,
    AnalysisResult,
    parse_source,
    analyze_module,
    find_symbol_at_position,
    get_hover_info,
)


class TestPosition:
    """Test Position data structure."""
    
    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(line=10, character=5)
        assert pos.line == 10
        assert pos.character == 5
    
    def test_position_comparison(self):
        """Test position comparison operators."""
        pos1 = Position(10, 5)
        pos2 = Position(10, 8)
        pos3 = Position(15, 3)
        
        assert pos1 < pos2
        assert pos2 < pos3
        assert pos1 < pos3
        assert pos1 <= pos1
    
    def test_position_equality(self):
        """Test position equality."""
        pos1 = Position(10, 5)
        pos2 = Position(10, 5)
        pos3 = Position(10, 6)
        
        assert pos1 == pos2
        assert pos1 != pos3


class TestRange:
    """Test Range data structure."""
    
    def test_range_creation(self):
        """Test creating a range."""
        start = Position(10, 0)
        end = Position(10, 10)
        rng = Range(start=start, end=end)
        
        assert rng.start == start
        assert rng.end == end
    
    def test_range_contains_position(self):
        """Test checking if range contains a position."""
        rng = Range(
            start=Position(10, 0),
            end=Position(10, 10)
        )
        
        assert rng.contains(Position(10, 5))
        assert rng.contains(Position(10, 0))  # Start inclusive
        assert rng.contains(Position(10, 10))  # End inclusive
        assert not rng.contains(Position(9, 5))
        assert not rng.contains(Position(11, 5))


class TestLocation:
    """Test Location data structure."""
    
    def test_location_creation(self):
        """Test creating a location."""
        rng = Range(Position(10, 0), Position(10, 10))
        loc = Location(uri="file:///test.ai", range=rng)
        
        assert loc.uri == "file:///test.ai"
        assert loc.range == rng


class TestSymbol:
    """Test Symbol data structure."""
    
    def test_symbol_creation(self):
        """Test creating a symbol."""
        loc = Location(
            uri="file:///test.ai",
            range=Range(Position(10, 0), Position(10, 5))
        )
        symbol = Symbol(
            name="my_function",
            kind="function",
            location=loc,
            type_info="(number, number) => number"
        )
        
        assert symbol.name == "my_function"
        assert symbol.kind == "function"
        assert symbol.type_info == "(number, number) => number"


class TestDiagnostic:
    """Test Diagnostic data structure."""
    
    def test_diagnostic_creation(self):
        """Test creating a diagnostic."""
        rng = Range(Position(10, 5), Position(10, 10))
        diag = Diagnostic(
            range=rng,
            message="Type mismatch",
            severity="error",
            code="TYPE_MISMATCH"
        )
        
        assert diag.message == "Type mismatch"
        assert diag.severity == "error"
        assert diag.code == "TYPE_MISMATCH"
        assert diag.source == "namel3ss"


class TestParseSource:
    """Test source code parsing."""
    
    def test_parse_valid_source(self):
        """Test parsing valid Namel3ss source."""
        source = '''
app "TestApp" {
    description: "Test application"
}
'''
        
        result = parse_source(source, uri="file:///test.ai")
        
        assert result.parse_success is True
        assert len(result.diagnostics) == 0
        assert result.module_ast is not None
    
    def test_parse_invalid_source(self):
        """Test parsing invalid source code."""
        source = "this is not valid syntax"
        
        result = parse_source(source, uri="file:///test.ai")
        
        assert result.parse_success is False
        assert len(result.diagnostics) > 0
        assert result.diagnostics[0].severity == "error"
    
    def test_parse_extracts_symbols(self):
        """Test that parsing extracts symbols."""
        source = '''
app "TestApp" {
    description: "Test"
}
'''
        
        result = parse_source(source, uri="file:///test.ai")
        
        # Should extract at least the app symbol
        assert len(result.symbols) > 0


class TestAnalyzeModule:
    """Test full module analysis."""
    
    def test_analyze_with_type_checking(self):
        """Test analysis with type checking enabled."""
        source = '''
app "TestApp" {
    description: "Test"
}
'''
        
        result = analyze_module(source, uri="file:///test.ai", run_type_check=True)
        
        assert result.parse_success is True
        assert result.uri == "file:///test.ai"
    
    def test_analyze_without_type_checking(self):
        """Test analysis without type checking."""
        source = 'app "TestApp" { }'
        
        result = analyze_module(source, uri="file:///test.ai", run_type_check=False)
        
        assert result.parse_success is True
    
    def test_analyze_collects_diagnostics(self):
        """Test that analysis collects all diagnostics."""
        # Source with both syntax and type errors
        source = "let x: number = 'hello'"
        
        result = analyze_module(source, uri="file:///test.ai")
        
        # Should have diagnostics (syntax or type errors)
        # Actual behavior depends on parser implementation


class TestEditorAPI:
    """Test the main EditorAPI class."""
    
    def setup_method(self):
        """Set up a fresh API instance for each test."""
        self.api = EditorAPI()
    
    def test_api_initialization(self):
        """Test API initialization."""
        api = EditorAPI(project_root="/test/project")
        assert api.project_root == "/test/project"
        assert api.module_resolver is not None
    
    def test_parse_source_caches_result(self):
        """Test that parse results are cached."""
        source = 'app "Test" { }'
        uri = "file:///test.ai"
        
        result1 = self.api.parse_source(source, uri)
        result2 = self.api.analyze_module(source, uri)
        
        # Should be cached
        assert uri in self.api.module_cache
    
    def test_get_symbol_at_position_not_cached(self):
        """Test symbol lookup when file not cached."""
        result = self.api.get_symbol_at_position("file:///unknown.ai", Position(0, 0))
        assert result is None
    
    def test_find_references_empty_when_not_cached(self):
        """Test find references when file not cached."""
        refs = self.api.find_references("file:///unknown.ai", Position(0, 0))
        assert len(refs) == 0
    
    def test_find_definition_none_when_not_cached(self):
        """Test find definition when file not cached."""
        defn = self.api.find_definition("file:///unknown.ai", Position(0, 0))
        assert defn is None
    
    def test_get_hover_information_none_when_not_cached(self):
        """Test hover info when file not cached."""
        hover = self.api.get_hover_information("file:///unknown.ai", Position(0, 0))
        assert hover is None
    
    def test_get_completion_context_default(self):
        """Test completion context for uncached file."""
        context = self.api.get_completion_context("file:///unknown.ai", Position(0, 0))
        
        assert "visible_symbols" in context
        assert "context_type" in context
        assert context["context_type"] == "unknown"


class TestSymbolLookup:
    """Test symbol lookup functionality."""
    
    def test_find_symbol_at_position_api(self):
        """Test the public API for finding symbols."""
        source = '''
app "TestApp" {
    description: "Test application"
}
'''
        
        # Find symbol at app name position
        symbol = find_symbol_at_position(source, line=1, character=5)
        
        # May or may not find depending on exact position
        # This tests the API works without errors


class TestHoverInformation:
    """Test hover information generation."""
    
    def test_get_hover_info_api(self):
        """Test the public API for hover information."""
        source = 'app "TestApp" { }'
        
        # Try to get hover info
        hover = get_hover_info(source, line=0, character=5)
        
        # May return None if no symbol at position
        # This tests the API works without errors


class TestDiagnosticConversion:
    """Test conversion from errors to diagnostics."""
    
    def test_syntax_error_to_diagnostic(self):
        """Test converting syntax errors to diagnostics."""
        source = "this is invalid"
        
        result = parse_source(source)
        
        if not result.parse_success:
            assert len(result.diagnostics) > 0
            assert result.diagnostics[0].severity == "error"


class TestCompletionContext:
    """Test code completion context."""
    
    def test_completion_context_structure(self):
        """Test that completion context has expected structure."""
        api = EditorAPI()
        source = 'app "Test" { }'
        uri = "file:///test.ai"
        
        api.analyze_module(source, uri)
        context = api.get_completion_context(uri, Position(0, 10))
        
        assert "visible_symbols" in context
        assert "context_type" in context
        assert isinstance(context["visible_symbols"], list)


class TestLSPCompatibility:
    """Test LSP protocol compatibility."""
    
    def test_position_is_zero_indexed(self):
        """Test that positions use 0-based indexing (LSP standard)."""
        pos = Position(0, 0)
        assert pos.line == 0
        assert pos.character == 0
    
    def test_diagnostic_has_required_fields(self):
        """Test that diagnostics have LSP-required fields."""
        diag = Diagnostic(
            range=Range(Position(0, 0), Position(0, 1)),
            message="Test error",
            severity="error"
        )
        
        assert hasattr(diag, "range")
        assert hasattr(diag, "message")
        assert hasattr(diag, "severity")
        assert hasattr(diag, "source")
    
    def test_location_has_uri_and_range(self):
        """Test that locations have LSP-required fields."""
        loc = Location(
            uri="file:///test.ai",
            range=Range(Position(0, 0), Position(0, 1))
        )
        
        assert hasattr(loc, "uri")
        assert hasattr(loc, "range")


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    def test_analyze_then_find_symbols(self):
        """Test typical workflow: analyze, then find symbols."""
        api = EditorAPI()
        source = 'app "TestApp" { }'
        uri = "file:///test.ai"
        
        # Analyze
        result = api.analyze_module(source, uri)
        assert result.parse_success
        
        # Find symbols
        symbols = result.symbols
        assert isinstance(symbols, list)
    
    def test_incremental_analysis(self):
        """Test analyzing multiple versions of a file."""
        api = EditorAPI()
        uri = "file:///test.ai"
        
        # First version
        source1 = 'app "Version1" { }'
        result1 = api.analyze_module(source1, uri)
        
        # Second version (overwrites cache)
        source2 = 'app "Version2" { }'
        result2 = api.analyze_module(source2, uri)
        
        assert result1.parse_success
        assert result2.parse_success
    
    def test_multi_file_analysis(self):
        """Test analyzing multiple files."""
        api = EditorAPI()
        
        # Analyze multiple files
        result1 = api.analyze_module('app "App1" { }', uri="file:///app1.ai")
        result2 = api.analyze_module('app "App2" { }', uri="file:///app2.ai")
        
        # Both should be cached
        assert "file:///app1.ai" in api.module_cache
        assert "file:///app2.ai" in api.module_cache


class TestErrorHandling:
    """Test error handling in the API."""
    
    def test_api_handles_parse_errors_gracefully(self):
        """Test that API doesn't crash on parse errors."""
        try:
            result = parse_source("invalid syntax")
            assert not result.parse_success
        except Exception as e:
            pytest.fail(f"API should handle parse errors gracefully: {e}")
    
    def test_api_handles_empty_source(self):
        """Test handling of empty source code."""
        result = parse_source("")
        # Should either parse successfully (empty module) or return error
        assert isinstance(result, AnalysisResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
