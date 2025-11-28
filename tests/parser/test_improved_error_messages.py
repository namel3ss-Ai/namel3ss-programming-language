"""Test suite for improved error messages in Step 1 of Processing Tools upgrade.

Tests verify that common syntax mistakes produce clear, helpful error messages
pointing users to the correct alternatives.
"""

import pytest
from namel3ss.lang.grammar.parser import _GrammarModuleParser
from namel3ss.parser.base import N3SyntaxError
from namel3ss.lang.grammar.helpers import GrammarUnsupportedError


class TestTypeKeywordError:
    """Test that 'type' keyword produces a helpful error message."""
    
    def test_type_declaration_error(self):
        """Test that 'type' keyword suggests dataset/frame alternatives."""
        source = 'type "FileProcessingJob":'
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = str(exc_info.value)
        assert "'type' keyword is not supported" in error
        assert "dataset" in error.lower()
        assert "frame" in error.lower()
        assert "DATA_MODELS_GUIDE.md" in error


class TestPageColonSyntaxError:
    """Test that page {...} produces error suggesting colon syntax."""
    
    def test_page_with_brace_error(self):
        """Test that page with { suggests : syntax."""
        source = 'page "Dashboard" at "/" {'
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = str(exc_info.value)
        assert "must end with ':'" in error
        assert "not '{'" in error
    
    def test_page_with_colon_works(self):
        """Test that page with : works correctly."""
        source = '''page "Dashboard" at "/":
    show text "Hello"'''
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        module = parser.parse()
        
        assert module is not None
        assert len(module.body) == 1
        assert module.body[0].pages[0].name == "Dashboard"
        assert len(module.body[0].pages[0].body) > 0


class TestPromptColonSyntaxError:
    """Test that prompt {...} produces error suggesting colon syntax."""
    
    def test_prompt_with_brace_error(self):
        """Test that prompt with { suggests : syntax."""
        source = 'prompt AnalyzeFile {'
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = str(exc_info.value)
        assert "must end with ':'" in error
        assert "not '{'" in error
    
    def test_prompt_with_colon_works(self):
        """Test that prompt with : works correctly."""
        source = '''prompt AnalyzeFile:
    model: "gpt-4"
    template: "Analyze this"'''
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        module = parser.parse()
        
        assert module is not None
        assert len(module.body[0].prompts) == 1
        assert module.body[0].prompts[0].name == "AnalyzeFile"


class TestQueryBlockError:
    """Test that 'query' keyword with improper syntax produces helpful errors."""
    
    def test_query_keyword_with_brace(self):
        """Test that query with { gets caught by our error."""
        source = 'query {}'
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error = str(exc_info.value)
        assert "query" in error.lower()
        # Should mention dataset operations or proper query syntax
        assert "dataset" in error.lower() or "QUERIES" in error


class TestTernaryOperatorError:
    """Test that ternary operators produce helpful error messages."""
    
    def test_ternary_in_dataset_filter(self):
        """Test that ternary operator in filter produces error."""
        source = '''dataset "Test" from postgres "db":
    filter by: status == "active" ? 1 : 0'''
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        error_str = str(exc_info.value)
        # Should detect the ? character and mention ternary
        assert "?" in error_str or "ternary" in error_str.lower()
    
    def test_if_else_block_works(self):
        """Test that if/else blocks work as alternative to ternaries."""
        source = '''page "Test" at "/":
    if status == "active":
        show text "Active"
    else:
        show text "Inactive"'''
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        module = parser.parse()
        
        assert module is not None
        assert len(module.body[0].pages) == 1
        assert len(module.body[0].pages[0].body) > 0


class TestComprehensiveErrorScenarios:
    """Test multiple error scenarios to ensure they don't interfere."""
    
    def test_multiple_errors_detected_separately(self):
        """Each error should be caught independently."""
        
        # Test type error
        with pytest.raises(N3SyntaxError) as exc_info:
            parser = _GrammarModuleParser('type "Test":', path="test.ai")
            parser.parse()
        assert "type" in str(exc_info.value).lower()
        
        # Test page brace error
        with pytest.raises(N3SyntaxError) as exc_info:
            parser = _GrammarModuleParser('page "Test" at "/" {', path="test.ai")
            parser.parse()
        assert ":" in str(exc_info.value)
        
        # Test prompt brace error  
        with pytest.raises(N3SyntaxError) as exc_info:
            parser = _GrammarModuleParser('prompt Test {', path="test.ai")
            parser.parse()
        assert ":" in str(exc_info.value)
    
    def test_error_messages_include_documentation_links(self):
        """All errors should point to relevant documentation."""
        
        # Type error links to DATA_MODELS_GUIDE
        with pytest.raises(N3SyntaxError) as exc_info:
            parser = _GrammarModuleParser('type "Test":', path="test.ai")
            parser.parse()
        assert "DATA_MODELS_GUIDE" in str(exc_info.value)
        
        # Query keyword error links to QUERIES docs
        with pytest.raises(N3SyntaxError) as exc_info:
            parser = _GrammarModuleParser('query {}', path="test.ai")
            parser.parse()
        error = str(exc_info.value)
        assert "QUERIES" in error or "dataset" in error.lower()


class TestValidAlternatives:
    """Test that all suggested alternatives actually work."""
    
    def test_dataset_as_alternative_to_type(self):
        """Dataset should work for defining data models."""
        source = '''dataset "FileProcessingJob" from postgres "jobs":
    filter by: status == "pending"'''
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        module = parser.parse()
        
        assert module is not None
        assert len(module.body[0].datasets) == 1
        assert module.body[0].datasets[0].name == "FileProcessingJob"
    
    def test_frame_as_alternative_to_type(self):
        """Frame should work for defining analytical schemas."""
        source = '''frame "JobAnalysis":
    columns: filename, status, created_at'''
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        module = parser.parse()
        
        assert module is not None
        assert len(module.body[0].frames) == 1
        assert module.body[0].frames[0].name == "JobAnalysis"
    
    def test_dataset_filter_as_alternative_to_query(self):
        """Dataset filter operations should work as alternative to query blocks."""
        source = '''dataset "Jobs" from postgres "jobs":
    filter by: status == "pending"
    group by: category'''
        parser = _GrammarModuleParser(source.strip(), path="test.ai")
        module = parser.parse()
        
        assert module is not None
        assert len(module.body[0].datasets) == 1
        dataset = module.body[0].datasets[0]
        assert len(dataset.operations) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
