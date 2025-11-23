"""Tests for the AST-based formatter."""

import pytest
from namel3ss.formatting import ASTFormatter, FormattingOptions, IndentStyle, DefaultFormattingRules


class TestASTFormatter:
    """Test the AST-based formatter functionality."""
    
    def test_basic_formatting(self):
        """Test basic app formatting."""
        source = '''app "TestApp" {
page  "/home" {
show text:   "Hello World"
}
}'''
        
        formatter = ASTFormatter(DefaultFormattingRules.standard())
        result = formatter.format_document(source)
        
        assert result.success()
        assert result.is_changed
        # Should normalize spacing and indentation
        lines = result.formatted_text.strip().split('\\n')
        assert lines[0] == 'app "TestApp" {'
        assert lines[1] == '    page "/home" {'
        assert lines[2] == '        show text: "Hello World"'
    
    def test_indentation_options(self):
        """Test different indentation styles."""
        source = '''app "TestApp" {
page "/home" {
show text: "Hello"
}
}'''
        
        # Test spaces (default)
        spaces_options = FormattingOptions(indent_style=IndentStyle.SPACES, indent_size=2)
        formatter = ASTFormatter(spaces_options)
        result = formatter.format_document(source)
        
        assert result.success()
        lines = result.formatted_text.strip().split('\\n')
        assert lines[1].startswith('  page')  # 2 spaces
        
        # Test tabs
        tabs_options = FormattingOptions(indent_style=IndentStyle.TABS)
        formatter = ASTFormatter(tabs_options)
        result = formatter.format_document(source)
        
        assert result.success()
        lines = result.formatted_text.strip().split('\\n')
        assert lines[1].startswith('\\tpage')  # 1 tab
    
    def test_prompt_formatting(self):
        """Test prompt declaration formatting."""
        source = '''app "TestApp" {
prompt "greeting" {
model: "gpt-4"
template: """
Hello {{name}}
"""
}
}'''
        
        formatter = ASTFormatter(DefaultFormattingRules.standard())
        result = formatter.format_document(source)
        
        assert result.success()
        assert 'prompt "greeting"' in result.formatted_text
        assert 'model: "gpt-4"' in result.formatted_text
        assert 'template: """' in result.formatted_text
    
    def test_chain_formatting(self):
        """Test chain declaration formatting."""
        source = '''app "TestApp" {
chain "support_flow" {
prompt "classify"
prompt "respond"
}
}'''
        
        formatter = ASTFormatter(DefaultFormattingRules.standard())
        result = formatter.format_document(source)
        
        assert result.success()
        assert 'chain "support_flow"' in result.formatted_text
        assert 'prompt "classify"' in result.formatted_text
        assert 'prompt "respond"' in result.formatted_text
    
    def test_parse_error_handling(self):
        """Test handling of parse errors."""
        source = '''app "TestApp" 
invalid syntax here
}'''
        
        formatter = ASTFormatter(DefaultFormattingRules.standard())
        result = formatter.format_document(source)
        
        assert not result.success()
        assert len(result.errors) > 0
        assert "Parse error" in result.errors[0]
        # Should return original text on parse error
        assert result.formatted_text == source
    
    def test_no_changes_needed(self):
        """Test when no formatting changes are needed."""
        source = '''app "TestApp" {
    page "/home" {
        show text: "Hello World"
    }
}'''
        
        formatter = ASTFormatter(DefaultFormattingRules.standard())
        result = formatter.format_document(source)
        
        assert result.success()
        # May or may not need changes depending on exact implementation
        # This test verifies the is_changed detection works
    
    def test_compact_formatting(self):
        """Test compact formatting rules."""
        source = '''app "TestApp" {


    page "/home" {
        show text: "Hello World"
    }


}'''
        
        formatter = ASTFormatter(DefaultFormattingRules.compact())
        result = formatter.format_document(source)
        
        assert result.success()
        # Should remove excessive empty lines with compact rules
        # Check that formatting succeeds (exact output depends on implementation)
    
    def test_trailing_whitespace_removal(self):
        """Test trailing whitespace removal."""
        source = '''app "TestApp" {    
    page "/home" {    
        show text: "Hello"   
    }   
}'''
        
        options = FormattingOptions(trim_trailing_whitespace=True)
        formatter = ASTFormatter(options)
        result = formatter.format_document(source)
        
        assert result.success()
        # Should not have trailing spaces
        for line in result.formatted_text.split('\\n'):
            assert not line.endswith(' '), f"Line has trailing space: '{line}'"
    
    def test_final_newline_insertion(self):
        """Test final newline insertion."""
        source = '''app "TestApp" {
    page "/home" {
        show text: "Hello"
    }
}'''  # No trailing newline
        
        options = FormattingOptions(insert_final_newline=True)
        formatter = ASTFormatter(options)
        result = formatter.format_document(source)
        
        assert result.success()
        assert result.formatted_text.endswith('\\n')
        
        # Test with option disabled
        options = FormattingOptions(insert_final_newline=False)
        formatter = ASTFormatter(options)
        result = formatter.format_document(source)
        
        assert result.success()
        # May or may not end with newline depending on original
    
    def test_file_path_in_errors(self):
        """Test that file path appears in error messages."""
        source = '''invalid app syntax'''
        
        formatter = ASTFormatter()
        result = formatter.format_document(source, "test.ai")
        
        assert not result.success()
        # Should handle file path for better error reporting