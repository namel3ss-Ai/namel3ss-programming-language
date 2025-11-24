"""Test basic unified parser functionality."""

import pytest
from namel3ss.lang.parser import parse_module, N3SyntaxError


def test_parse_simple_app():
    """Test parsing a simple app declaration."""
    source = '''
app "Test App" {
  description: "A test application"
  version: "1.0.0"
}
'''
    
    module = parse_module(source)
    assert module is not None
    assert len(module.body) > 0


def test_parse_llm_declaration():
    """Test parsing an LLM declaration."""
    source = '''
llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
}
'''
    
    module = parse_module(source)
    assert module is not None


def test_parse_with_import():
    """Test parsing with import statement."""
    source = '''
import ai.models as models

app "Test" {
  version: "1.0"
}
'''
    
    module = parse_module(source)
    assert len(module.imports) == 1
    assert module.imports[0].module == "ai.models"
    assert module.imports[0].alias == "models"


def test_syntax_error_reporting():
    """Test that syntax errors are properly reported."""
    source = '''
app "Test" 
  missing: "brace"
}
'''
    
    with pytest.raises(N3SyntaxError) as exc_info:
        parse_module(source)
    
    error = exc_info.value
    assert error.line is not None
    assert "Expected" in str(error) or "Unexpected" in str(error)


def test_quoted_names_required():
    """Test that quoted names are required for declarations."""
    source = '''
app TestApp {
  description: "Test"
}
'''
    
    with pytest.raises(N3SyntaxError):
        parse_module(source)


if __name__ == "__main__":
    # Run basic tests
    print("Testing simple app parsing...")
    test_parse_simple_app()
    print("✓ Simple app parsed")
    
    print("\nTesting LLM declaration...")
    test_parse_llm_declaration()
    print("✓ LLM declaration parsed")
    
    print("\nTesting imports...")
    test_parse_with_import()
    print("✓ Imports parsed")
    
    print("\nTesting error reporting...")
    test_syntax_error_reporting()
    print("✓ Syntax errors reported correctly")
    
    print("\nTesting quoted name requirement...")
    test_quoted_names_required()
    print("✓ Quoted names enforced")
    
    print("\n✅ All basic parser tests passed!")
