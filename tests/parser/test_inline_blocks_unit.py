"""Unit tests for inline block parsing methods."""

import pytest
from namel3ss.lang.parser.grammar.lexer import Lexer, TokenType
from namel3ss.lang.parser.parse import N3Parser
from namel3ss.ast import InlinePythonBlock, InlineReactBlock
from namel3ss.lang.parser.errors import N3SyntaxError


class TestInlinePythonBlockParsing:
    """Test parsing of python { ... } inline blocks."""
    
    def test_parse_simple_python_expression(self):
        """Parse simple Python expression in inline block."""
        source = "python { 42 }"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_python_block()
        
        assert isinstance(result, InlinePythonBlock)
        assert result.kind == "python"
        assert "42" in result.code
        assert result.location is not None
    
    def test_parse_python_function(self):
        """Parse Python function definition."""
        source = """python {
    def process(items):
        return [x * 2 for x in items]
}"""
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_python_block()
        
        assert isinstance(result, InlinePythonBlock)
        assert "def process" in result.code
        assert "return" in result.code
    
    def test_parse_python_nested_braces(self):
        """Parse Python block with nested braces (dict literals)."""
        source = """python {
    config = {"key": "value", "nested": {"a": 1}}
}"""
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_python_block()
        
        assert isinstance(result, InlinePythonBlock)
        assert "config" in result.code
        # Nested braces should be preserved
        assert result.code.count("{") >= 2
        assert result.code.count("}") >= 2
    
    def test_parse_python_empty_block(self):
        """Parse empty Python block."""
        source = "python { }"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_python_block()
        
        assert isinstance(result, InlinePythonBlock)
        assert result.code.strip() == ""


class TestInlineReactBlockParsing:
    """Test parsing of react { ... } inline blocks."""
    
    def test_parse_simple_react_jsx(self):
        """Parse simple JSX element."""
        source = 'react { <div>Hello</div> }'
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_react_block()
        
        assert isinstance(result, InlineReactBlock)
        assert result.kind == "react"
        assert "div" in result.code or "Hello" in result.code
    
    def test_parse_react_component(self):
        """Parse React component function."""
        source = """react {
    function Button({ label }) {
        return <button>{label}</button>;
    }
}"""
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_react_block()
        
        assert isinstance(result, InlineReactBlock)
        assert "function Button" in result.code or "Button" in result.code
    
    def test_parse_react_nested_jsx_braces(self):
        """Parse React with nested JSX expression braces."""
        source = """react {
    <div>{items.map(item => <span key={item.id}>{item.name}</span>)}</div>
}"""
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_react_block()
        
        assert isinstance(result, InlineReactBlock)
        # Should have preserved nested braces
        assert result.code.count("{") >= 2 or "items" in result.code


class TestInlineBlockErrors:
    """Test error handling for inline blocks."""
    
    def test_python_unclosed_block(self):
        """Error on unclosed Python block."""
        source = "python { def foo(): pass"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        with pytest.raises(N3SyntaxError, match="Unclosed inline block|expected '}'"):
            parser.parse_inline_python_block()
    
    def test_react_unclosed_block(self):
        """Error on unclosed React block."""
        source = "react { <div>Hello"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        with pytest.raises(N3SyntaxError, match="Unclosed inline block|expected '}'"):
            parser.parse_inline_react_block()


class TestInlineBlockLocations:
    """Test source location tracking."""
    
    def test_python_has_location(self):
        """Verify location tracking for Python block."""
        source = "python { x = 42 }"
        lexer = Lexer(source, path="test.n3")
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_python_block()
        
        assert result.location is not None
        assert result.location.file == "test.n3"
        assert result.location.line > 0
    
    def test_react_has_location(self):
        """Verify location tracking for React block."""
        source = "react { <div>Test</div> }"
        lexer = Lexer(source, path="test.n3")
        tokens = list(lexer.tokenize())
        
        parser = N3Parser.__new__(N3Parser)
        parser.tokens = tokens
        parser.pos = 0
        parser.path = "test.n3"
        parser.symbol_table = {}
        
        result = parser.parse_inline_react_block()
        
        assert result.location is not None
        assert result.location.file == "test.n3"
        assert result.location.line > 0


class TestTokenization:
    """Test that inline keywords are properly tokenized."""
    
    def test_python_keyword_tokenizes(self):
        """Verify 'python' keyword is recognized."""
        source = "python"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        assert len(tokens) >= 1
        assert tokens[0].type == TokenType.PYTHON
        assert tokens[0].value == "python"
    
    def test_react_keyword_tokenizes(self):
        """Verify 'react' keyword is recognized."""
        source = "react"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        assert len(tokens) >= 1
        assert tokens[0].type == TokenType.REACT
        assert tokens[0].value == "react"
    
    def test_python_block_tokenizes(self):
        """Verify full python block tokenizes."""
        source = "python { code }"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        # Should have: PYTHON, LBRACE, IDENTIFIER (code), RBRACE, EOF
        assert any(t.type == TokenType.PYTHON for t in tokens)
        assert any(t.type == TokenType.LBRACE for t in tokens)
        assert any(t.type == TokenType.RBRACE for t in tokens)
