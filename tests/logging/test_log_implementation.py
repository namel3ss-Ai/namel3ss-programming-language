#!/usr/bin/env python3
"""Quick test to verify log statement implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from namel3ss.lang.parser import parse_module
from namel3ss.ast import LogLevel, LogStatement

def test_log_parsing():
    """Test that log statements parse correctly."""
    
    # Test basic log statement - need to parse in a page context
    source = '''
app "Test App"

page "test" at "/":
    log "Hello world"
'''
    module = parse_module(source, "test.ai")
    pages = module.pages
    assert len(pages) == 1, f"Expected 1 page, got {len(pages)}"
    statements = pages[0].statements
    assert len(statements) == 1, f"Expected 1 statement, got {len(statements)}"
    
    result = statements[0]
    assert isinstance(result, LogStatement), f"Expected LogStatement, got {type(result)}"
    assert result.level == LogLevel.INFO, f"Expected INFO level, got {result.level}"
    assert result.message == "Hello world", f"Expected 'Hello world', got {result.message}"
    
    # Test log with level  
    source2 = '''
app "Test App"

page "test" at "/":
    log error "Something went wrong"
'''
    module2 = parse_module(source2, "test.ai")
    result2 = module2.pages[0].statements[0]
    assert isinstance(result2, LogStatement)
    assert result2.level == LogLevel.ERROR, f"Expected ERROR level, got {result2.level}"
    assert result2.message == "Something went wrong"
    
    # Test all levels
    levels = ["debug", "info", "warn", "error"]
    for level in levels:
        source_level = f'''
app "Test App"

page "test" at "/":
    log {level} "Test message"
'''
        module_level = parse_module(source_level, "test.ai")
        result_level = module_level.pages[0].statements[0]
        assert isinstance(result_level, LogStatement)
        expected_level = getattr(LogLevel, level.upper())
        assert result_level.level == expected_level
    
    print("✅ All log parsing tests passed!")

def test_log_statement_encoding():
    """Test that LogStatement encodes correctly for backend."""
    from namel3ss.codegen.backend.state.statements import _encode_statement
    
    # Create a LogStatement
    stmt = LogStatement(
        level=LogLevel.INFO,
        message="Test log message",
        source_location="test.py:10"
    )
    
    # Encode it
    component = _encode_statement(stmt, set(), {})
    
    assert component is not None, "LogStatement should encode to a component"
    assert component.type == "log", f"Expected type 'log', got {component.type}"
    assert component.payload["level"] == "info", f"Expected level 'info', got {component.payload['level']}"
    assert component.payload["message"] == "Test log message"
    assert component.payload["source_location"] == "test.py:10"
    
    print("✅ Log statement encoding test passed!")

if __name__ == "__main__":
    test_log_parsing()
    test_log_statement_encoding()
    print("🎉 All tests passed! Logging implementation is working.")