#!/usr/bin/env python3
"""Direct test of log statement creation and encoding."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from namel3ss.ast import LogLevel, LogStatement

def test_log_statement_creation():
    """Test LogStatement creation directly."""
    # Test LogStatement creation
    stmt = LogStatement(
        level=LogLevel.ERROR,
        message="Test error message",
        source_location="test.py:42"
    )
    
    assert stmt.level == LogLevel.ERROR
    assert stmt.message == "Test error message" 
    assert stmt.source_location == "test.py:42"
    print("✅ LogStatement creation test passed!")

def test_log_levels():
    """Test LogLevel enum."""
    assert str(LogLevel.DEBUG) == "debug"
    assert str(LogLevel.INFO) == "info"
    assert str(LogLevel.WARN) == "warn"
    assert str(LogLevel.ERROR) == "error"
    print("✅ LogLevel enum test passed!")

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

def test_log_different_levels():
    """Test encoding with different log levels."""
    from namel3ss.codegen.backend.state.statements import _encode_statement
    
    levels = [(LogLevel.DEBUG, "debug"), (LogLevel.INFO, "info"), 
              (LogLevel.WARN, "warn"), (LogLevel.ERROR, "error")]
    
    for level_enum, level_str in levels:
        stmt = LogStatement(
            level=level_enum,
            message=f"Test {level_str} message",
            source_location="test.py:1"
        )
        
        component = _encode_statement(stmt, set(), {})
        assert component.type == "log"
        assert component.payload["level"] == level_str
        assert component.payload["message"] == f"Test {level_str} message"
    
    print("✅ All log level encoding tests passed!")

if __name__ == "__main__":
    test_log_statement_creation()
    test_log_levels()
    test_log_statement_encoding()
    test_log_different_levels()
    print("🎉 All direct log tests passed!")