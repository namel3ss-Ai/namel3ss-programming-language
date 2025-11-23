#!/usr/bin/env python3
"""Test CLI logging configuration."""

import sys
import os
import logging
sys.path.insert(0, os.path.dirname(__file__))

def test_cli_logging_setup():
    """Test that CLI logging configuration works."""
    
    # Import and configure with debug level
    from namel3ss.cli import _configure_runtime_logging
    
    # Create mock args object
    class MockArgs:
        log_level = "debug"
    
    args = MockArgs()
    _configure_runtime_logging(args)
    
    # Test that the logger is configured correctly
    runtime_logger = logging.getLogger('namel3ss.runtime')
    assert runtime_logger.level == logging.DEBUG, f"Expected DEBUG level, got {runtime_logger.level}"
    assert len(runtime_logger.handlers) > 0, "Expected at least one handler"
    assert not runtime_logger.propagate, "Expected propagate=False"
    
    print("✅ CLI logging configuration test passed!")
    
    # Test that log messages work
    runtime_logger.debug("Test debug message")
    runtime_logger.info("Test info message") 
    runtime_logger.warning("Test warning message")
    runtime_logger.error("Test error message")
    
    print("✅ Log message test completed!")

def test_environment_variable():
    """Test that NAMEL3SS_LOG_LEVEL environment variable works."""
    
    # Set environment variable
    os.environ['NAMEL3SS_LOG_LEVEL'] = 'error'
    
    from namel3ss.cli import _configure_runtime_logging
    
    # Create mock args object with no log_level
    class MockArgs:
        pass
    
    args = MockArgs()
    _configure_runtime_logging(args)
    
    # Test that the logger uses environment variable
    runtime_logger = logging.getLogger('namel3ss.runtime')
    assert runtime_logger.level == logging.ERROR, f"Expected ERROR level from env var, got {runtime_logger.level}"
    
    print("✅ Environment variable test passed!")
    
    # Clean up
    del os.environ['NAMEL3SS_LOG_LEVEL']

if __name__ == "__main__":
    test_cli_logging_setup()
    test_environment_variable()
    print("🎉 All CLI logging tests passed!")