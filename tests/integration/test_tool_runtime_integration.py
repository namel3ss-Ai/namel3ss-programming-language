"""Integration tests for tool block runtime generation."""

import os
import tempfile

from namel3ss.parser import Parser
from namel3ss.codegen.backend import generate_backend


def test_tool_block_compilation():
    """Test that tool blocks compile and generate proper runtime code."""
    source = """
app "TestApp".

tool get_weather:
    type: http
    endpoint: https://api.weather.com/v1/current
    method: GET
    timeout: 10.0

tool search_api:
    type: http
    endpoint: https://api.search.com/v1/query
    method: POST
    timeout: 15.0
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify tools were parsed
    assert len(app.tools) == 2
    assert app.tools[0].name == "get_weather"
    assert app.tools[0].type == "http"
    assert app.tools[0].endpoint == "https://api.weather.com/v1/current"
    assert app.tools[0].method == "GET"
    assert app.tools[0].timeout == 10.0
    
    assert app.tools[1].name == "search_api"
    assert app.tools[1].type == "http"
    assert app.tools[1].method == "POST"
    
    # Compile to backend
    output_dir = "/tmp/test_tool_compilation"
    generate_backend(app, output_dir)
    
    # Verify runtime was generated
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    assert os.path.exists(runtime_path)
    
    # Read generated code
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify tool initialization code was generated
    assert "TOOL_REGISTRY" in runtime_code
    assert "_TOOL_INSTANCES" in runtime_code
    assert "_initialize_tool_instances" in runtime_code
    assert "from namel3ss.tools import create_tool" in runtime_code
    assert "get_weather" in runtime_code
    assert "search_api" in runtime_code


def test_tool_registry_structure():
    """Test that tool configurations are properly included in generated code."""
    source = """
app "TestApp".

tool weather_tool:
    type: http
    endpoint: https://api.weather.com/v1/current
    method: GET
    timeout: 20.0
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    output_dir = "/tmp/test_tool_registry"
    generate_backend(app, output_dir)
    
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify all tool properties are in generated code
    assert "weather_tool" in runtime_code
    assert "https://api.weather.com/v1/current" in runtime_code
    assert "20.0" in runtime_code


def test_multiple_tool_types():
    """Test multiple tool types in one app."""
    source = """
app "TestApp".

tool http_tool:
    type: http
    endpoint: https://api.example.com
    method: POST

tool python_tool:
    type: python
    timeout: 5.0
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    # Verify all tools were parsed
    assert len(app.tools) == 2
    
    types = {tool.type for tool in app.tools}
    assert types == {"http", "python"}
    
    # Compile
    output_dir = "/tmp/test_multiple_tools"
    generate_backend(app, output_dir)
    
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    assert "_TOOL_INSTANCES" in runtime_code
    assert "http_tool" in runtime_code
    assert "python_tool" in runtime_code


def test_tool_initialization_error_handling():
    """Test that tool initialization handles errors gracefully."""
    source = """
app "TestApp".

tool valid_tool:
    type: http
    endpoint: https://api.example.com
"""
    
    module = Parser(source).parse()
    app = module.body[0]
    
    output_dir = "/tmp/test_tool_errors"
    generate_backend(app, output_dir)
    
    runtime_path = os.path.join(output_dir, "generated", "runtime.py")
    with open(runtime_path, 'r') as f:
        runtime_code = f.read()
    
    # Verify error handling is present
    assert "try:" in runtime_code
    assert "except Exception as e:" in runtime_code
    assert "logger.warning" in runtime_code
    assert "Failed to initialize tool" in runtime_code


if __name__ == "__main__":
    test_tool_block_compilation()
    test_tool_registry_structure()
    test_multiple_tool_types()
    test_tool_initialization_error_handling()
    print("All tool integration tests passed!")
