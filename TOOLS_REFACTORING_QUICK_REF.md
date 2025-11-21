# Tools Subsystem Refactoring - Quick Reference

**Status:** ✅ Complete | **Date:** November 20, 2025

## Import Cheat Sheet

### New Recommended Imports
```python
# Core functionality
from namel3ss.tools import create_tool, get_registry

# Validation
from namel3ss.tools import validate_tool_config, validate_tool_name

# Errors
from namel3ss.tools import ToolValidationError, ToolRegistrationError, ToolExecutionError

# Types
from namel3ss.tools import BaseTool, ToolResult
```

### Complete Import (All Validators)
```python
from namel3ss.tools import (
    # Core
    BaseTool, ToolResult, ToolError,
    create_tool, get_registry, reset_registry,
    
    # Errors
    ToolValidationError,
    ToolRegistrationError,
    ToolExecutionError,
    
    # Validation
    validate_tool_config,
    validate_tool_name,
    validate_tool_type,
    validate_timeout,
    validate_schema,
    validate_http_method,
    validate_http_endpoint,
    validate_http_headers,
    validate_python_code,
    validate_tool_instance,
    validate_execution_inputs,
)
```

## Quick Start Examples

### Create HTTP Tool
```python
from namel3ss.tools import create_tool

tool = create_tool(
    name="weather_api",
    tool_type="http",
    endpoint="https://api.weather.com/v1/current",
    method="GET",
    headers={"API-Key": "your-key"},
    timeout=10.0,
    register=True  # Register in global registry
)

result = tool.execute(query={"location": "NYC", "units": "metric"})
if result.success:
    print(f"Temperature: {result.output['temp']}°C")
else:
    print(f"Error: {result.error}")
```

### Create Python Tool
```python
def calculate_tax(amount, rate):
    return amount * rate

tool = create_tool(
    name="tax_calculator",
    tool_type="python",
    function=calculate_tax,
    input_schema={
        "amount": {"type": "number", "required": True},
        "rate": {"type": "number", "required": True}
    },
    register=True
)

result = tool.execute(amount=100, rate=0.08)
print(result.output)  # 8.0
```

### Validation Example
```python
from namel3ss.tools import validate_tool_config
from namel3ss.tools import ToolValidationError

try:
    validate_tool_config(
        name="api",
        tool_type="http",
        endpoint="https://api.example.com",
        method="GET",
        timeout=30.0
    )
except ToolValidationError as e:
    print(f"[{e.code}] {e.message}")
    print(f"Field: {e.field}")
```

## Error Handling Patterns

### Handle Tool Execution Errors
```python
result = tool.execute(**inputs)

if result.success:
    # Process output
    data = result.output
    print(f"Success: {data}")
else:
    # Handle error
    print(f"Error: {result.error}")
    
    # Check if retryable
    if result.metadata.get("retryable"):
        print("This error can be retried")
    
    # Check status code for HTTP tools
    status_code = result.metadata.get("status_code")
    if status_code and status_code >= 500:
        print("Server error - retry later")
```

### Catch Validation Errors
```python
from namel3ss.tools import create_tool, ToolValidationError

try:
    tool = create_tool(
        name="",  # Invalid: empty name
        tool_type="http",
        endpoint="https://api.com"
    )
except ToolValidationError as e:
    print(f"Validation failed: {e.format()}")
    # Output:
    # [TOOL001] Tool Validation Error
    # Tool: ''
    # Field: name
    # Message: Tool name cannot be empty
```

### Catch Registration Errors
```python
from namel3ss.tools import get_registry, ToolRegistrationError

registry = get_registry()

try:
    # Trying to register duplicate tool
    registry.register("existing_tool", tool)
except ToolRegistrationError as e:
    print(f"[{e.code}] {e.message}")
    print(f"Conflict: {e.conflict}")
```

## Error Codes Reference

| Code | Category | Description |
|------|----------|-------------|
| TOOL001 | Validation | Tool name cannot be empty |
| TOOL002 | Validation | Tool name must be a string |
| TOOL003 | Validation | Tool name too long (>200 chars) |
| TOOL004 | Validation | Tool type cannot be empty |
| TOOL005 | Validation | Tool type must be a string |
| TOOL006 | Validation | Timeout must be a number |
| TOOL007 | Validation | Timeout must be positive |
| TOOL008 | Validation | Timeout too long (>3600s) |
| TOOL009 | Validation | Schema must be a dictionary |
| TOOL010 | Validation | Schema field name must be string |
| TOOL011 | Validation | Unknown schema keys |
| TOOL012 | Validation | Invalid HTTP method |
| TOOL013 | Validation | HTTP endpoint cannot be empty |
| TOOL014 | Validation | HTTP endpoint must be string |
| TOOL015 | Validation | HTTP endpoint must start with http:// or https:// |
| TOOL016 | Validation | HTTP headers must be dictionary |
| TOOL017 | Validation | Header name must be string |
| TOOL018 | Validation | Header value must be string |
| TOOL019 | Validation | Python code cannot be empty |
| TOOL020 | Validation | Python code must be string |
| TOOL021 | Validation | Not a BaseTool instance |
| TOOL022 | Validation | Tool missing required attribute |
| TOOL023 | Validation | Execution inputs must be dictionary |
| TOOL024 | Validation | Missing required input field |
| TOOL025 | Validation | Python tool requires function or code |
| TOOL026 | Registration | Tool already registered |
| TOOL031 | Execution | HTTP tool execution failed |
| TOOL032 | Execution | No function or code to execute |

## Validation Functions Summary

| Function | Validates | Raises |
|----------|-----------|--------|
| `validate_tool_name()` | Non-empty tool name, max 200 chars | TOOL001-003 |
| `validate_tool_type()` | Non-empty tool type string | TOOL004-005 |
| `validate_timeout()` | Positive number, max 3600s | TOOL006-008 |
| `validate_schema()` | Dictionary with valid field specs | TOOL009-011 |
| `validate_http_method()` | GET, POST, PUT, DELETE, PATCH | TOOL012 |
| `validate_http_endpoint()` | Valid URL starting with http(s):// | TOOL013-015 |
| `validate_http_headers()` | Dictionary of string key-value pairs | TOOL016-018 |
| `validate_python_code()` | Non-empty code string | TOOL019-020 |
| `validate_tool_instance()` | Object is BaseTool subclass | TOOL021-022 |
| `validate_execution_inputs()` | Inputs match schema | TOOL023-024 |
| `validate_tool_config()` | Complete tool configuration | All above |

## Module Structure

```
namel3ss/tools/
├── __init__.py          # Public API exports
├── base.py              # BaseTool, ToolResult, ToolError
├── errors.py            # ToolValidationError, ToolRegistrationError, ToolExecutionError
├── validation.py        # 11 validation functions
├── registry.py          # ToolRegistry, global registry
├── factory.py           # create_tool(), provider registration
├── http_tool.py         # HttpTool implementation
└── python_tool.py       # PythonTool implementation
```

## Common Patterns

### Pattern 1: Validate Before Creating
```python
from namel3ss.tools import validate_tool_config, create_tool

# Validate configuration first
try:
    validate_tool_config(
        name="api",
        tool_type="http",
        endpoint="https://api.com",
        method="GET"
    )
except ToolValidationError as e:
    print(f"Invalid config: {e.format()}")
    exit(1)

# Now create tool (will succeed)
tool = create_tool(name="api", tool_type="http", endpoint="https://api.com", method="GET")
```

### Pattern 2: Registry Management
```python
from namel3ss.tools import create_tool, get_registry

# Create and register tools during initialization
tools = [
    ("weather", "http", "https://api.weather.com"),
    ("news", "http", "https://api.news.com"),
]

for name, tool_type, endpoint in tools:
    create_tool(
        name=name,
        tool_type=tool_type,
        endpoint=endpoint,
        register=True  # Register automatically
    )

# Later, retrieve from registry
registry = get_registry()
weather_tool = registry.get("weather")
result = weather_tool.execute(location="NYC")
```

### Pattern 3: Error-Tolerant Execution
```python
def execute_with_retry(tool, max_retries=3, **inputs):
    """Execute tool with automatic retry on transient errors."""
    from namel3ss.tools import ToolExecutionError
    import time
    
    for attempt in range(max_retries):
        result = tool.execute(**inputs)
        
        if result.success:
            return result
        
        # Check if error is retryable
        if result.metadata.get("retryable"):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        # Non-retryable error or final attempt
        return result
    
    return result

# Usage
result = execute_with_retry(http_tool, location="NYC")
```

## Tips & Best Practices

### Security
- ✅ Never execute untrusted Python code strings
- ✅ Validate all inputs before execution
- ✅ Use environment variables for API keys
- ✅ Set appropriate timeouts to prevent hangs
- ✅ Sanitize user inputs in tool parameters

### Performance
- ✅ Register tools once during initialization
- ✅ Reuse tool instances, don't recreate
- ✅ Set appropriate timeouts (shorter for fast APIs)
- ✅ Use caching for frequently called tools
- ✅ Consider connection pooling for HTTP tools

### Error Handling
- ✅ Always check `result.success` before using output
- ✅ Handle both expected (success=False) and unexpected (exceptions) errors
- ✅ Check `metadata.retryable` for retry logic
- ✅ Log error codes and messages for debugging
- ✅ Use specific error types (ToolValidationError, etc.)

### Validation
- ✅ Use `validate_tool_config()` before creating tools
- ✅ Define input/output schemas for type safety
- ✅ Validate execution inputs with `validate_execution_inputs()`
- ✅ Use tool-specific validators (validate_http_*, etc.)
- ✅ Catch ToolValidationError for graceful handling

## Complete Working Example

```python
from namel3ss.tools import (
    create_tool,
    get_registry,
    validate_tool_config,
    ToolValidationError,
)

# 1. Validate configuration
try:
    validate_tool_config(
        name="weather_api",
        tool_type="http",
        endpoint="https://api.openweathermap.org/data/2.5/weather",
        method="GET",
        timeout=10.0
    )
    print("✓ Configuration valid")
except ToolValidationError as e:
    print(f"✗ Configuration invalid: {e.format()}")
    exit(1)

# 2. Create and register tool
weather_tool = create_tool(
    name="weather_api",
    tool_type="http",
    endpoint="https://api.openweathermap.org/data/2.5/weather",
    method="GET",
    headers={"Accept": "application/json"},
    timeout=10.0,
    register=True,
    input_schema={
        "q": {"type": "string", "required": True},
        "appid": {"type": "string", "required": True},
        "units": {"type": "string", "required": False}
    }
)

# 3. Execute tool
result = weather_tool.execute(
    query={
        "q": "New York",
        "appid": "your_api_key",
        "units": "metric"
    }
)

# 4. Handle result
if result.success:
    data = result.output
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Conditions: {data['weather'][0]['description']}")
    print(f"Status Code: {result.metadata['status_code']}")
else:
    print(f"Error: {result.error}")
    if result.metadata.get("retryable"):
        print("This error can be retried")

# 5. Retrieve from registry later
registry = get_registry()
same_tool = registry.get("weather_api")
print(f"Tool from registry: {same_tool.name}")
```

## Migration Notes

### Changes from Previous Version
- ✅ Added `errors.py` with 3 exception types
- ✅ Added `validation.py` with 11 validators  
- ✅ Enhanced all docstrings with examples
- ✅ Added validation calls to registry and factory
- ✅ Improved error messages with error codes
- ✅ Added retryable flag to execution errors
- ✅ Exported validation and error types in `__init__.py`

### Backward Compatibility
- ✅ All existing imports still work
- ✅ `create_tool()` signature unchanged (validation added internally)
- ✅ `ToolError` still base exception
- ✅ No breaking changes to public API

---

**Next Steps:**
1. Use new validation functions in tool creation code
2. Catch specific error types (ToolValidationError, etc.)
3. Add error code handling in error logging
4. Consider adding custom tool types with `register_provider()`

**Questions?** Check comprehensive documentation in `TOOLS_REFACTORING_COMPLETE.md`
