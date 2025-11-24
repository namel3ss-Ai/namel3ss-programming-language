# Namel3ss Security Module

Production-grade security and capability model for the Namel3ss AI programming language.

## Quick Start

### Basic Security Configuration

Create a `namel3ss.toml` in your project root:

```toml
[security]
default_environment = "development"

[security.global_policy]
rate_limit_requests_per_minute = 60
max_tokens_per_request = 8000
```

### Define Agent with Capabilities

```python
from namel3ss.ast.agents import AgentDefinition
from namel3ss.ast.security import PermissionLevel

agent = AgentDefinition(
    name="researcher",
    llm_name="gpt-4",
    capabilities=["http_read", "database_read"],
    permission_level=PermissionLevel.NETWORK,
    tool_names=["web_search", "db_query"]
)
```

### Define Tool with Requirements

```python
from namel3ss.ast.ai_tools import ToolDefinition

tool = ToolDefinition(
    name="web_search",
    description="Search the web",
    required_capabilities=["http_read"],
    permission_level=PermissionLevel.NETWORK,
    rate_limit_per_minute=10,
    timeout_seconds=30.0
)
```

### Validate Security

```python
from namel3ss.security.validation import validate_tool_access

result = validate_tool_access(agent, tool, app)

if not result.allowed:
    print(f"Security violation: {result.reason}")
    for violation in result.violations:
        print(f"  - {violation}")
```

### Runtime Enforcement

```python
from namel3ss.security.runtime import get_security_guard

guard = get_security_guard()

# Before tool invocation
result = guard.check_tool_invocation("researcher", "web_search")
if not result.allowed:
    raise CapabilityDenied(result.reason)

# Execute tool...

# After completion
guard.record_tool_completion("researcher", "web_search", success=True)
```

## Architecture

The security model has three layers:

1. **Data Model** (`namel3ss/ast/security.py`): Core security data structures
2. **Static Validation** (`namel3ss/security/validation.py`): Compile-time checks
3. **Runtime Enforcement** (`namel3ss/security/runtime.py`): Execution-time guards

## Capability Model

Agents must explicitly declare capabilities. Tools declare required capabilities. The system validates at compile-time and enforces at runtime.

### Capability Types

- `HTTP_READ`: Read-only HTTP requests
- `HTTP_WRITE`: HTTP requests that modify state (POST, PUT, DELETE)
- `DATABASE_READ`: Read database queries
- `DATABASE_WRITE`: Write database operations
- `FILESYSTEM_READ`: Read files
- `FILESYSTEM_WRITE`: Write/delete files
- `CODE_EXECUTION`: Execute arbitrary code
- `NETWORK_ACCESS`: General network access
- `ADMIN`: Administrative operations

### Permission Levels

Permission levels form a hierarchy:

```
NONE < READ_ONLY < READ_WRITE < NETWORK/FILESYSTEM < ADMIN < UNRESTRICTED
```

Agents must have permission level ≥ tool requirement.

## Environment Profiles

Four standard environments with different security policies:

- **Development**: Permissive, warnings only
- **Staging**: Moderate restrictions, rate limits enforced
- **Production**: Strict, limited capabilities, HTTPS required
- **Sandbox**: Isolated, no network/filesystem access

Switch environments:

```bash
export NAMEL3SS_ENV=production
namel3ss build --env production
```

## Security Policies

Policies define quantitative constraints:

- **Rate Limits**: Requests per minute/hour, scoped by agent/tool/global
- **Timeouts**: Tool and LLM operation timeouts
- **Token Limits**: Per-request, per-agent, and global token limits
- **Cost Limits**: Per-request, per-agent, and global cost budgets
- **Concurrency**: Max concurrent tool/LLM calls

Example policy:

```toml
[security.policies.strict]
rate_limit_requests_per_minute = 10
rate_limit_requests_per_hour = 100
tool_timeout_seconds = 30.0
llm_timeout_seconds = 120.0
max_tokens_per_request = 4000
max_tokens_per_agent = 100000
max_cost_per_request = 0.50
max_cost_per_agent = 10.00
max_concurrent_tool_calls = 5
```

## Testing

Run security tests:

```bash
pytest tests/security/ -v
```

Test coverage:
- 24 validation tests
- 33 runtime enforcement tests
- 57 total tests, all passing

## Integration

### Resolver Integration

The resolver calls validation functions during static analysis:

```python
from namel3ss.security.validation import validate_application_security

result = validate_application_security(app, config)
if not result.allowed:
    raise SecurityViolation(result.violations)
```

### Runtime Integration

The runtime embeds security guard checks:

```python
# Before tool call
check_result = security_guard.check_tool_invocation(agent_name, tool_name)
if not check_result.allowed:
    raise CapabilityDenied(check_result.reason)

# Execute tool
tool_result = execute_tool(...)

# Record completion
security_guard.record_tool_completion(agent_name, tool_name, success=True)
```

## Documentation

- **Architecture**: `docs/SECURITY_ARCHITECTURE.md`
- **Implementation Summary**: `SECURITY_IMPLEMENTATION_SUMMARY.md`
- **API Reference**: See docstrings in source files

## Provider Neutrality

The security model is completely provider-neutral:
- Works with any LLM provider (OpenAI, Anthropic, local models)
- Works with any database (PostgreSQL, MongoDB, DuckDB)
- Works with any HTTP client
- Abstract capability types that map to any concrete implementation

## License

See LICENSE file in project root.
