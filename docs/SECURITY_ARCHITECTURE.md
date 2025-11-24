# Namel3ss Security Architecture

**Status**: Implementation in Progress  
**Version**: 1.0.0  
**Last Updated**: 2025-11-24

## Overview

This document defines the comprehensive security and capability model for Namel3ss, the AI programming language. The security model is:

- **Language-level**: Built into the AST, typechecker, and runtime
- **Enforceable**: Statically validated and dynamically guarded
- **Provider-neutral**: Works regardless of LLM/DB/HTTP provider
- **Production-ready**: Designed for real-world deployment scenarios

## Core Principles

### 1. **Explicit is Better Than Implicit**
- Tools must be explicitly declared in `.ai` source files
- Agents must explicitly list which tools they can access
- No ambient authority or implicit capability grants

### 2. **Fail Secure by Default**
- Undeclared tools cannot be invoked
- Agents without tool grants cannot call those tools
- Policy violations result in immediate denial

### 3. **Layered Defense**
- **Static layer**: Typechecker validates capabilities at compile time
- **Runtime layer**: Security guards enforce policies at execution time
- **Configuration layer**: Environment profiles provide deployment-specific controls

### 4. **Principle of Least Privilege**
- Agents start with no capabilities
- Must explicitly request each tool/permission
- Environment can further restrict based on deployment context

---

## Security Model Components

### 1. Capability Model

A **capability** represents a granted permission for an agent to perform specific actions. Capabilities are:

- **Declarative**: Defined in `.ai` source code
- **Scoped**: Limited to specific tools or categories
- **Checkable**: Validated at compile and runtime

#### Capability Types

```
enum CapabilityType {
    TOOL_ACCESS,        // Can invoke specific tool
    TOOL_CATEGORY,      // Can invoke tool category (http, db, fs)
    MEMORY_READ,        // Can read from memory scope
    MEMORY_WRITE,       // Can write to memory scope
    NETWORK,            // Can make network calls
    FILESYSTEM,         // Can access filesystem
    DATABASE            // Can query databases
}
```

#### Example Capability Declaration

```namel3ss
// In .ai source
agent researcher {
    llm: gpt_4o
    tools: [web_search, calculator]  // Explicit tool grants
    capabilities: [NETWORK, TOOL_CATEGORY:http_readonly]
    memory: conversation
    goal: "Research and summarize topics"
}
```

### 2. Permission Levels

Tools are classified by **permission level**, representing their potential impact:

```
enum PermissionLevel {
    READ_ONLY,      // Read-only operations (safe)
    READ_WRITE,     // Can modify data
    ADMIN,          // Administrative operations
    NETWORK,        // Makes external network calls
    FILESYSTEM,     // Accesses local filesystem
    UNRESTRICTED    // No restrictions (dangerous)
}
```

#### Permission Hierarchy

```
READ_ONLY < READ_WRITE < ADMIN < UNRESTRICTED
         < NETWORK
         < FILESYSTEM
```

Agents must be granted permission levels >= the tools they invoke.

#### Tool Permission Declaration

```namel3ss
tool web_search {
    description: "Search the web for information"
    permission_level: NETWORK
    required_capabilities: [NETWORK, TOOL_CATEGORY:http]
    
    parameters: {
        query: {type: "string", required: true}
    }
    
    implementation: {
        endpoint: "https://api.search.example.com",
        method: "GET",
        auth: "bearer_token"
    }
}
```

### 3. Environment Profiles

An **environment profile** defines deployment-specific security policies:

```
enum Environment {
    DEVELOPMENT,    // Loose restrictions, verbose warnings
    STAGING,        // Moderate restrictions, realistic testing
    PRODUCTION,     // Strict restrictions, minimal attack surface
    SANDBOX         // Isolated, no external access
}
```

#### Environment Configuration

In `namel3ss.toml`:

```toml
[security]
default_environment = "development"

[security.environments.development]
allowed_permission_levels = ["READ_ONLY", "READ_WRITE", "NETWORK"]
allowed_tool_categories = ["http", "db", "vector"]
warn_on_elevated_permissions = true
enforce_rate_limits = false

[security.environments.production]
allowed_permission_levels = ["READ_ONLY", "NETWORK"]
allowed_tool_categories = ["http", "db"]
deny_filesystem_access = true
enforce_rate_limits = true
enforce_strict_timeouts = true
require_audit_logging = true
```

### 4. Security Policies

Policies define **quantitative constraints** on execution:

#### Policy Schema

```python
@dataclass
class SecurityPolicy:
    """Configurable security constraints."""
    
    # Rate limiting
    rate_limit_requests_per_minute: Optional[int] = None
    rate_limit_requests_per_hour: Optional[int] = None
    rate_limit_scope: str = "agent"  # agent, tool, global
    
    # Timeouts
    tool_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 60.0
    total_execution_timeout_seconds: Optional[float] = None
    
    # Token limits
    max_tokens_per_request: Optional[int] = None
    max_tokens_per_agent: Optional[int] = None
    max_total_tokens: Optional[int] = None
    
    # Cost controls
    max_cost_per_request: Optional[float] = None
    max_cost_per_agent: Optional[float] = None
    
    # Concurrency
    max_concurrent_tool_calls: int = 10
    max_concurrent_llm_calls: int = 5
```

#### Policy Stacking

Policies stack in order of specificity:

1. **Global policy**: Applied to all agents/tools
2. **Environment policy**: Overrides global for current environment
3. **Agent policy**: Overrides environment for specific agent
4. **Tool policy**: Overrides agent for specific tool

#### Example Policy Declaration

```namel3ss
// Global policy
policy default_limits {
    rate_limit_requests_per_minute: 60
    tool_timeout_seconds: 30.0
    llm_timeout_seconds: 60.0
    max_tokens_per_request: 4000
}

// Agent-specific override
agent high_volume_processor {
    llm: gpt_4o_mini
    tools: [batch_processor]
    
    policy: {
        rate_limit_requests_per_minute: 300  // Override global
        max_concurrent_tool_calls: 50
    }
}
```

---

## Enforcement Architecture

### Static Enforcement (Compile Time)

The **resolver** validates security constraints:

1. **Tool Declaration Check**
   - Agents can only reference declared tools
   - Compile error: "Tool 'X' not declared in application"

2. **Capability Validation**
   - Agents' tool lists match their declared capabilities
   - Compile error: "Agent 'A' lacks capability for tool 'T'"

3. **Permission Level Check**
   - Agent permission level >= tool permission level
   - Compile error: "Agent 'A' permission 'READ_ONLY' insufficient for tool 'T' (requires 'NETWORK')"

4. **Policy Validation**
   - Policy values are within valid ranges
   - Compile error: "Invalid policy: negative timeout value"

### Runtime Enforcement (Execution Time)

The **SecurityGuard** enforces policies at runtime:

```python
class SecurityGuard:
    """Runtime security enforcement layer."""
    
    def check_tool_invocation(
        self,
        agent: str,
        tool: str,
        context: ExecutionContext
    ) -> SecurityCheckResult:
        """Validate tool invocation is allowed."""
        # 1. Check tool is declared
        # 2. Check agent has tool in allowed list
        # 3. Check capability grants
        # 4. Check permission levels
        # 5. Check environment restrictions
        # 6. Check rate limits
        # 7. Record audit event
        
    def check_llm_invocation(
        self,
        agent: str,
        model: str,
        tokens: int,
        context: ExecutionContext
    ) -> SecurityCheckResult:
        """Validate LLM call is allowed."""
        # 1. Check token limits
        # 2. Check rate limits
        # 3. Check cost limits (if configured)
        # 4. Check timeout constraints
        # 5. Record audit event
```

#### Enforcement Points

1. **Before tool call**: `SecurityGuard.check_tool_invocation()`
2. **Before LLM call**: `SecurityGuard.check_llm_invocation()`
3. **During execution**: Timeout enforcement via asyncio
4. **After completion**: Token/cost accounting

---

## Configuration System

### Project Configuration (`namel3ss.toml`)

```toml
[security]
default_environment = "development"
audit_log_path = "logs/security_audit.log"
fail_mode = "closed"  # closed = deny on error, open = allow on error

# Global policies
[security.global_policy]
rate_limit_requests_per_minute = 100
tool_timeout_seconds = 30.0
llm_timeout_seconds = 120.0
max_tokens_per_request = 8000

# Development environment
[security.environments.development]
allowed_permission_levels = ["READ_ONLY", "READ_WRITE", "NETWORK", "FILESYSTEM"]
allowed_tool_categories = ["http", "db", "vector", "fs"]
enforce_rate_limits = false
warn_on_policy_violations = true

# Production environment
[security.environments.production]
allowed_permission_levels = ["READ_ONLY", "NETWORK"]
allowed_tool_categories = ["http", "db", "vector"]
deny_filesystem_access = true
enforce_rate_limits = true
enforce_strict_timeouts = true
require_https = true
max_concurrent_agents = 20

[security.audit]
enabled = true
log_level = "info"  # debug, info, warn, error
log_tool_calls = true
log_llm_calls = true
log_policy_violations = true
```

### Environment Selection

```bash
# CLI flag
namel3ss build app.ai --env production

# Environment variable
export NAMEL3SS_ENV=production
namel3ss run app.ai

# In code (for testing)
from namel3ss.security import set_environment
set_environment("sandbox")
```

---

## Data Flow

### 1. Development → Compilation

```
.ai source file
    ↓
Parser → AST with security annotations
    ↓
Resolver validates:
  - Tool declarations
  - Agent capabilities
  - Permission levels
  - Policy constraints
    ↓
IR with security metadata
    ↓
Codegen embeds security checks
```

### 2. Runtime Execution

```
Agent invokes tool
    ↓
SecurityGuard.check_tool_invocation()
    - Verify capability grant
    - Check environment restrictions
    - Enforce rate limits
    - Check timeouts
    ↓
[ALLOWED] → Execute tool → Record audit
[DENIED]  → Raise SecurityViolation error
```

---

## Error Handling

### Compile-Time Errors

```
Error: Agent 'researcher' cannot access tool 'file_writer'
  → Tool 'file_writer' not in agent's tools list
  Location: app.ai:45:12

Error: Tool 'admin_panel' requires permission level ADMIN
  → Agent 'user_assistant' has permission level READ_ONLY
  Location: app.ai:52:8

Error: Invalid security policy: negative timeout
  → tool_timeout_seconds: -5.0 (must be positive)
  Location: app.ai:30:5
```

### Runtime Errors

```python
class SecurityViolation(Exception):
    """Raised when security constraint is violated at runtime."""
    
class RateLimitExceeded(SecurityViolation):
    """Rate limit exceeded."""
    
class TimeoutExceeded(SecurityViolation):
    """Operation timeout exceeded."""
    
class TokenLimitExceeded(SecurityViolation):
    """Token limit exceeded."""
    
class CapabilityDenied(SecurityViolation):
    """Required capability not granted."""
```

---

## Best Practices

### For Development

1. **Use explicit tool grants**: Always list tools in agent definitions
2. **Start restrictive**: Begin with READ_ONLY, escalate as needed
3. **Test security**: Run `namel3ss security check` before deployment
4. **Review audit logs**: Understand actual tool/LLM usage patterns

### For Staging

1. **Mirror production constraints**: Use production-like policies
2. **Enable audit logging**: Capture all security events
3. **Test rate limits**: Verify limits don't block legitimate traffic
4. **Validate timeouts**: Ensure realistic timeout values

### For Production

1. **Principle of least privilege**: Grant minimum necessary capabilities
2. **Enable strict mode**: Set `fail_mode = "closed"`
3. **Monitor audit logs**: Watch for suspicious patterns
4. **Set conservative limits**: Prefer safety over convenience
5. **Use HTTPS only**: Enforce secure transport for all external calls

---

## Future Enhancements

1. **Dynamic capability grants**: Runtime-evaluated conditions for capabilities
2. **Policy inheritance**: Hierarchical policy composition
3. **Custom validators**: User-defined security constraint validators
4. **Security profiles**: Pre-packaged security configurations (e.g., "HIPAA", "PCI-DSS")
5. **Threat modeling integration**: Automated threat analysis of agent graphs
6. **Cost prediction**: Estimate costs before execution based on policies

---

## References

- **AST Nodes**: `namel3ss/ast/security.py`
- **Configuration**: `namel3ss/security/config.py`
- **Validation**: `namel3ss/security/validation.py`
- **Runtime**: `namel3ss/security/runtime.py`
- **Tests**: `tests/security/`
- **Examples**: `examples/security/`
