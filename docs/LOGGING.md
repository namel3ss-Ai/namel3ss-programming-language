# Namel3ss Logging System

**Version:** 1.0.0  
**Date:** November 23, 2025  
**Status:** Complete

This document describes the comprehensive logging system for the Namel3ss programming language, providing runtime observability and debugging capabilities for AI applications.

---

## Overview

The Namel3ss logging system provides **production-ready logging statements** that can be embedded directly within pages and application logic. These statements compile to efficient runtime logging that integrates with the observability infrastructure.

### Key Features

- **First-class language support**: `log` statements as native language constructs
- **Multiple log levels**: `debug`, `info`, `warn`, `error` with CLI configuration
- **Template integration**: Log messages support template variables and context
- **CLI configuration**: `--log-level` flag and `NAMEL3SS_LOG_LEVEL` environment variable
- **Production ready**: Structured logging with JSON formatting and metadata
- **Zero overhead**: Compile-time optimization with configurable runtime filtering

---

## Log Statement Syntax

### Basic Syntax

```text
log "message"                    # INFO level (default)
log info "informational message"  # Explicit INFO level
log warn "warning message"       # WARNING level
log error "error occurred"       # ERROR level  
log debug "debugging info"       # DEBUG level
```

### Grammar Definition

```ebnf
LogStatement = "log" , [ LogLevel ] , STRING_LITERAL , "\n" ;
LogLevel     = "debug" | "info" | "warn" | "error" ;
```

### Template Variables

Log messages support full template variable interpolation:

```text
page "User Profile" at "/user/{user_id}":
  log info "Loading profile for user {user_id}"
  
  let user_data = dataset users where id == user_id
  
  if user_data.count == 0:
    log warn "User {user_id} not found in database"
    show text "User not found"
  else:
    log debug "Found user data: {user_data.name}"
    show text "Welcome, {user_data.name}!"
```

---

## Log Levels

### DEBUG
- **Purpose**: Detailed diagnostic information for development and troubleshooting
- **Visibility**: Only shown when explicitly enabled with `--log-level debug`
- **Usage**: Variable values, flow control decisions, internal state

```text
log debug "Processing {record_count} records from dataset"
log debug "Cache hit for key '{cache_key}'"
log debug "Agent response: {agent.last_response}"
```

### INFO
- **Purpose**: General informational messages about application flow
- **Visibility**: Default level, shown unless explicitly suppressed
- **Usage**: Application milestones, user actions, successful operations

```text
log info "Page loaded successfully" 
log info "User {user_id} logged in"
log info "Processing payment for order {order_id}"
```

### WARN
- **Purpose**: Warning conditions that don't prevent operation but may need attention
- **Visibility**: Always shown unless filtered to ERROR-only
- **Usage**: Deprecated features, fallback behavior, recoverable errors

```text
log warn "API rate limit approaching for provider {provider}"
log warn "Using fallback model due to primary model unavailability"
log warn "Dataset '{dataset_name}' has no records matching filter"
```

### ERROR
- **Purpose**: Error conditions that indicate problems requiring attention
- **Visibility**: Always shown (highest priority)
- **Usage**: Failed operations, exceptions, data consistency issues

```text
log error "Failed to connect to database: {error_message}"
log error "Invalid API key for provider {provider}"
log error "Memory allocation failed for conversation {conversation_id}"
```

---

## CLI Configuration

### Command Line Flag

Use the `--log-level` flag to set the minimum log level for runtime execution:

```bash
namel3ss run app.ai --log-level debug    # Show all log levels
namel3ss run app.ai --log-level info     # Show info, warn, error (default)
namel3ss run app.ai --log-level warn     # Show warn, error only
namel3ss run app.ai --log-level error    # Show error only
```

### Environment Variable

Set the `NAMEL3SS_LOG_LEVEL` environment variable for persistent configuration:

```bash
export NAMEL3SS_LOG_LEVEL=debug
namel3ss run app.ai

# Or inline
NAMEL3SS_LOG_LEVEL=warn namel3ss run app.ai
```

### Configuration Precedence

1. CLI `--log-level` flag (highest priority)
2. `NAMEL3SS_LOG_LEVEL` environment variable
3. Default: `info`

---

## Runtime Integration

### Logger Configuration

The logging system uses Python's standard `logging` module with the logger name `namel3ss.runtime`:

```python
import logging

# The Namel3ss runtime logger
logger = logging.getLogger('namel3ss.runtime')

# Example output format:
# 2025-11-23 20:50:36,259 - namel3ss.runtime - INFO - User 12345 logged in
```

### Structured Output

Log messages include structured metadata for observability:

- **Timestamp**: ISO 8601 formatted timestamp
- **Logger name**: Always `namel3ss.runtime`
- **Level**: DEBUG, INFO, WARNING, ERROR
- **Message**: Template-rendered message text
- **Source location**: File and line number (when available)

### Integration with Observability

Log statements integrate with the existing observability infrastructure:

```python
# Generated backend includes logging configuration
from namel3ss.codegen.backend.core.runtime_sections.observability import configure_logging

# Automatic setup in generated applications
configure_logging(
    level=log_level,
    format="json" if is_production else "console",
    include_request_context=True
)
```

---

## Best Practices

### 1. Use Appropriate Log Levels

```text
# ✅ Good: Appropriate levels for different scenarios
log debug "Entering function processPayment with amount {amount}"
log info "Payment processed successfully for order {order_id}"
log warn "Payment provider responded slowly ({response_time}ms)"
log error "Payment failed: {error_details}"

# ❌ Bad: Wrong levels
log error "User clicked button"  # Should be DEBUG or INFO
log info "Database connection failed"  # Should be ERROR
```

### 2. Include Relevant Context

```text
# ✅ Good: Rich context for debugging
log info "User {user_id} updated profile: {changed_fields}"
log error "Database query failed: {query} - Error: {db_error}"

# ❌ Bad: Vague or missing context  
log info "User updated profile"
log error "Database error"
```

### 3. Avoid Sensitive Information

```text
# ✅ Good: Safe logging
log info "Authentication successful for user {user_id}"
log debug "API request to {endpoint} completed"

# ❌ Bad: Exposes sensitive data
log info "User logged in with password {password}"
log debug "API key: {api_key}"
```

### 4. Use Templates Effectively

```text
# ✅ Good: Template variables for dynamic content
log info "Processing {batch_size} records from {dataset_name}"
log warn "Cache miss for key '{cache_key}', falling back to database"

# ❌ Bad: String concatenation (not supported)
log info "Processing " + batch_size + " records"  # Invalid syntax
```

### 5. Log State Transitions

```text
page "Checkout" at "/checkout":
  log info "Checkout process started for user {user_id}"
  
  if cart.items.count == 0:
    log warn "User {user_id} attempted checkout with empty cart"
    show text "Your cart is empty"
  else:
    log info "Processing {cart.items.count} items for user {user_id}"
    
    # ... checkout logic ...
    
    log info "Checkout completed successfully for order {order.id}"
```

---

## Advanced Usage

### Conditional Logging

Combine log statements with control flow for conditional logging:

```text
if debug_mode:
  log debug "Debug mode enabled, showing detailed information"
  log debug "Current memory usage: {memory.current_usage}"
  log debug "Active connections: {database.connection_count}"

# Error handling with logging
let result = call_external_api(endpoint)
if result.status == "error":
  log error "API call to {endpoint} failed: {result.error_message}"
  show toast "Service temporarily unavailable"
else:
  log info "API call to {endpoint} successful"
```

### Performance Monitoring

```text
# Time-based logging for performance monitoring
log info "Starting batch job for {record_count} records"
# ... processing logic ...
log info "Batch job completed in {processing_time}ms"

# Resource utilization
log debug "Memory usage: {memory.used}/{memory.total} MB"
log debug "Database connections: {db.active_connections}/{db.max_connections}"
```

### Error Recovery

```text
chain "DataProcessing":
  steps:
    - try:
        step "process_data":
          kind: python
          module: data.processor
      catch:
        log error "Data processing failed: {error.message}"
        log info "Attempting recovery with backup processor"
        step "backup_process":
          kind: python  
          module: data.backup_processor
```

---

## Examples

### Basic Web Application

```text
app "E-commerce Site" connects to postgres "SHOP_DB".

page "Product Detail" at "/product/{product_id}":
  log info "Loading product {product_id}"
  
  let product = dataset products where id == product_id
  
  if product.count == 0:
    log warn "Product {product_id} not found"
    show text "Product not found"
  else:
    log debug "Product data loaded: {product.name}"
    show text product.name
    show text product.description
    
    if product.stock_count < 5:
      log warn "Low stock for product {product_id}: {product.stock_count} remaining"
      show text "Only {product.stock_count} left in stock!"

page "Add to Cart" at "/cart/add":
  show form "Add to Cart":
    fields: product_id, quantity
    on submit:
      log info "Adding {form.quantity} of product {form.product_id} to cart"
      
      if form.quantity > 10:
        log warn "Large quantity requested: {form.quantity} for product {form.product_id}"
      
      # Add to cart logic
      log info "Successfully added {form.quantity} items to cart"
      show toast "Added to cart successfully!"
```

### AI Chain with Logging

```text
prompt "AnalyzeDocument":
  input:
    document: text
  output:
    summary: text
    keywords: list
  using model "gpt-4o":
    """
    Analyze this document and provide a summary with keywords.
    
    Document: {{document}}
    """

chain "DocumentAnalysis":
  steps:
    - log info "Starting document analysis for {document.title}"
    
    - if document.length > 10000:
        then:
          log warn "Large document detected ({document.length} chars), may take longer"
    
    - step "analyze":
        kind: prompt
        target: AnalyzeDocument
        
    - log debug "Analysis completed: {steps.analyze.result.keywords}"
    - log info "Document analysis finished for {document.title}"

page "Document Upload" at "/analyze":
  show form "Upload Document":
    fields: document_text
    on submit:
      log info "User initiated document analysis"
      run chain DocumentAnalysis with:
        document = form.document_text
      show text "Analysis: {chain_result.summary}"
```

---

## Implementation Details

### AST Representation

```python
from dataclasses import dataclass
from enum import Enum

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    
    def __str__(self) -> str:
        return self.value

@dataclass
class LogStatement:
    level: LogLevel
    message: str
    source_location: Optional[str] = None
```

### Backend Compilation

Log statements compile to runtime function calls:

```python
# Source: log info "User {user_id} logged in"
# Compiles to:
await _render_log_statement({
    "type": "log",
    "level": "info", 
    "message": "User {user_id} logged in",
    "source_location": "app.ai:15"
}, context, scope)
```

### Runtime Execution

```python
async def _render_log_statement(statement, context, scope):
    import logging
    from .template import _render_template_value
    
    level_str = statement.get("level", "info").lower()
    message_template = statement.get("message", "")
    
    # Render template variables
    message = _render_template_value(message_template, context)
    
    # Map to logging levels
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
    }
    
    log_level = level_map.get(level_str, logging.INFO)
    logger = logging.getLogger("namel3ss.runtime")
    logger.log(log_level, message)
```

---

## Testing

### Unit Tests

The logging system includes comprehensive test coverage:

```python
# Test AST creation
def test_log_statement_creation():
    stmt = LogStatement(LogLevel.INFO, "Test message")
    assert stmt.level == LogLevel.INFO
    assert stmt.message == "Test message"

# Test backend encoding  
def test_log_statement_encoding():
    stmt = LogStatement(LogLevel.ERROR, "Error occurred")
    component = _encode_statement(stmt, set(), {})
    assert component.type == "log"
    assert component.payload["level"] == "error"

# Test CLI configuration
def test_cli_log_configuration():
    args = MockArgs(log_level="debug")
    _configure_runtime_logging(args)
    logger = logging.getLogger('namel3ss.runtime')
    assert logger.level == logging.DEBUG
```

### Integration Testing

```bash
# Run logging tests
pytest tests/test_logging_statements.py -v

# Test CLI integration
namel3ss run test_app.ai --log-level debug 2>&1 | grep "namel3ss.runtime"
```

---

## Migration and Compatibility

### From Manual Logging

Replace manual logging calls with native log statements:

```text
# Before: Manual Python logging
def custom_handler():
    import logging
    logger = logging.getLogger("app")
    logger.info("Processing user request")

# After: Native log statements  
page "Handler" at "/process":
  log info "Processing user request"
```

### Backward Compatibility

- Log statements are a new feature with no breaking changes
- Existing applications continue to work without modification
- Optional adoption - can be added incrementally to pages and chains

### Performance Impact

- **Development**: Minimal impact, statements compile to efficient runtime calls
- **Production**: Configurable filtering at runtime, debug messages can be disabled
- **Overhead**: Comparable to standard Python logging with template rendering

---

## Troubleshooting

### Common Issues

1. **Log messages not appearing**
   - Check log level configuration (`--log-level` or `NAMEL3SS_LOG_LEVEL`)
   - Verify logger is configured correctly in runtime environment

2. **Template variables not rendering**
   - Ensure variables exist in current scope and context
   - Check variable naming matches template syntax `{variable_name}`

3. **Performance impact**
   - Use appropriate log levels (avoid debug in production)
   - Configure runtime filtering to exclude unnecessary messages

### Debugging

```bash
# Enable all logging
export NAMEL3SS_LOG_LEVEL=debug
namel3ss run app.ai

# Check logging configuration
python -c "import logging; print(logging.getLogger('namel3ss.runtime').level)"

# Test with specific log level
namel3ss run app.ai --log-level info --verbose
```

---

## Changelog

### v1.0.0 (November 23, 2025)
- Initial implementation of log statements
- Support for debug, info, warn, error levels
- CLI configuration with `--log-level` flag
- Environment variable support (`NAMEL3SS_LOG_LEVEL`)
- Template variable integration
- Comprehensive test suite
- Production-ready runtime integration

---

For more information about Namel3ss language features, see:
- [Grammar Specification](GRAMMAR.md)
- [Template Engine](TEMPLATE_ENGINE.md)  
- [Runtime System](RUNTIME.md)
- [CLI Documentation](CLI.md)