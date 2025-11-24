# Namel3ss Debugging & Observability System

A comprehensive debugging and observability framework for namel3ss applications, providing execution tracing, deterministic replay, and step-by-step debugging capabilities.

## Overview

The namel3ss debugging system provides first-class debugging support for AI applications with:

- **Execution Tracing**: Capture detailed execution events across agents, prompts, chains, tools, and LLM calls
- **Deterministic Replay**: Replay previous executions with mock responses for reproducible debugging  
- **Step-by-Step Debugging**: Interactive debugging with breakpoints and execution control
- **Performance Analysis**: Detailed performance profiling and resource usage monitoring
- **CLI Debug Tools**: Rich command-line interface for debugging workflows

## Quick Start

### 1. Enable Debug Tracing

```bash
# Enable debugging for all executions
export NAMEL3SS_DEBUG_ENABLED=true
export NAMEL3SS_DEBUG_OUTPUT_DIR=./debug/traces

# Run your application with tracing
namel3ss run my_app.ai
```

### 2. Trace a Specific Execution

```bash
# Trace execution with filtering
namel3ss debug trace my_app.ai --filter agent --filter llm --output trace.jsonl

# Trace with custom arguments
namel3ss debug trace my_app.ai --run-args '{"input": "Hello world"}' --memory --performance
```

### 3. Replay and Debug

```bash
# Full replay
namel3ss debug replay trace.jsonl

# Interactive step-by-step replay
namel3ss debug replay trace.jsonl --step

# Replay with breakpoints
namel3ss debug replay trace.jsonl --breakpoint agent_turn_start:3 --mock "MyAgent:{\"response\":\"test\"}"
```

### 4. Analyze Performance

```bash
# Performance analysis
namel3ss debug analyze trace.jsonl --performance --errors

# Inspect specific components
namel3ss debug inspect trace.jsonl --agent MyAgent
namel3ss debug inspect trace.jsonl --event 42
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NAMEL3SS_DEBUG_ENABLED` | Enable/disable debug tracing | `false` |
| `NAMEL3SS_DEBUG_OUTPUT_DIR` | Trace output directory | `./debug/traces` |
| `NAMEL3SS_DEBUG_COMPONENTS` | Filter by components (comma-separated) | All components |
| `NAMEL3SS_DEBUG_BUFFER_SIZE` | Event buffer size | `1000` |
| `NAMEL3SS_DEBUG_CAPTURE_MEMORY` | Capture memory usage | `true` |
| `NAMEL3SS_DEBUG_CAPTURE_PERFORMANCE` | Capture performance metrics | `true` |

### Workspace Configuration

Create `.namel3ss/debug.json` in your workspace:

```json
{
  "enabled": true,
  "auto_trace": false,
  "trace_output_dir": "./debug/traces",
  "trace_retention_days": 30,
  "default_components": ["agent", "llm"],
  "buffer_size": 1000,
  "capture_memory": true,
  "capture_performance": true,
  "max_payload_size": 16384
}
```

## Architecture

### Core Components

#### 1. TraceEvent Data Model

```python
from namel3ss.debugging import TraceEvent, TraceEventType

# Event types captured:
# - agent_execution_start/end
# - agent_turn_start/end  
# - llm_call_start/end
# - tool_call_start/end
# - prompt_execution_start/end
# - chain_execution_start/end
# - chain_step_start/end
# - error_occurred
# - memory_operation
# - validation_event

event = TraceEvent(
    event_type=TraceEventType.AGENT_EXECUTION_START,
    component="agent",
    component_name="MyAgent",
    inputs={"user_input": "Hello"},
    metadata={"turn": 1}
)
```

#### 2. ExecutionTracer

```python
from namel3ss.debugging import initialize_tracing, get_global_tracer

# Initialize tracing
tracer = initialize_tracing()

# Start execution trace
context = await tracer.start_execution_trace(app_name="MyApp")

# Emit events
await tracer.emit_event(
    TraceEventType.AGENT_EXECUTION_START,
    component="agent",
    component_name="MyAgent"
)

# End trace
trace_file = await tracer.end_execution_trace()
```

#### 3. ExecutionReplayer

```python
from namel3ss.debugging import ExecutionReplayer, ReplayBreakpoint

# Create replayer
replayer = ExecutionReplayer(
    trace_file=Path("trace.jsonl"),
    breakpoints=[
        ReplayBreakpoint(event_type=TraceEventType.AGENT_TURN_START, event_index=3)
    ]
)

# Step-by-step replay
while not replayer.state.completed:
    event = replayer.replay_step()
    print(f"Event: {event.event_type} - {event.component}")
```

### Integration with Runtime

#### 1. Agent Runtime Integration

```python
from namel3ss.debugging.hooks import trace_agent_execution

class AgentRuntime:
    @trace_agent_execution(capture_inputs=True, capture_outputs=True)
    async def aact(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        # Agent execution automatically traced
        pass
```

#### 2. Prompt Execution Integration

```python
from namel3ss.debugging.hooks import trace_prompt_execution

@trace_prompt_execution(capture_inputs=True, capture_outputs=True) 
async def execute_structured_prompt(prompt_def, llm, args):
    # Prompt execution automatically traced
    pass
```

#### 3. Manual Event Tracing

```python
from namel3ss.debugging.hooks import trace_agent_turn, trace_memory_operation

# Trace agent turns
async with trace_agent_turn("MyAgent", turn_number=1, user_input="Hello"):
    # Agent turn processing
    pass

# Trace memory operations
await trace_memory_operation(
    operation_type="summarize",
    agent_name="MyAgent", 
    details={"messages_before": 10, "messages_after": 5}
)
```

## Trace File Format

Trace files use JSON Lines format (`.jsonl`) with one event per line:

```json
{"event_type": "app_load_start", "event_id": "evt_001", "timestamp": 1234567890.123, "component": "app", "inputs": {"execution_id": "exec_001"}}
{"event_type": "agent_execution_start", "event_id": "evt_002", "timestamp": 1234567890.456, "component": "agent", "component_name": "MyAgent", "parent_event_id": "evt_001"}
{"event_type": "llm_call_start", "event_id": "evt_003", "timestamp": 1234567890.789, "component": "llm", "component_name": "openai", "parent_event_id": "evt_002"}
{"event_type": "llm_call_end", "event_id": "evt_004", "timestamp": 1234567891.012, "component": "llm", "duration_ms": 223.0, "status": "completed", "tokens_used": 150}
```

### Event Schema

Each trace event contains:

- **Identification**: `event_type`, `event_id`, `timestamp`, `parent_event_id`
- **Context**: `execution_id`, `component`, `component_name`  
- **Data**: `inputs`, `outputs`, `metadata`
- **Timing**: `duration_ms`, `status`, `error`
- **Resources**: `memory_usage_mb`, `tokens_used`, `cost_estimate`

## CLI Commands

### `namel3ss debug trace`

Trace execution of a namel3ss application:

```bash
# Basic tracing
namel3ss debug trace my_app.ai

# Filtered tracing  
namel3ss debug trace my_app.ai --filter agent --filter llm

# Custom output and arguments
namel3ss debug trace my_app.ai \
  --output custom_trace.jsonl \
  --run-args '{"input": "Debug test"}' \
  --memory \
  --performance \
  --buffer-size 2000
```

**Options:**
- `--output, -o`: Output trace file path
- `--filter, -f`: Filter by component type (agent/prompt/chain/tool/llm)
- `--format`: Trace file format (jsonl/json)
- `--memory/--no-memory`: Capture memory usage
- `--performance/--no-performance`: Capture performance metrics
- `--buffer-size`: Event buffer size
- `--run-args`: JSON arguments for app execution

### `namel3ss debug replay`

Replay execution from trace files:

```bash
# Full replay
namel3ss debug replay trace.jsonl

# Interactive step-by-step
namel3ss debug replay trace.jsonl --step

# With breakpoints and mocks
namel3ss debug replay trace.jsonl \
  --breakpoint agent_turn_start:3 \
  --breakpoint llm_call_start \
  --mock "MyAgent:{\"response\":\"Mocked response\"}" \
  --filter agent
```

**Options:**
- `--step/--no-step`: Step through events interactively
- `--breakpoint, -b`: Set breakpoints (`event_type:index` or `component:name`)
- `--mock, -m`: Mock responses (`component_name:response_json`)
- `--filter, -f`: Filter events by component type

### `namel3ss debug analyze`

Analyze trace files for insights:

```bash
# Basic analysis
namel3ss debug analyze trace.jsonl

# Detailed analysis
namel3ss debug analyze trace.jsonl --performance --errors --format json

# Summary only
namel3ss debug analyze trace.jsonl --summary --format table
```

**Options:**
- `--performance/--no-performance`: Show performance analysis
- `--errors/--no-errors`: Show error analysis  
- `--summary/--no-summary`: Show execution summary
- `--format`: Output format (table/json)

### `namel3ss debug inspect`

Inspect specific events or components:

```bash
# Inspect specific event
namel3ss debug inspect trace.jsonl --event 42

# Inspect by component
namel3ss debug inspect trace.jsonl --agent MyAgent
namel3ss debug inspect trace.jsonl --chain ProcessingChain
namel3ss debug inspect trace.jsonl --prompt SummarizePrompt

# Show only errors
namel3ss debug inspect trace.jsonl --errors-only

# Overview (default)
namel3ss debug inspect trace.jsonl
```

**Options:**
- `--event, -e`: Inspect specific event by index
- `--agent, -a`: Show events for specific agent
- `--chain, -c`: Show events for specific chain
- `--prompt, -p`: Show events for specific prompt
- `--errors-only`: Show only error events

## Debugging Workflows

### 1. Basic Debugging Workflow

```bash
# 1. Enable tracing and run your app
export NAMEL3SS_DEBUG_ENABLED=true
namel3ss run my_app.ai

# 2. Check for trace files
ls debug/traces/

# 3. Analyze the trace
namel3ss debug analyze debug/traces/trace_latest.jsonl --performance --errors

# 4. Inspect specific issues
namel3ss debug inspect debug/traces/trace_latest.jsonl --errors-only
```

### 2. Performance Debugging

```bash
# 1. Trace with performance focus
namel3ss debug trace my_app.ai --filter llm --performance

# 2. Analyze performance bottlenecks
namel3ss debug analyze trace.jsonl --performance

# 3. Identify slow operations
namel3ss debug inspect trace.jsonl --agent SlowAgent
```

### 3. Reproducible Bug Investigation

```bash
# 1. Trace the problematic execution
namel3ss debug trace my_app.ai --run-args '{"input": "problem case"}'

# 2. Replay with step-by-step debugging
namel3ss debug replay trace.jsonl --step

# 3. Replay with mock responses to isolate issues  
namel3ss debug replay trace.jsonl --mock "ExternalAPI:{\"status\":\"error\"}"
```

### 4. Memory Debugging

```bash
# 1. Trace with memory monitoring
namel3ss debug trace my_app.ai --memory --filter agent

# 2. Analyze memory usage patterns
namel3ss debug analyze trace.jsonl --format json | jq '.resource_usage'

# 3. Identify memory spikes
namel3ss debug inspect trace.jsonl --agent MemoryIntensiveAgent
```

## Programming API

### Programmatic Tracing

```python
from namel3ss.debugging import (
    initialize_tracing, 
    DebugConfiguration,
    TraceFilter,
    TraceEventType
)

# Configure tracing
config = DebugConfiguration(
    enabled=True,
    trace_output_dir=Path("./traces"),
    trace_filter=TraceFilter(
        components={"agent", "llm"},
        min_duration_ms=100.0
    )
)

# Initialize tracer
tracer = initialize_tracing(config)

# Trace execution
async def traced_execution():
    context = await tracer.start_execution_trace(app_name="DebugApp")
    
    # Your application code here
    
    trace_file = await tracer.end_execution_trace()
    print(f"Trace written to: {trace_file}")
```

### Custom Event Tracing

```python
from namel3ss.debugging.hooks import trace_error, trace_validation_event

# Trace custom errors
await trace_error(
    component="custom",
    component_name="DataProcessor",
    error=RuntimeError("Processing failed"),
    context={"input_size": 1000}
)

# Trace validation events
await trace_validation_event(
    validator_name="SchemaValidator",
    input_data=user_input,
    result=validation_result,
    success=validation_result.valid,
    errors=validation_result.errors
)
```

### Performance Monitoring

```python
from namel3ss.debugging.profiling import (
    get_debug_profiler, 
    profile_debug_execution,
    MemoryTracker
)

# Profile execution
profiler = get_debug_profiler()

with profile_debug_execution("custom_execution"):
    # Your code here
    pass

# Memory tracking
tracker = MemoryTracker()
tracker.start_tracking()
tracker.record_checkpoint("before_processing")
# ... processing ...
tracker.record_checkpoint("after_processing")
report = tracker.get_memory_report()
```

### Replay Analysis

```python
from namel3ss.debugging import ExecutionReplayer, TraceAnalyzer

# Programmatic replay
replayer = ExecutionReplayer(Path("trace.jsonl"))

# Get all agent events
agent_events = replayer.get_events_by_component("agent")

# Analyze performance
analyzer = TraceAnalyzer(Path("trace.jsonl"))
performance_report = analyzer.analyze_performance()
error_report = analyzer.analyze_errors()
```

## Best Practices

### 1. Configuration

- **Environment-based**: Use environment variables for CI/CD and production settings
- **Workspace-based**: Use `.namel3ss/debug.json` for project-specific settings
- **Component filtering**: Filter by relevant components to reduce noise and overhead
- **Retention policy**: Configure trace retention to manage disk usage

### 2. Performance Considerations

- **Buffer size**: Larger buffers reduce I/O overhead but increase memory usage
- **Filtering**: Use component and event type filters to reduce trace volume
- **Payload limits**: Set appropriate payload size limits for large inputs/outputs
- **Background flushing**: Enable buffering for better performance in production

### 3. Security

- **Sensitive data**: Be cautious with tracing when handling sensitive information
- **Payload truncation**: Configure payload size limits to prevent data leakage
- **File permissions**: Ensure trace files have appropriate access controls
- **Retention**: Clean up old traces regularly

### 4. Debugging Strategies

- **Iterative debugging**: Start with broad tracing, then narrow down to specific components
- **Reproduction**: Use deterministic replay to reproduce issues consistently
- **Mock testing**: Use mocked responses to test edge cases and error conditions
- **Performance profiling**: Monitor memory and timing to identify bottlenecks

## Examples

### Example 1: Agent Debugging

```python
# my_agent_app.ai
agent MyAgent {
    prompt: "You are a helpful assistant."
    max_turns: 3
}

chain ProcessUserQuery {
    steps {
        step analyze_input {
            kind: "agent"
            target: "MyAgent"
        }
    }
}
```

Debug the agent:

```bash
# Trace agent execution
namel3ss debug trace my_agent_app.ai --filter agent --run-args '{"input": "Complex query"}'

# Analyze agent performance
namel3ss debug analyze trace.jsonl --performance

# Step through agent turns
namel3ss debug replay trace.jsonl --step --filter agent
```

### Example 2: LLM Call Analysis

```bash
# Trace LLM calls specifically
namel3ss debug trace my_app.ai --filter llm --performance

# Analyze LLM performance
namel3ss debug analyze trace.jsonl --performance --format json | jq '.performance.llm_performance'

# Inspect slow LLM calls
namel3ss debug inspect trace.jsonl | grep -A5 -B5 "duration.*[5-9][0-9][0-9][0-9]"
```

### Example 3: Chain Debugging with Mocks

```bash
# Trace chain execution  
namel3ss debug trace my_chain_app.ai --filter chain

# Replay with mocked LLM responses
namel3ss debug replay trace.jsonl \
  --mock "openai:{\"text\":\"Mocked LLM response\", \"tokens\": 50}" \
  --step

# Analyze chain step performance
namel3ss debug inspect trace.jsonl --chain ProcessingChain
```

## Troubleshooting

### Common Issues

1. **No trace files generated**
   - Check that `NAMEL3SS_DEBUG_ENABLED=true`
   - Verify write permissions to trace output directory
   - Check that application is actually executing traced components

2. **Large trace files**
   - Reduce `buffer_size` or enable more aggressive filtering
   - Set `max_payload_size` to limit event payload sizes
   - Use component filters to trace only relevant parts

3. **Performance overhead**
   - Increase `buffer_size` to reduce I/O frequency
   - Disable memory capture if not needed: `capture_memory=false`
   - Use filtering to reduce event volume

4. **Replay failures**
   - Check that trace file format is valid JSON Lines
   - Verify that mock response format matches expected structure
   - Ensure all referenced components exist in trace

### Debug Configuration Validation

```bash
# Check debug configuration
namel3ss debug --help

# Validate configuration (if command exists)
python -c "
from namel3ss.debugging.config import get_debug_config_manager
manager = get_debug_config_manager()
issues = manager.validate_config()
print('Configuration issues:' if issues else 'Configuration valid')
for issue in issues: print(f'  - {issue}')
"
```

## Contributing

The debugging system is designed to be extensible. Key extension points:

- **Custom event types**: Add new `TraceEventType` values
- **Custom hooks**: Create new tracing decorators and context managers  
- **Analysis tools**: Extend `TraceAnalyzer` with custom analysis methods
- **CLI commands**: Add new debug subcommands
- **Metrics**: Add custom debug metrics and profiling

For more information, see the source code in `namel3ss/debugging/`.