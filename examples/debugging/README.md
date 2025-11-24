# Namel3ss Debugging Examples

Practical examples demonstrating debugging workflows for common namel3ss development scenarios.

## Example Applications

### Simple Agent App

```namel3ss
// examples/debugging/simple_agent.ai
agent WeatherBot {
    prompt: """You are a weather assistant. 
    Provide weather information for the requested location.
    Be concise and helpful."""
    
    max_turns: 3
    
    tools: [get_weather, format_response]
}

// Main interaction
WeatherBot.ask("What's the weather in Tokyo?")
```

### Multi-Step Chain App

```namel3ss
// examples/debugging/processing_chain.ai
prompt AnalyzeInput {
    input: string
    
    """Analyze the user input and categorize it.
    Input: {{input}}
    
    Return analysis in this format:
    {
        "category": "...",
        "intent": "...", 
        "entities": [...]
    }"""
}

prompt GenerateResponse {
    analysis: object
    context: string
    
    """Based on the analysis, generate an appropriate response.
    Analysis: {{analysis}}
    Context: {{context}}
    
    Generate a helpful response."""
}

chain ProcessUserQuery {
    input_key: "user_input"
    
    steps {
        step analyze {
            kind: "prompt"
            target: "AnalyzeInput"
            inputs: { input: "{{input}}" }
        }
        
        step respond {
            kind: "prompt"
            target: "GenerateResponse"
            inputs: { 
                analysis: "{{steps.analyze.output}}",
                context: "Customer support chat"
            }
        }
    }
}
```

### Tool Integration App

```namel3ss
// examples/debugging/tool_app.ai
tool search_knowledge_base {
    description: "Search company knowledge base"
    parameters: {
        query: { type: "string", description: "Search query" }
        category: { type: "string", description: "Category filter" }
    }
}

tool format_answer {
    description: "Format answer with citations"
    parameters: {
        content: { type: "string", description: "Answer content" }
        sources: { type: "array", description: "Source references" }
    }
}

agent SupportBot {
    prompt: "You are a customer support assistant."
    tools: [search_knowledge_base, format_answer]
    max_turns: 5
}
```

## Debugging Scenarios

### Scenario 1: Agent Not Responding Correctly

**Problem**: Agent gives irrelevant responses or stops unexpectedly.

**Debugging Steps**:

```bash
# 1. Trace agent execution with detailed logging
export NAMEL3SS_DEBUG_ENABLED=true
namel3ss debug trace simple_agent.ai \
  --filter agent \
  --filter llm \
  --run-args '{"input": "What'\''s the weather in Tokyo?"}'

# 2. Check the execution summary
namel3ss debug analyze trace_*.jsonl --summary

# 3. Inspect agent turns step by step
namel3ss debug replay trace_*.jsonl --step --filter agent

# 4. Look for errors in agent execution
namel3ss debug inspect trace_*.jsonl --agent WeatherBot --errors-only
```

**Expected Trace Events**:
```json
{"event_type": "agent_execution_start", "component": "agent", "component_name": "WeatherBot"}
{"event_type": "agent_turn_start", "component": "agent", "metadata": {"turn": 1}}
{"event_type": "llm_call_start", "component": "llm", "component_name": "openai"}
{"event_type": "llm_call_end", "component": "llm", "duration_ms": 234.5, "tokens_used": 156}
{"event_type": "tool_call_start", "component": "tool", "component_name": "get_weather"}
{"event_type": "tool_call_end", "component": "tool", "status": "completed"}
{"event_type": "agent_turn_end", "component": "agent", "status": "completed"}
```

**Common Issues**:
- Tool calls failing (check `tool_call_end` events for errors)
- LLM responses not triggering tool usage (check LLM response content)
- Agent reaching max turns without completion

### Scenario 2: Chain Step Failures  

**Problem**: Chain execution stops at a specific step.

**Debugging Steps**:

```bash
# 1. Trace chain execution
namel3ss debug trace processing_chain.ai \
  --filter chain \
  --filter prompt \
  --run-args '{"user_input": "I need help with my order"}'

# 2. Analyze chain performance
namel3ss debug analyze trace_*.jsonl --performance --errors

# 3. Inspect failed steps
namel3ss debug inspect trace_*.jsonl --chain ProcessUserQuery

# 4. Step through chain execution
namel3ss debug replay trace_*.jsonl --step --filter chain
```

**Expected Trace Events**:
```json
{"event_type": "chain_execution_start", "component": "chain", "component_name": "ProcessUserQuery"}
{"event_type": "chain_step_start", "component": "chain", "metadata": {"step": "analyze"}}
{"event_type": "prompt_execution_start", "component": "prompt", "component_name": "AnalyzeInput"}
{"event_type": "llm_call_start", "component": "llm"}
{"event_type": "llm_call_end", "component": "llm", "status": "completed"}
{"event_type": "validation_start", "component": "prompt"}
{"event_type": "validation_end", "component": "prompt", "status": "completed"}
{"event_type": "prompt_execution_end", "component": "prompt", "status": "completed"}
{"event_type": "chain_step_end", "component": "chain", "status": "completed"}
```

**Replay with Debugging**:
```bash
# Replay with breakpoints at each step
namel3ss debug replay trace_*.jsonl \
  --breakpoint chain_step_start \
  --step

# Test with mock responses  
namel3ss debug replay trace_*.jsonl \
  --mock "AnalyzeInput:{\"category\":\"support\",\"intent\":\"order_help\"}"
```

### Scenario 3: Performance Issues

**Problem**: Application runs slowly, need to identify bottlenecks.

**Debugging Steps**:

```bash
# 1. Trace with performance focus
namel3ss debug trace tool_app.ai \
  --performance \
  --memory \
  --run-args '{"input": "Complex query requiring multiple tools"}'

# 2. Analyze performance bottlenecks
namel3ss debug analyze trace_*.jsonl --performance --format json > perf_report.json

# 3. Identify slow operations
cat perf_report.json | jq '.performance.slow_operations'

# 4. Inspect LLM performance
cat perf_report.json | jq '.performance.llm_performance'

# 5. Memory usage analysis
cat perf_report.json | jq '.resource_usage'
```

**Performance Analysis Script**:
```python
#!/usr/bin/env python3
import json
from pathlib import Path
from namel3ss.debugging import TraceAnalyzer

# Load trace and analyze
analyzer = TraceAnalyzer(Path("trace_latest.jsonl"))

# Performance analysis
perf_report = analyzer.analyze_performance()
print("=== Performance Report ===")
print(f"LLM calls: {perf_report['llm_performance']['total_calls']}")
print(f"Avg LLM duration: {perf_report['llm_performance']['avg_duration_ms']:.2f}ms")
print(f"Max LLM duration: {perf_report['llm_performance']['max_duration_ms']:.2f}ms")

# Find slowest operations
slow_ops = perf_report['slow_operations']
print(f"\nSlow operations (>{len(slow_ops)} found):")
for op in slow_ops[:5]:  # Top 5
    print(f"  {op['component']}/{op['component_name']}: {op['duration_ms']:.2f}ms")

# Memory usage
summary = analyzer.replayer.get_execution_summary()
memory_info = summary.get('resource_usage', {})
print(f"\nMemory usage: {memory_info.get('avg_memory_usage_mb', 0):.1f}MB average")
```

### Scenario 4: Tool Integration Debugging

**Problem**: Tools not being called or returning unexpected results.

**Debugging Steps**:

```bash
# 1. Trace tool calls specifically
namel3ss debug trace tool_app.ai \
  --filter agent \
  --filter tool \
  --run-args '{"input": "Search for pricing information"}'

# 2. Inspect tool call patterns
namel3ss debug inspect trace_*.jsonl --errors-only

# 3. Step through tool execution
namel3ss debug replay trace_*.jsonl --step --filter tool

# 4. Test with mock tool responses
namel3ss debug replay trace_*.jsonl \
  --mock "search_knowledge_base:{\"results\":[{\"title\":\"Pricing\",\"content\":\"...\"}]}" \
  --mock "format_answer:{\"formatted_response\":\"Here is the pricing information...\"}"
```

**Tool Call Analysis**:
```bash
# Extract tool call information
namel3ss debug inspect trace_*.jsonl --format json | \
  jq '.events[] | select(.event_type == "tool_call_start" or .event_type == "tool_call_end")'

# Count tool calls by type
namel3ss debug inspect trace_*.jsonl --format json | \
  jq -r '.events[] | select(.event_type == "tool_call_start") | .component_name' | \
  sort | uniq -c
```

### Scenario 5: Memory Issues

**Problem**: Application using excessive memory or experiencing memory leaks.

**Memory Debugging Setup**:

```python
# memory_debug.py
import asyncio
from pathlib import Path
from namel3ss.debugging import initialize_tracing, DebugConfiguration
from namel3ss.debugging.profiling import MemoryTracker
from namel3ss.cli.loading import load_n3_app

async def debug_memory_usage():
    # Configure memory-focused tracing
    config = DebugConfiguration(
        enabled=True,
        capture_memory_usage=True,
        capture_performance_markers=True,
        trace_output_dir=Path("./memory_traces")
    )
    
    # Initialize tracer and memory tracker
    tracer = initialize_tracing(config)
    tracker = MemoryTracker(sample_interval_ms=500)
    
    # Start tracking
    tracker.start_tracking()
    context = await tracer.start_execution_trace(app_name="MemoryDebugApp")
    
    try:
        # Load app
        tracker.record_checkpoint("app_load_start")
        app = load_n3_app(Path("tool_app.ai"))
        tracker.record_checkpoint("app_load_complete")
        
        # Execute multiple times to check for leaks
        for i in range(5):
            tracker.record_checkpoint(f"execution_{i}_start")
            # Your app execution code here
            tracker.record_checkpoint(f"execution_{i}_end")
    
    finally:
        # Generate reports
        await tracer.end_execution_trace()
        memory_report = tracker.get_memory_report()
        
        print("=== Memory Report ===")
        print(f"Baseline: {memory_report['baseline_mb']:.1f}MB")
        print(f"Current: {memory_report['current_mb']:.1f}MB")
        print(f"Peak: {memory_report['peak_mb']:.1f}MB")
        print(f"Growth: {memory_report['delta_from_baseline_mb']:.1f}MB")

if __name__ == "__main__":
    asyncio.run(debug_memory_usage())
```

Run memory debugging:
```bash
python memory_debug.py

# Analyze memory patterns in trace
namel3ss debug analyze memory_traces/trace_*.jsonl --format json | \
  jq '.resource_usage'
```

## Advanced Debugging Techniques

### Custom Event Injection

```python
# custom_debug.py
from namel3ss.debugging import get_global_tracer, TraceEventType

async def trace_custom_component():
    tracer = get_global_tracer()
    if not tracer:
        return
    
    # Inject custom performance markers
    await tracer.emit_event(
        TraceEventType.PERFORMANCE_MARKER,
        component="custom",
        component_name="DataProcessor",
        inputs={"data_size": 1000},
        metadata={"checkpoint": "preprocessing_start"}
    )
    
    # Your processing code here
    
    await tracer.emit_event(
        TraceEventType.PERFORMANCE_MARKER,
        component="custom",
        component_name="DataProcessor",
        outputs={"processed_items": 850},
        metadata={"checkpoint": "preprocessing_end"}
    )
```

### Conditional Breakpoints

```python
# conditional_replay.py
from namel3ss.debugging import ExecutionReplayer, ReplayBreakpoint

def is_slow_llm_call(event):
    """Breakpoint condition: stop on slow LLM calls"""
    return (event.event_type.name == "llm_call_end" and 
            event.duration_ms and event.duration_ms > 1000)

def is_error_with_retries(event):
    """Breakpoint condition: stop on errors with retry metadata"""
    return (event.status == "failed" and 
            event.metadata.get("retry_count", 0) > 0)

# Create replayer with conditional breakpoints
replayer = ExecutionReplayer(
    trace_file=Path("trace.jsonl"),
    breakpoints=[
        ReplayBreakpoint(condition=is_slow_llm_call),
        ReplayBreakpoint(condition=is_error_with_retries),
    ]
)

# Run replay
while not replayer.state.completed:
    event = replayer.replay_step()
    if replayer.state.paused:
        print(f"Breakpoint hit: {event.event_type}")
        print(f"Duration: {event.duration_ms}ms")
        print(f"Error: {event.error}")
        
        # Interactive debugging
        action = input("Continue (c), step (s), or quit (q)? ")
        if action == 'q':
            break
        elif action == 's':
            replayer.state.paused = False
```

### Trace Comparison

```bash
# Compare traces from different executions
namel3ss debug trace app.ai --run-args '{"input": "test 1"}' --output trace1.jsonl
namel3ss debug trace app.ai --run-args '{"input": "test 2"}' --output trace2.jsonl

# Generate comparison report
python compare_traces.py trace1.jsonl trace2.jsonl
```

```python
# compare_traces.py
import sys
from pathlib import Path
from namel3ss.debugging import TraceAnalyzer

def compare_traces(trace1_path, trace2_path):
    analyzer1 = TraceAnalyzer(Path(trace1_path))
    analyzer2 = TraceAnalyzer(Path(trace2_path))
    
    summary1 = analyzer1.replayer.get_execution_summary()
    summary2 = analyzer2.replayer.get_execution_summary()
    
    print("=== Trace Comparison ===")
    
    # Event count comparison
    events1 = summary1["execution_overview"]["total_events"]
    events2 = summary2["execution_overview"]["total_events"]
    print(f"Events: {events1} vs {events2} (diff: {events2 - events1:+d})")
    
    # Duration comparison
    duration1 = summary1["execution_overview"]["execution_duration_seconds"]
    duration2 = summary2["execution_overview"]["execution_duration_seconds"]
    print(f"Duration: {duration1:.2f}s vs {duration2:.2f}s (diff: {duration2 - duration1:+.2f}s)")
    
    # Error comparison
    errors1 = summary1["execution_overview"]["error_count"]
    errors2 = summary2["execution_overview"]["error_count"]
    print(f"Errors: {errors1} vs {errors2} (diff: {errors2 - errors1:+d})")
    
    # Component breakdown
    components1 = set(summary1["component_breakdown"].keys())
    components2 = set(summary2["component_breakdown"].keys())
    
    only_in_1 = components1 - components2
    only_in_2 = components2 - components1
    
    if only_in_1:
        print(f"Components only in trace1: {only_in_1}")
    if only_in_2:
        print(f"Components only in trace2: {only_in_2}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_traces.py trace1.jsonl trace2.jsonl")
        sys.exit(1)
    
    compare_traces(sys.argv[1], sys.argv[2])
```

## Debugging Checklists

### Pre-debugging Checklist

- [ ] Debug tracing enabled (`NAMEL3SS_DEBUG_ENABLED=true`)
- [ ] Trace output directory writable
- [ ] Component filters configured appropriately
- [ ] Buffer size set for performance vs memory trade-off
- [ ] Retention policy configured to manage disk space

### Agent Debugging Checklist

- [ ] Agent execution events present in trace
- [ ] Agent turns completing successfully
- [ ] Tool calls being made when expected
- [ ] LLM responses triggering appropriate actions
- [ ] Memory operations working correctly
- [ ] Max turns not being exceeded unexpectedly

### Chain Debugging Checklist  

- [ ] Chain execution starting and completing
- [ ] All steps executing in correct order
- [ ] Step inputs/outputs flowing correctly
- [ ] Conditional logic working as expected
- [ ] Error handling functioning properly
- [ ] Performance within acceptable bounds

### Performance Debugging Checklist

- [ ] LLM call durations reasonable
- [ ] Tool execution times acceptable
- [ ] Memory usage stable over time
- [ ] No memory leaks detected
- [ ] Buffer flush frequency appropriate
- [ ] Trace file sizes manageable

### Error Debugging Checklist

- [ ] Error events captured with full context
- [ ] Stack traces available for runtime errors
- [ ] Validation failures have detailed messages
- [ ] Retry logic working correctly
- [ ] Error recovery mechanisms functioning
- [ ] Error patterns identified and documented

## Tips and Best Practices

### 1. Efficient Tracing

```bash
# Start with broad tracing
namel3ss debug trace app.ai --filter agent --filter llm

# Narrow down to specific components
namel3ss debug trace app.ai --filter llm --performance

# Use custom buffer sizes for different scenarios
namel3ss debug trace app.ai --buffer-size 5000  # For high-volume apps
namel3ss debug trace app.ai --buffer-size 100   # For memory-constrained environments
```

### 2. Effective Replay

```bash
# Use breakpoints strategically
namel3ss debug replay trace.jsonl --breakpoint agent_turn_start:1 --step

# Mock external dependencies
namel3ss debug replay trace.jsonl --mock "WeatherAPI:{\"temperature\":\"25C\"}"

# Filter replay for focus
namel3ss debug replay trace.jsonl --filter agent --step
```

### 3. Analysis Automation

```bash
# Create analysis pipeline
namel3ss debug analyze trace.jsonl --format json > analysis.json
cat analysis.json | jq '.performance.slow_operations | length' # Count slow ops
cat analysis.json | jq '.errors.total_errors'                 # Error count
cat analysis.json | jq -r '.execution_overview.success_rate'  # Success rate
```

### 4. Integration with CI/CD

```bash
# In CI pipeline
export NAMEL3SS_DEBUG_ENABLED=true
export NAMEL3SS_DEBUG_COMPONENTS=agent,llm
namel3ss test my_app.ai

# Analyze test results
if namel3ss debug analyze test_trace.jsonl --errors --format json | jq -e '.errors.total_errors > 0'; then
    echo "Errors detected in test execution"
    exit 1
fi
```

These examples provide comprehensive coverage of debugging scenarios you'll encounter while developing namel3ss applications. Use them as starting points for your own debugging workflows!