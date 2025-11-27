# Advanced Planning & Reasoning in Namel3ss (N3) 🧠

## Overview

The Namel3ss planning system provides production-ready advanced reasoning capabilities that augment deterministic chains with intelligent planners. This system enables dynamic task decomposition, multi-step reasoning, and adaptive problem-solving while maintaining full backward compatibility.

## 🚀 Quick Start

### Basic ReAct Planner
```n3
planner "smart_assistant" {
  type: "react"
  goal: "Help users solve complex problems"
  max_cycles: 5
  
  reasoning_prompt: "reasoning_template"
  action_tools: ["search", "calculator", "code_executor"]
  
  success_condition: {
    problem_solved: true,
    user_satisfied: true
  }
}

chain "ai_assistant" {
  steps: [
    {
      kind: "planner"
      target: "smart_assistant"
      planner_type: "react"
      options: {
        user_query: "$input"
      }
    }
  ]
}
```

### Chain-of-Thought Analysis
```n3
planner "data_analyst" {
  type: "chain_of_thought"
  problem: "Analyze sales data and identify trends"
  
  reasoning_steps: [
    { step: "data_validation", description: "Check data quality" },
    { step: "trend_analysis", description: "Find patterns" },
    { step: "insights", description: "Generate recommendations" }
  ]
  
  step_tools: {
    data_validation: ["data_checker"]
    trend_analysis: ["statistical_analyzer"]
    insights: ["report_generator"]
  }
}
```

## 🏗️ Architecture Overview

### Core Components

1. **AST Layer** (`namel3ss/ast/planning.py`)
   - Planner nodes for all planning types
   - Search policy configurations
   - Integration with existing workflow nodes

2. **IR Layer** (`namel3ss/ir/spec.py`)
   - Runtime-agnostic planner specifications
   - Security and resource metadata
   - Planning workflow definitions

3. **Runtime Layer** (`namel3ss/runtime/planning_*`)
   - Core planning execution engines
   - Integration with chain execution
   - Memory and context management

4. **Backend Layer** (Generated routers and handlers)
   - FastAPI endpoints for planner execution
   - State management and persistence
   - Observability and metrics

### Data Flow

```
User Request → Chain → Planner Step → Planning Engine → Tools/Actions → Results
                ↓                                           ↑
           Planning Context ←→ Memory Management ←→ Security Layer
```

## 📚 Planner Types

### 1. ReAct Planners

**Use Case**: Dynamic problem-solving with iterative reasoning and action

```n3
planner "react_agent" {
  type: "react"
  goal: "Solve customer support tickets"
  max_cycles: 10
  
  reasoning_prompt: "support_reasoning"
  action_tools: ["knowledge_base", "ticket_system", "escalation"]
  
  success_condition: {
    ticket_resolved: true,
    customer_satisfied: true
  }
  
  fallback_action: "escalate_to_human"
  
  security: {
    permission_level: "support_agent"
    allowed_tools: ["knowledge_base", "ticket_system"]
    capabilities: ["customer_interaction"]
  }
}
```

**Key Features**:
- Iterative reasoning loops
- Dynamic tool selection
- Adaptive goal pursuit
- Fallback strategies

### 2. Chain-of-Thought Planners

**Use Case**: Structured multi-step analysis and reasoning

```n3
planner "financial_analyzer" {
  type: "chain_of_thought"
  problem: "Analyze quarterly financial performance"
  
  reasoning_steps: [
    { step: "revenue_analysis", description: "Analyze revenue trends" },
    { step: "cost_analysis", description: "Examine cost structures" },
    { step: "profit_margins", description: "Calculate margin changes" },
    { step: "forecasting", description: "Project future performance" },
    { step: "recommendations", description: "Strategic recommendations" }
  ]
  
  dependencies: {
    cost_analysis: ["revenue_analysis"]
    profit_margins: ["revenue_analysis", "cost_analysis"]
    forecasting: ["profit_margins"]
    recommendations: ["forecasting"]
  }
  
  step_tools: {
    revenue_analysis: ["financial_db", "trend_analyzer"]
    cost_analysis: ["accounting_system", "cost_calculator"]
    profit_margins: ["calculator", "chart_generator"]
    forecasting: ["ml_predictor", "scenario_modeler"]
    recommendations: ["strategy_ai", "report_generator"]
  }
  
  intermediate_validation: true
}
```

**Key Features**:
- Sequential step execution
- Dependency management
- Step-specific tools and prompts
- Intermediate validation

### 3. Graph-Based Planners

**Use Case**: Complex state space exploration and optimization

```n3
planner "process_optimizer" {
  type: "graph_based"
  
  initial_state: {
    status: "new_request"
    priority: "medium"
    resources: 0
  }
  
  goal_state: {
    status: "completed"
    quality_score: ">= 0.9"
    cost: "<= 1000"
  }
  
  search_policy: {
    policy_type: "a_star"
    max_depth: 15
    scoring_function: "cost_quality_balance"
  }
  
  state_transitions: [
    {
      from: {status: "new_request"}
      action: "quick_triage"
      to: {status: "triaged", resources: 1}
      cost: 2
    },
    {
      from: {status: "triaged", priority: "high"}
      action: "fast_track"
      to: {status: "in_progress", resources: 5}
      cost: 10
    },
    {
      from: {status: "in_progress"}
      action: "complete_work"
      to: {status: "completed", quality_score: 0.95}
      cost: 15
    }
  ]
  
  heuristic_function: "completion_estimator"
  max_search_time: 30.0
}
```

**Key Features**:
- State space exploration
- Multiple search algorithms
- Heuristic functions
- Cost optimization

## 🔧 Planning Workflows

Multi-stage workflows combine different planners for complex processes:

```n3
planning_workflow "incident_response" {
  input_schema: "IncidentAlert"
  output_schema: "ResolutionReport"
  
  stages: [
    {
      name: "assessment"
      planner: "react_incident_agent"
      goal: "Assess incident severity and impact"
    },
    {
      name: "analysis"
      planner: "cot_data_analyzer"
      goal: "Analyze root causes and system impact"
    },
    {
      name: "resolution"
      planner: "graph_workflow_optimizer"
      goal: "Plan optimal resolution strategy"
    }
  ]
  
  stage_dependencies: {
    analysis: ["assessment"]
    resolution: ["analysis"]
  }
  
  error_handling: {
    retry_failed_stages: true
    max_retries: 3
    fallback_to_manual: true
  }
}
```

## 🛡️ Security Model

### Permission-Based Access

```n3
planner "secure_planner" {
  type: "react"
  
  security: {
    permission_level: "data_analyst"
    allowed_tools: ["public_api", "analytics_db"]
    capabilities: ["data_read", "report_generate"]
    
    # Tool-specific permissions
    tool_permissions: {
      analytics_db: ["read"]
      public_api: ["read", "query"]
    }
    
    # Resource limits
    max_execution_time: 300
    max_memory_usage: 100000000  # 100MB
    max_tool_calls: 50
  }
}
```

### Audit Trail

All planning executions are logged with:
- Planning steps and decisions
- Tool usage and permissions
- Resource consumption
- Security events

## 📊 Observability & Monitoring

### Built-in Metrics

```n3
dataset "planning_metrics" {
  columns: [
    {name: "planner_name", type: "string"},
    {name: "execution_time", type: "float"},
    {name: "success_rate", type: "float"},
    {name: "step_count", type: "integer"},
    {name: "cost", type: "float"},
    {name: "session_id", type: "string"}
  ]
}
```

### Performance Tracking

- Planning execution times
- Success/failure rates
- Resource utilization
- Tool usage patterns
- Memory consumption

### Debug Traces

```n3
dataset "planning_traces" {
  columns: [
    {name: "session_id", type: "string"},
    {name: "step_type", type: "string"},
    {name: "step_content", type: "json"},
    {name: "timestamp", type: "datetime"}
  ]
}
```

## 🔄 Integration with Chains

Planners integrate seamlessly with existing chains:

```n3
chain "intelligent_workflow" {
  steps: [
    {
      kind: "prompt"
      target: "initial_analysis"
    },
    {
      kind: "planner"
      target: "problem_solver"
      planner_type: "react"
      options: {
        context: "$analysis_result"
        priority: "$priority"
      }
      planning_context: {
        user_id: "$user_id"
        session_id: "$session_id"
      }
    },
    {
      kind: "conditional"
      condition: "$planner_confidence > 0.8"
      then: [
        { kind: "tool", target: "success_notification" }
      ]
      else: [
        { kind: "planning_workflow", target: "escalation_workflow" }
      ]
    }
  ]
}
```

## 🏃 Runtime Performance

### Optimization Features

1. **Lazy Loading**: Planners load tools and prompts on demand
2. **Caching**: Intelligent caching of planning states and results  
3. **Parallel Execution**: Independent planning steps run in parallel
4. **Resource Limits**: Configurable timeouts and memory limits
5. **Streaming**: Real-time streaming of planning progress

### Scaling Considerations

- **Memory Management**: Automatic cleanup of planning contexts
- **Tool Rate Limiting**: Built-in rate limiting for external tools
- **Concurrent Planners**: Support for multiple concurrent planning sessions
- **State Persistence**: Optional persistence of planning states

## 🧪 Testing & Validation

### Comprehensive Test Suite

The planning system includes extensive tests:

```bash
# Run planning system tests
pytest tests/test_planning_system.py -v

# Test specific planner types
pytest tests/test_planning_system.py::test_react_planner -v
pytest tests/test_planning_system.py::test_cot_planner -v
pytest tests/test_planning_system.py::test_graph_planner -v
```

### Test Coverage

- ✅ All planner types (ReAct, CoT, Graph-based)
- ✅ Planning workflow execution
- ✅ Security and permissions
- ✅ Error handling and recovery
- ✅ Performance and resource limits
- ✅ Integration with chains
- ✅ Memory management
- ✅ Tool integration

## 📖 Advanced Examples

### Multi-Modal Planning

```n3
planner "multimodal_analyst" {
  type: "chain_of_thought"
  problem: "Analyze customer feedback across text, images, and audio"
  
  reasoning_steps: [
    { step: "text_analysis", description: "Process text feedback" },
    { step: "image_analysis", description: "Analyze uploaded images" },
    { step: "audio_analysis", description: "Process voice feedback" },
    { step: "sentiment_fusion", description: "Combine all sentiment signals" },
    { step: "insights", description: "Generate comprehensive insights" }
  ]
  
  step_tools: {
    text_analysis: ["nlp_processor", "sentiment_analyzer"]
    image_analysis: ["vision_ai", "ocr_tool"]
    audio_analysis: ["speech_to_text", "audio_sentiment"]
    sentiment_fusion: ["ml_fusion", "confidence_scorer"]
    insights: ["report_generator", "visualization_tool"]
  }
}
```

### Adaptive Learning Planner

```n3
planner "adaptive_support" {
  type: "react"
  goal: "Provide personalized customer support"
  
  reasoning_prompt: "adaptive_reasoning"
  action_tools: ["knowledge_search", "user_history", "personalization"]
  
  # Learning from past interactions
  learning_config: {
    feedback_integration: true
    success_pattern_recognition: true
    failure_analysis: true
  }
  
  adaptation_rules: [
    {
      condition: "user_satisfaction < 0.7"
      action: "increase_personalization"
    },
    {
      condition: "resolution_time > target_time"
      action: "optimize_tool_selection"
    }
  ]
}
```

## 🔮 Future Enhancements

### Planned Features

1. **Meta-Planning**: Planners that plan the planning strategy
2. **Hierarchical Planning**: Multi-level planning decomposition
3. **Collaborative Planning**: Multi-agent planning systems
4. **Learning Planners**: Self-improving planning strategies
5. **Visual Planning**: Graph-based planning with visual interfaces

### Integration Roadmap

- [ ] Integration with reinforcement learning
- [ ] Advanced search algorithms (MCTS, genetic algorithms)
- [ ] Planning with uncertainty and probabilistic models
- [ ] Distributed planning across multiple compute nodes
- [ ] Real-time planning with streaming data

## 📝 Best Practices

### Planning Design

1. **Clear Goals**: Define specific, measurable planning objectives
2. **Appropriate Type**: Choose the right planner type for your use case
3. **Tool Selection**: Provide relevant, well-designed tools
4. **Error Handling**: Include comprehensive fallback strategies
5. **Resource Limits**: Set appropriate timeouts and memory limits

### Performance Optimization

1. **Caching Strategy**: Cache expensive computations and tool calls
2. **Parallel Execution**: Utilize parallel step execution where possible
3. **Memory Management**: Clean up planning contexts regularly
4. **Tool Efficiency**: Optimize tool implementations for speed
5. **Monitoring**: Monitor planning performance continuously

### Security Considerations

1. **Least Privilege**: Grant minimum necessary permissions
2. **Tool Sandboxing**: Isolate tool execution environments
3. **Input Validation**: Validate all planning inputs
4. **Audit Logging**: Log all security-relevant events
5. **Resource Limits**: Prevent resource exhaustion attacks

## 🤝 Contributing

The planning system is extensible and welcomes contributions:

1. **Custom Planners**: Implement new planning algorithms
2. **Search Policies**: Add new search strategies for graph-based planners
3. **Tools**: Create specialized tools for planning operations
4. **Optimizations**: Improve performance and efficiency
5. **Documentation**: Enhance examples and guides

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: See examples and test files
- **Issues**: Report bugs and feature requests
- **Community**: Join discussions about planning strategies
- **Performance**: Share optimization techniques and benchmarks

---

*The Namel3ss planning system represents a significant advancement in AI programming languages, providing production-ready intelligent reasoning capabilities while maintaining the simplicity and reliability of deterministic programming paradigms.*# Planning & Reasoning Quick Reference 📋

## 🚀 Planner Types at a Glance

### ReAct Planner - Dynamic Problem Solving
```n3
planner "my_react" {
  type: "react"
  goal: "Solve problems step by step"
  max_cycles: 8
  reasoning_prompt: "react_template"
  action_tools: ["tool1", "tool2"]
  success_condition: { solved: true }
  fallback_action: "escalate"
}
```
**When to use**: Dynamic, iterative problem solving

### Chain-of-Thought - Structured Reasoning  
```n3
planner "my_cot" {
  type: "chain_of_thought"
  problem: "Analyze complex data"
  reasoning_steps: [
    { step: "gather", description: "Collect data" },
    { step: "analyze", description: "Find patterns" },
    { step: "conclude", description: "Generate insights" }
  ]
  dependencies: {
    analyze: ["gather"]
    conclude: ["analyze"]
  }
}
```
**When to use**: Multi-step analysis with dependencies

### Graph-Based - State Space Search
```n3
planner "my_graph" {
  type: "graph_based"
  initial_state: { status: "start" }
  goal_state: { status: "done", quality: ">= 0.9" }
  search_policy: {
    policy_type: "a_star"
    max_depth: 10
  }
  state_transitions: [
    {
      from: {status: "start"}
      action: "process"
      to: {status: "done", quality: 0.95}
      cost: 5
    }
  ]
}
```
**When to use**: Complex workflows with optimization

## 🔧 Integration with Chains

```n3
chain "smart_chain" {
  steps: [
    {
      kind: "planner"
      target: "my_planner"
      planner_type: "react"
      options: { input: "$user_request" }
    }
  ]
}
```

## 🔒 Security & Limits

```n3
planner "secure_planner" {
  security: {
    permission_level: "analyst"
    allowed_tools: ["safe_tool"]
    capabilities: ["read_data"]
    max_execution_time: 300
    max_memory_usage: 50000000
  }
}
```

## 🎯 Search Policies

| Policy | Use Case | Configuration |
|--------|----------|---------------|
| `beam_search` | Balanced exploration | `beam_width: 3` |
| `a_star` | Optimal pathfinding | `scoring_function: "cost"` |
| `best_first` | Greedy optimization | `max_depth: 15` |
| `depth_first` | Exhaustive search | `max_depth: 20` |

## 📊 Planning Workflows

```n3
planning_workflow "multi_stage" {
  stages: [
    { name: "stage1", planner: "react_planner" },
    { name: "stage2", planner: "cot_planner" },
    { name: "stage3", planner: "graph_planner" }
  ]
  stage_dependencies: {
    stage2: ["stage1"]
    stage3: ["stage2"]
  }
}
```

## 🛠️ Common Patterns

### Error Handling
```n3
planner "robust_planner" {
  fallback_action: "escalate_to_human"
  max_cycles: 10
  success_condition: {
    confidence: ">= 0.8",
    completed: true
  }
}
```

### Tool Integration
```n3
planner "tool_user" {
  action_tools: ["api_caller", "data_processor", "notifier"]
  step_tools: {
    analysis: ["data_processor"]
    notification: ["notifier"]
  }
}
```

### Memory & Context
```n3
memory "planning_memory" {
  scope: "session"
  kind: "conversational"
  max_items: 100
}

planner "contextual" {
  memory_config: {
    context_window: 50
    include_history: true
  }
}
```

## 🎛️ Configuration Options

### Performance Tuning
```n3
planner "optimized" {
  performance: {
    timeout_seconds: 120
    max_memory_usage: 100000000
    enable_caching: true
    parallel_steps: true
  }
}
```

### Monitoring & Debugging
```n3
planner "observable" {
  observability: {
    trace_execution: true
    collect_metrics: true
    debug_mode: false
  }
}
```

## 📈 Success Conditions

```n3
# Simple boolean check
success_condition: { task_complete: true }

# Numeric thresholds
success_condition: {
  confidence: ">= 0.8",
  score: "> 0.9"
}

# Complex conditions
success_condition: {
  status: "resolved",
  customer_satisfied: true,
  time_taken: "< 300"
}
```

## 🔄 State Transitions (Graph Planners)

```n3
state_transitions: [
  {
    from: {stage: "start"}
    action: "begin_work"
    to: {stage: "working", progress: 0.1}
    cost: 2
    conditions: {resources: "available"}
  },
  {
    from: {stage: "working"}
    action: "complete"
    to: {stage: "done", progress: 1.0}
    cost: 8
  }
]
```

## 🎨 Prompt Templates

```n3
prompt "react_reasoning" {
  template: """
  Current situation: {{current_state}}
  Goal: {{goal}}
  Available tools: {{tools}}
  
  Think step by step:
  1. What do I know?
  2. What do I need to find out?
  3. What action should I take?
  
  Action: {{action}}
  Input: {{input}}
  """
}
```

## 📋 Testing Commands

```bash
# Test all planners
pytest tests/test_planning_system.py -v

# Test specific type
pytest tests/test_planning_system.py::test_react_planner -v

# Performance testing
pytest tests/test_planning_system.py::test_performance -v
```

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Planner loops infinitely | Set `max_cycles` limit |
| Memory usage too high | Set `max_memory_usage` |
| Tools not found | Check tool registration |
| Permissions denied | Verify security config |
| Slow execution | Enable caching, parallel steps |

## 💡 Best Practices

1. **Start Simple**: Begin with basic planners, add complexity gradually
2. **Set Limits**: Always configure timeouts and resource limits  
3. **Test Thoroughly**: Use the comprehensive test suite
4. **Monitor Performance**: Track metrics and execution traces
5. **Secure by Default**: Apply least-privilege security model
6. **Handle Errors**: Include fallback strategies
7. **Document Goals**: Clear success conditions and objectives

## 🔗 Key Files

- `namel3ss/ast/planning.py` - AST definitions
- `namel3ss/ir/spec.py` - IR specifications  
- `namel3ss/runtime/planning_core.py` - Core engines
- `tests/test_planning_system.py` - Comprehensive tests
- `examples/planning_demo.n3` - Full example app

---
*Quick reference for the Namel3ss Planning & Reasoning System v1.0*