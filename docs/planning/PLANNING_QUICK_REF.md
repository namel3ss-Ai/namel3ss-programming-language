# Planning & Reasoning Quick Reference 📋

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