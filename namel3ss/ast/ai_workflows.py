"""
Workflow and chain definitions for multi-step AI processes.

This module contains AST nodes for orchestrating multi-step AI workflows:
- ChainStep: Individual steps in a workflow (extended with planning support)
- Control flow blocks: if/elif/else, for, while loops  
- Chain: Complete workflow definitions
- WorkflowNode: Union type for all workflow nodes
- Planning integration: Support for ReAct, CoT, and Graph-based planners
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .base import Expression


@dataclass
class StepEvaluationConfig:
    """
    Evaluation configuration for a chain step.
    
    Defines how a step's output should be evaluated for quality,
    safety, and correctness before proceeding.
    
    Example:
        evaluation: {
            evaluators: ["toxicity_check", "relevance_score"],
            guardrail: "content_policy"
        }
    """
    evaluators: List[str] = field(default_factory=list)
    guardrail: Optional[str] = None


@dataclass
class ChainStep:
    """
    A single step in a workflow chain.
    
    Represents one operation in a multi-step AI workflow, such as:
    - Invoking a prompt
    - Calling a tool
    - Running a sub-chain
    - Querying a knowledge base
    - Executing a planner (ReAct, CoT, Graph-based)
    
    Example DSL:
        step summarize {
            kind: prompt
            target: "summarize_text"
            options: {
                text: $input
            }
            stop_on_error: true
            evaluation: {
                evaluators: ["quality_check"]
            }
        }
        
        step plan_resolution {
            kind: planner
            target: "react_incident_planner"
            options: {
                incident_data: $input,
                max_cycles: 5
            }
        }
    """
    kind: str  # prompt, tool, chain, knowledge_query, planner, etc.
    target: str  # Name of the prompt/tool/chain/planner to execute
    options: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    stop_on_error: bool = True
    evaluation: Optional[StepEvaluationConfig] = None
    
    # Planning-specific options
    planner_type: Optional[str] = None  # "react", "chain_of_thought", "graph_based"
    planning_context: Optional[Dict[str, Any]] = None  # Additional context for planners


@dataclass
class WorkflowIfBlock:
    """
    Conditional branching in a workflow.
    
    Supports if/elif/else logic for dynamic workflow control
    based on runtime conditions.
    
    Example DSL:
        if confidence > 0.8:
            step high_confidence_path { ... }
        elif confidence > 0.5:
            step medium_confidence_path { ... }
        else:
            step fallback_path { ... }
    """
    condition: Expression
    then_steps: List["WorkflowNode"] = field(default_factory=list)
    elif_steps: List[Tuple[Expression, List["WorkflowNode"]]] = field(default_factory=list)
    else_steps: List["WorkflowNode"] = field(default_factory=list)


@dataclass
class WorkflowForBlock:
    """
    Iteration over a collection in a workflow.
    
    Enables processing each item in a dataset, list, or query result
    through a series of steps.
    
    Example DSL:
        for item in dataset customer_feedback:
            step process_feedback {
                kind: prompt
                target: "analyze_sentiment"
                options: { text: $item.text }
            }
    """
    loop_var: str
    source_kind: str = "expression"  # expression, dataset, table, frame
    source_name: Optional[str] = None
    source_expression: Optional[Expression] = None
    body: List["WorkflowNode"] = field(default_factory=list)
    max_iterations: Optional[int] = None


@dataclass
class WorkflowWhileBlock:
    """
    Conditional iteration in a workflow.
    
    Repeats steps while a condition remains true, with optional
    max iteration limit for safety.
    
    Example DSL:
        while retry_count < 3 and not success:
            step retry_operation { ... }
            set retry_count = retry_count + 1
    """
    condition: Expression
    body: List["WorkflowNode"] = field(default_factory=list)
    max_iterations: Optional[int] = None


@dataclass
class Chain:
    """
    A complete multi-step AI workflow definition.
    
    Chains orchestrate complex AI processes by composing:
    - Sequential steps
    - Conditional logic (if/elif/else)
    - Loops (for, while)
    - Error handling
    - Evaluation and guardrails
    
    Example DSL:
        define chain support_triage {
            input_key: "ticket"
            
            step classify {
                kind: prompt
                target: "classify_ticket"
                options: { text: $input.ticket }
            }
            
            if classify.category == "urgent":
                step escalate {
                    kind: tool
                    target: "notify_oncall"
                }
            
            metadata: {
                version: "1.0",
                owner: "support_team"
            }
        }
    """
    name: str
    input_key: str = "input"
    steps: List["WorkflowNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    declared_effect: Optional[str] = None  # pure, io, stateful
    effects: Set[str] = field(default_factory=set)
    policy_name: Optional[str] = None  # Reference to safety policy


# Union type representing any node that can appear in a workflow
WorkflowNode = Union[ChainStep, WorkflowIfBlock, WorkflowForBlock, WorkflowWhileBlock]


__all__ = [
    "StepEvaluationConfig",
    "ChainStep",
    "WorkflowIfBlock",
    "WorkflowForBlock",
    "WorkflowWhileBlock",
    "Chain",
    "WorkflowNode",
]
