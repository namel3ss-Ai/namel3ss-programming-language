"""
Planning and Reasoning AST Nodes for Namel3ss

This module extends the AI workflow system with advanced planning capabilities,
including ReAct, Chain-of-Thought, and Graph-based planners.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Literal
from .base import ASTNode
from .expressions import Expression


# =============================================================================
# Planning Policy Types
# =============================================================================

@dataclass
class SearchPolicy(ASTNode):
    """Base class for search policies used in planning"""
    policy_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeamSearchPolicy(SearchPolicy):
    """Beam search policy for exploring multiple solution paths"""
    beam_width: int = 3
    max_depth: int = 10
    scoring_function: Optional[str] = None
    
    def __post_init__(self):
        self.policy_type = "beam_search"
        self.parameters = {
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "scoring_function": self.scoring_function
        }


@dataclass 
class GreedySearchPolicy(SearchPolicy):
    """Greedy search policy for single-path exploration"""
    max_steps: int = 20
    confidence_threshold: float = 0.8
    
    def __post_init__(self):
        self.policy_type = "greedy_search"
        self.parameters = {
            "max_steps": self.max_steps,
            "confidence_threshold": self.confidence_threshold
        }


@dataclass
class MCTSPolicy(SearchPolicy):
    """Monte Carlo Tree Search policy for exploration-exploitation balance"""
    num_simulations: int = 100
    exploration_constant: float = 1.41
    max_simulation_depth: int = 15
    
    def __post_init__(self):
        self.policy_type = "mcts"
        self.parameters = {
            "num_simulations": self.num_simulations,
            "exploration_constant": self.exploration_constant,
            "max_simulation_depth": self.max_simulation_depth
        }


# =============================================================================
# Planner Definitions
# =============================================================================

@dataclass
class PlannerStep(ASTNode):
    """
    A single step in a planning process.
    
    Planning steps can be:
    - Actions (tool calls, API requests)
    - Observations (reading results, status checks)
    - Reasoning (analysis, hypothesis generation)
    - Decisions (branching, goal selection)
    """
    step_type: str  # "action", "observation", "reasoning", "decision"
    description: str
    target: Optional[str] = None  # Tool, prompt, or chain reference
    condition: Optional[Expression] = None  # Conditional execution
    output_key: Optional[str] = None  # Where to store results
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActPlanner(ASTNode):
    """
    ReAct (Reasoning + Acting) planner implementation.
    
    Interleaves reasoning steps with action steps in cycles:
    1. Thought: Analyze current state and plan next action
    2. Action: Execute tool calls or operations
    3. Observation: Process results and update state
    4. Repeat until goal achieved or max steps reached
    
    Example DSL:
        planner react_support {
            goal: "Resolve customer support ticket"
            max_cycles: 5
            
            reasoning_prompt: "analyze_situation"
            action_tools: ["search_kb", "create_ticket", "send_email"]
            
            success_condition: ticket_resolved == true
            fallback_action: escalate_to_human
        }
    """
    name: str
    goal: str
    max_cycles: int = 10
    reasoning_prompt: str  # Prompt for reasoning steps
    action_tools: List[str] = field(default_factory=list)  # Available tools
    success_condition: Optional[Expression] = None
    fallback_action: Optional[str] = None
    initial_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainOfThoughtPlanner(ASTNode):
    """
    Chain-of-Thought planner for step-by-step reasoning.
    
    Decomposes complex problems into sequential reasoning steps,
    with each step building on previous conclusions.
    
    Example DSL:
        planner cot_analyzer {
            problem: "Analyze customer churn risk"
            
            reasoning_steps: [
                { step: "gather_customer_data", depends_on: [] },
                { step: "analyze_usage_patterns", depends_on: ["gather_customer_data"] },
                { step: "identify_risk_factors", depends_on: ["analyze_usage_patterns"] },
                { step: "calculate_churn_score", depends_on: ["identify_risk_factors"] }
            ]
            
            step_prompts: {
                "gather_customer_data": "data_collection",
                "analyze_usage_patterns": "usage_analysis", 
                "identify_risk_factors": "risk_identification",
                "calculate_churn_score": "churn_scoring"
            }
        }
    """
    name: str
    problem: str
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    step_prompts: Dict[str, str] = field(default_factory=dict)
    step_tools: Dict[str, List[str]] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    intermediate_validation: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphBasedPlanner(ASTNode):
    """
    Graph-based planner using search algorithms over state space.
    
    Models the problem as a graph where:
    - Nodes represent states
    - Edges represent actions/transitions
    - Goal is to find path from initial to goal state
    
    Example DSL:
        planner graph_workflow {
            initial_state: { status: "new", priority: "medium" }
            goal_state: { status: "resolved" }
            
            search_policy: beam_search {
                beam_width: 3
                max_depth: 8
            }
            
            state_transitions: [
                { from: "new", action: "triage", to: "triaged", cost: 1 },
                { from: "triaged", action: "assign", to: "assigned", cost: 2 },
                { from: "assigned", action: "resolve", to: "resolved", cost: 5 }
            ]
            
            heuristic_function: "estimate_resolution_cost"
        }
    """
    name: str
    initial_state: Dict[str, Any]
    goal_state: Dict[str, Any]
    search_policy: SearchPolicy
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    heuristic_function: Optional[str] = None
    max_search_time: Optional[float] = 30.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Planning Integration with Chain System
# =============================================================================

@dataclass
class PlannerChainStep(ASTNode):
    """
    Extended chain step that can invoke planners.
    
    Integrates with existing ChainStep infrastructure to allow
    planners to be used within workflows alongside prompts, tools, etc.
    """
    planner_type: Literal["react", "chain_of_thought", "graph_based"]
    planner_ref: str  # Reference to planner definition
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    error_handling: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanningWorkflow(ASTNode):
    """
    High-level workflow that combines multiple planners.
    
    Orchestrates complex reasoning by chaining different types
    of planners and integrating with existing workflow primitives.
    
    Example DSL:
        define planning_workflow incident_response {
            input_schema: IncidentAlert
            
            stage initial_assessment {
                planner: react_analyzer
                goal: "Understand incident scope and impact"
            }
            
            stage resolution_planning {
                planner: cot_planner  
                depends_on: initial_assessment
                goal: "Create step-by-step resolution plan"
            }
            
            stage execution {
                planner: graph_executor
                depends_on: resolution_planning
                goal: "Execute resolution plan optimally"
            }
        }
    """
    name: str
    input_schema: Optional[str] = None
    output_schema: Optional[str] = None
    stages: List[Dict[str, Any]] = field(default_factory=list)
    stage_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Union Type for All Planning Nodes
# =============================================================================

PlannerNode = Union[
    ReActPlanner,
    ChainOfThoughtPlanner, 
    GraphBasedPlanner,
    PlannerChainStep,
    PlanningWorkflow
]

__all__ = [
    "SearchPolicy",
    "BeamSearchPolicy", 
    "GreedySearchPolicy",
    "MCTSPolicy",
    "PlannerStep",
    "ReActPlanner",
    "ChainOfThoughtPlanner",
    "GraphBasedPlanner",
    "PlannerChainStep",
    "PlanningWorkflow",
    "PlannerNode"
]