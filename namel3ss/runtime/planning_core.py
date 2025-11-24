"""
Planning and Reasoning Runtime Core for Namel3ss

This module implements the core planning engines that execute planning strategies
within the Namel3ss runtime environment. It provides ReAct, Chain-of-Thought,
and Graph-based planners with search policies and reasoning capabilities.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, Callable, AsyncGenerator
from enum import Enum
import logging

from ..ir.spec import PlannerSpec, SearchPolicySpec, TypeSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Planning State Management
# =============================================================================

class PlanningStatus(Enum):
    """Status of a planning execution"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class PlanningContext:
    """Runtime context for planning operations"""
    planner_id: str
    session_id: str
    goal: str
    initial_state: Dict[str, Any]
    current_state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def add_step(self, step_type: str, content: Dict[str, Any]):
        """Add a step to the planning history"""
        step = {
            "timestamp": time.time(),
            "step_type": step_type,
            "content": content,
            "state_snapshot": self.current_state.copy()
        }
        self.history.append(step)
    
    def update_state(self, updates: Dict[str, Any]):
        """Update current planning state"""
        self.current_state.update(updates)
        
    def elapsed_time(self) -> float:
        """Get elapsed planning time in seconds"""
        return time.time() - self.start_time


@dataclass
class PlanningResult:
    """Result of a planning operation"""
    status: PlanningStatus
    plan: List[Dict[str, Any]]
    final_state: Dict[str, Any]
    execution_trace: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    error: Optional[str] = None


# =============================================================================
# Abstract Planner Interface
# =============================================================================

class BasePlanner(ABC):
    """Abstract base class for all planners"""
    
    def __init__(self, spec: PlannerSpec, tool_registry: Dict[str, Callable]):
        self.spec = spec
        self.tool_registry = tool_registry
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def plan(self, context: PlanningContext) -> PlanningResult:
        """Execute the planning strategy"""
        pass
    
    @abstractmethod
    async def step(self, context: PlanningContext) -> Dict[str, Any]:
        """Execute a single planning step"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input against planner's input schema"""
        # TODO: Implement schema validation using self.spec.input_schema
        return True
    
    def format_output(self, result: PlanningResult) -> Dict[str, Any]:
        """Format output according to planner's output schema"""
        return {
            "status": result.status.value,
            "plan": result.plan,
            "final_state": result.final_state,
            "metrics": result.metrics,
            "error": result.error
        }


# =============================================================================
# ReAct Planner Implementation
# =============================================================================

class ReActPlanner(BasePlanner):
    """
    ReAct (Reasoning + Acting) planner implementation.
    
    Interleaves reasoning steps with action execution:
    1. Thought: Analyze current state and plan next action
    2. Action: Execute tool calls or operations  
    3. Observation: Process results and update state
    4. Repeat until goal achieved or max cycles reached
    """
    
    async def plan(self, context: PlanningContext) -> PlanningResult:
        """Execute ReAct planning strategy"""
        plan = []
        metrics = {
            "cycles_completed": 0,
            "actions_taken": 0,
            "reasoning_steps": 0,
            "tool_calls": 0,
            "success": False
        }
        
        try:
            self.logger.info(f"Starting ReAct planning for goal: {context.goal}")
            context.add_step("planning_start", {"goal": context.goal})
            
            max_cycles = self.spec.max_cycles or 10
            
            for cycle in range(max_cycles):
                # Reasoning step
                thought_result = await self._reasoning_step(context)
                plan.append(thought_result)
                metrics["reasoning_steps"] += 1
                
                # Check if goal is achieved
                if await self._check_success_condition(context):
                    metrics["success"] = True
                    break
                
                # Action step
                action_result = await self._action_step(context, thought_result)
                plan.append(action_result)
                metrics["actions_taken"] += 1
                
                if action_result.get("tool_called"):
                    metrics["tool_calls"] += 1
                
                # Observation step
                observation_result = await self._observation_step(context, action_result)
                plan.append(observation_result)
                
                metrics["cycles_completed"] = cycle + 1
                
                # Check timeout
                if context.elapsed_time() > (self.spec.timeout_seconds or 60):
                    return PlanningResult(
                        status=PlanningStatus.TIMEOUT,
                        plan=plan,
                        final_state=context.current_state,
                        execution_trace=context.history,
                        metrics=metrics,
                        error="Planning timeout exceeded"
                    )
            
            # Final success check
            if not metrics["success"] and self.spec.fallback_action:
                fallback_result = await self._execute_fallback(context)
                plan.append(fallback_result)
            
            status = PlanningStatus.COMPLETED if metrics["success"] else PlanningStatus.FAILED
            
            return PlanningResult(
                status=status,
                plan=plan,
                final_state=context.current_state,
                execution_trace=context.history,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"ReAct planning failed: {str(e)}")
            return PlanningResult(
                status=PlanningStatus.FAILED,
                plan=plan,
                final_state=context.current_state,
                execution_trace=context.history,
                metrics=metrics,
                error=str(e)
            )
    
    async def step(self, context: PlanningContext) -> Dict[str, Any]:
        """Execute a single ReAct step (thought + action + observation)"""
        thought = await self._reasoning_step(context)
        action = await self._action_step(context, thought)
        observation = await self._observation_step(context, action)
        
        return {
            "thought": thought,
            "action": action,
            "observation": observation
        }
    
    async def _reasoning_step(self, context: PlanningContext) -> Dict[str, Any]:
        """Execute reasoning/thought step"""
        prompt_data = {
            "goal": context.goal,
            "current_state": context.current_state,
            "history": context.history[-3:],  # Last 3 steps for context
            "available_tools": context.available_tools
        }
        
        # Call reasoning prompt
        if self.spec.reasoning_prompt and self.spec.reasoning_prompt in self.tool_registry:
            reasoning_tool = self.tool_registry[self.spec.reasoning_prompt]
            thought_result = await reasoning_tool(prompt_data)
        else:
            # Default reasoning logic
            thought_result = {
                "reasoning": f"Analyzing current state to achieve goal: {context.goal}",
                "next_action": "continue_planning"
            }
        
        context.add_step("reasoning", thought_result)
        return {
            "step_type": "thought",
            "content": thought_result,
            "timestamp": time.time()
        }
    
    async def _action_step(self, context: PlanningContext, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action step based on reasoning"""
        action_name = thought.get("content", {}).get("next_action", "continue")
        
        if action_name in self.tool_registry and action_name in self.spec.action_tools:
            try:
                tool = self.tool_registry[action_name]
                action_input = thought.get("content", {}).get("action_input", {})
                action_result = await tool(action_input)
                
                context.add_step("action", {
                    "tool": action_name,
                    "input": action_input,
                    "result": action_result
                })
                
                return {
                    "step_type": "action",
                    "tool_called": True,
                    "tool_name": action_name,
                    "result": action_result,
                    "timestamp": time.time()
                }
            except Exception as e:
                error_result = {"error": str(e), "tool": action_name}
                context.add_step("action_error", error_result)
                return {
                    "step_type": "action",
                    "tool_called": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        else:
            # No action or invalid tool
            return {
                "step_type": "action",
                "tool_called": False,
                "message": f"No valid action: {action_name}",
                "timestamp": time.time()
            }
    
    async def _observation_step(self, context: PlanningContext, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process action results and update state"""
        observation = {
            "action_successful": action_result.get("tool_called", False),
            "result_summary": str(action_result.get("result", "No result")),
            "state_updates": {}
        }
        
        # Update context state based on action results
        if action_result.get("result"):
            result_data = action_result["result"]
            if isinstance(result_data, dict):
                context.update_state(result_data)
                observation["state_updates"] = result_data
        
        context.add_step("observation", observation)
        return {
            "step_type": "observation", 
            "content": observation,
            "timestamp": time.time()
        }
    
    async def _check_success_condition(self, context: PlanningContext) -> bool:
        """Check if the success condition is met"""
        if not self.spec.success_condition:
            return False
        
        # TODO: Implement expression evaluation against context.current_state
        # For now, simple key-value check
        condition = self.spec.success_condition
        if isinstance(condition, dict):
            for key, expected_value in condition.items():
                if context.current_state.get(key) != expected_value:
                    return False
            return True
        
        return False
    
    async def _execute_fallback(self, context: PlanningContext) -> Dict[str, Any]:
        """Execute fallback action when planning fails"""
        if self.spec.fallback_action and self.spec.fallback_action in self.tool_registry:
            try:
                fallback_tool = self.tool_registry[self.spec.fallback_action]
                fallback_result = await fallback_tool({"context": context.current_state})
                
                context.add_step("fallback", {
                    "action": self.spec.fallback_action,
                    "result": fallback_result
                })
                
                return {
                    "step_type": "fallback",
                    "action": self.spec.fallback_action,
                    "result": fallback_result,
                    "timestamp": time.time()
                }
            except Exception as e:
                return {
                    "step_type": "fallback",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        return {
            "step_type": "fallback",
            "message": "No fallback action configured",
            "timestamp": time.time()
        }


# =============================================================================
# Chain-of-Thought Planner Implementation  
# =============================================================================

class ChainOfThoughtPlanner(BasePlanner):
    """
    Chain-of-Thought planner for step-by-step reasoning.
    
    Decomposes complex problems into sequential reasoning steps,
    with each step building on previous conclusions.
    """
    
    async def plan(self, context: PlanningContext) -> PlanningResult:
        """Execute Chain-of-Thought planning strategy"""
        plan = []
        metrics = {
            "steps_completed": 0,
            "reasoning_steps": len(self.spec.reasoning_steps),
            "tool_calls": 0,
            "success": False
        }
        
        try:
            self.logger.info(f"Starting CoT planning for problem: {self.spec.goal}")
            context.add_step("cot_planning_start", {"goal": self.spec.goal})
            
            # Execute reasoning steps in dependency order
            executed_steps = set()
            step_results = {}
            
            while len(executed_steps) < len(self.spec.reasoning_steps):
                progress_made = False
                
                for step_config in self.spec.reasoning_steps:
                    step_name = step_config["step"]
                    
                    if step_name in executed_steps:
                        continue
                    
                    # Check if dependencies are satisfied
                    dependencies = self.spec.dependencies.get(step_name, [])
                    if all(dep in executed_steps for dep in dependencies):
                        # Execute this step
                        step_result = await self._execute_reasoning_step(
                            context, step_name, step_config, step_results
                        )
                        
                        plan.append(step_result)
                        step_results[step_name] = step_result
                        executed_steps.add(step_name)
                        metrics["steps_completed"] += 1
                        progress_made = True
                        
                        if step_result.get("tool_called"):
                            metrics["tool_calls"] += 1
                
                if not progress_made:
                    raise Exception("Circular dependency detected in reasoning steps")
                
                # Check timeout
                if context.elapsed_time() > (self.spec.timeout_seconds or 120):
                    return PlanningResult(
                        status=PlanningStatus.TIMEOUT,
                        plan=plan,
                        final_state=context.current_state,
                        execution_trace=context.history,
                        metrics=metrics,
                        error="Planning timeout exceeded"
                    )
            
            metrics["success"] = True
            
            return PlanningResult(
                status=PlanningStatus.COMPLETED,
                plan=plan,
                final_state=context.current_state,
                execution_trace=context.history,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"CoT planning failed: {str(e)}")
            return PlanningResult(
                status=PlanningStatus.FAILED,
                plan=plan,
                final_state=context.current_state,
                execution_trace=context.history,
                metrics=metrics,
                error=str(e)
            )
    
    async def step(self, context: PlanningContext) -> Dict[str, Any]:
        """Execute a single reasoning step"""
        # Find next step to execute based on dependencies
        for step_config in self.spec.reasoning_steps:
            step_name = step_config["step"]
            dependencies = self.spec.dependencies.get(step_name, [])
            
            # Check if dependencies are satisfied and step not yet executed
            executed_steps = {entry["content"].get("step_name") for entry in context.history 
                            if entry.get("step_type") == "reasoning_step"}
            
            if (all(dep in executed_steps for dep in dependencies) and 
                step_name not in executed_steps):
                
                return await self._execute_reasoning_step(
                    context, step_name, step_config, {}
                )
        
        return {"message": "No more steps to execute", "timestamp": time.time()}
    
    async def _execute_reasoning_step(self, context: PlanningContext, step_name: str,
                                    step_config: Dict[str, Any], 
                                    previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reasoning step"""
        
        step_input = {
            "step_name": step_name,
            "goal": context.goal,
            "current_state": context.current_state,
            "previous_results": previous_results,
            "step_config": step_config
        }
        
        # Use step-specific prompt if available
        prompt_name = self.spec.step_prompts.get(step_name)
        if prompt_name and prompt_name in self.tool_registry:
            try:
                prompt_tool = self.tool_registry[prompt_name]
                reasoning_result = await prompt_tool(step_input)
            except Exception as e:
                reasoning_result = {"error": str(e), "step": step_name}
        else:
            # Default reasoning
            reasoning_result = {
                "step": step_name,
                "reasoning": f"Executing reasoning step: {step_name}",
                "conclusion": "Step completed"
            }
        
        # Execute step-specific tools if configured
        tool_calls = []
        step_tools = self.spec.step_tools.get(step_name, [])
        for tool_name in step_tools:
            if tool_name in self.tool_registry:
                try:
                    tool = self.tool_registry[tool_name]
                    tool_result = await tool({
                        "reasoning": reasoning_result,
                        "context": context.current_state
                    })
                    tool_calls.append({
                        "tool": tool_name,
                        "result": tool_result
                    })
                except Exception as e:
                    tool_calls.append({
                        "tool": tool_name,
                        "error": str(e)
                    })
        
        # Update context state
        if reasoning_result and isinstance(reasoning_result, dict):
            state_updates = reasoning_result.get("state_updates", {})
            if state_updates:
                context.update_state(state_updates)
        
        step_result = {
            "step_type": "reasoning_step",
            "step_name": step_name,
            "reasoning": reasoning_result,
            "tool_calls": tool_calls,
            "tool_called": len(tool_calls) > 0,
            "timestamp": time.time()
        }
        
        context.add_step("reasoning_step", step_result)
        return step_result


# =============================================================================
# Graph-based Planner Implementation
# =============================================================================

class GraphBasedPlanner(BasePlanner):
    """
    Graph-based planner using search algorithms over state space.
    
    Models the problem as a graph where:
    - Nodes represent states
    - Edges represent actions/transitions
    - Goal is to find path from initial to goal state
    """
    
    def __init__(self, spec: PlannerSpec, tool_registry: Dict[str, Callable]):
        super().__init__(spec, tool_registry)
        self.search_policy = self._create_search_policy(spec.search_policy)
    
    def _create_search_policy(self, policy_config: Optional[Dict[str, Any]]):
        """Create search policy from configuration"""
        if not policy_config:
            return GreedySearch()
        
        policy_type = policy_config.get("policy_type", "greedy_search")
        
        if policy_type == "beam_search":
            return BeamSearch(
                beam_width=policy_config.get("beam_width", 3),
                max_depth=policy_config.get("max_depth", 10),
                scoring_function=policy_config.get("scoring_function")
            )
        elif policy_type == "mcts":
            return MCTSSearch(
                num_simulations=policy_config.get("num_simulations", 100),
                exploration_constant=policy_config.get("exploration_constant", 1.41),
                max_simulation_depth=policy_config.get("max_simulation_depth", 15)
            )
        else:
            return GreedySearch(
                max_steps=policy_config.get("max_steps", 20),
                confidence_threshold=policy_config.get("confidence_threshold", 0.8)
            )
    
    async def plan(self, context: PlanningContext) -> PlanningResult:
        """Execute graph-based planning strategy"""
        plan = []
        metrics = {
            "states_explored": 0,
            "transitions_evaluated": 0,
            "search_depth": 0,
            "success": False
        }
        
        try:
            self.logger.info(f"Starting graph-based planning with {self.search_policy.__class__.__name__}")
            
            # Initialize search state
            initial_state = self.spec.initial_state or context.initial_state
            goal_state = self.spec.goal_state or {"status": "completed"}
            
            search_result = await self.search_policy.search(
                initial_state=initial_state,
                goal_state=goal_state,
                transitions=self.spec.state_transitions,
                context=context,
                planner=self
            )
            
            plan = search_result.get("path", [])
            metrics.update(search_result.get("metrics", {}))
            metrics["success"] = search_result.get("success", False)
            
            status = PlanningStatus.COMPLETED if metrics["success"] else PlanningStatus.FAILED
            
            return PlanningResult(
                status=status,
                plan=plan,
                final_state=search_result.get("final_state", context.current_state),
                execution_trace=context.history,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Graph-based planning failed: {str(e)}")
            return PlanningResult(
                status=PlanningStatus.FAILED,
                plan=plan,
                final_state=context.current_state,
                execution_trace=context.history,
                metrics=metrics,
                error=str(e)
            )
    
    async def step(self, context: PlanningContext) -> Dict[str, Any]:
        """Execute a single graph search step"""
        return await self.search_policy.step(context, self)
    
    async def evaluate_state(self, state: Dict[str, Any], goal_state: Dict[str, Any]) -> float:
        """Evaluate how close a state is to the goal"""
        if self.spec.heuristic_function and self.spec.heuristic_function in self.tool_registry:
            try:
                heuristic_tool = self.tool_registry[self.spec.heuristic_function]
                result = await heuristic_tool({
                    "current_state": state,
                    "goal_state": goal_state
                })
                return float(result.get("score", 0.0))
            except Exception as e:
                self.logger.warning(f"Heuristic function failed: {e}")
        
        # Default heuristic: count matching key-value pairs
        matches = sum(1 for key, value in goal_state.items() 
                     if state.get(key) == value)
        return matches / len(goal_state) if goal_state else 0.0
    
    async def get_possible_transitions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get possible transitions from current state"""
        possible = []
        
        for transition in self.spec.state_transitions:
            from_state = transition.get("from")
            
            # Check if transition applies to current state
            if (isinstance(from_state, str) and state.get("status") == from_state) or \
               (isinstance(from_state, dict) and all(state.get(k) == v for k, v in from_state.items())):
                possible.append(transition)
        
        return possible
    
    async def apply_transition(self, state: Dict[str, Any], transition: Dict[str, Any],
                              context: PlanningContext) -> Dict[str, Any]:
        """Apply a transition to get the next state"""
        action = transition.get("action")
        to_state = transition.get("to")
        cost = transition.get("cost", 1)
        
        # Execute action if it's a tool
        if action in self.tool_registry:
            try:
                tool = self.tool_registry[action]
                action_result = await tool({
                    "current_state": state,
                    "transition": transition
                })
                
                # Update state based on action result
                new_state = state.copy()
                if isinstance(action_result, dict):
                    new_state.update(action_result)
                
                # Apply transition target state
                if isinstance(to_state, str):
                    new_state["status"] = to_state
                elif isinstance(to_state, dict):
                    new_state.update(to_state)
                
                context.add_step("transition", {
                    "from_state": state,
                    "action": action,
                    "to_state": new_state,
                    "cost": cost,
                    "tool_result": action_result
                })
                
                return new_state
                
            except Exception as e:
                self.logger.error(f"Transition action '{action}' failed: {e}")
                return state  # Return unchanged state on error
        else:
            # Simple state transition without tool execution
            new_state = state.copy()
            if isinstance(to_state, str):
                new_state["status"] = to_state
            elif isinstance(to_state, dict):
                new_state.update(to_state)
            
            context.add_step("transition", {
                "from_state": state,
                "action": action,
                "to_state": new_state,
                "cost": cost
            })
            
            return new_state


# =============================================================================
# Search Policy Implementations
# =============================================================================

class SearchPolicy(ABC):
    """Abstract base class for search policies"""
    
    @abstractmethod
    async def search(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any],
                    transitions: List[Dict[str, Any]], context: PlanningContext,
                    planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute search strategy"""
        pass
    
    @abstractmethod
    async def step(self, context: PlanningContext, planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute single search step"""
        pass


class GreedySearch(SearchPolicy):
    """Greedy search policy for single-path exploration"""
    
    def __init__(self, max_steps: int = 20, confidence_threshold: float = 0.8):
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
    
    async def search(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any],
                    transitions: List[Dict[str, Any]], context: PlanningContext,
                    planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute greedy search"""
        path = []
        current_state = initial_state
        metrics = {
            "states_explored": 1,
            "transitions_evaluated": 0,
            "search_depth": 0
        }
        
        for step in range(self.max_steps):
            # Check if goal reached
            goal_score = await planner.evaluate_state(current_state, goal_state)
            if goal_score >= self.confidence_threshold:
                return {
                    "success": True,
                    "path": path,
                    "final_state": current_state,
                    "metrics": metrics
                }
            
            # Get possible transitions
            possible_transitions = await planner.get_possible_transitions(current_state)
            if not possible_transitions:
                break
            
            # Choose best transition greedily
            best_transition = None
            best_score = -1
            
            for transition in possible_transitions:
                next_state = await planner.apply_transition(current_state, transition, context)
                score = await planner.evaluate_state(next_state, goal_state)
                metrics["transitions_evaluated"] += 1
                
                if score > best_score:
                    best_score = score
                    best_transition = transition
            
            if best_transition:
                current_state = await planner.apply_transition(current_state, best_transition, context)
                path.append({
                    "action": best_transition.get("action"),
                    "from_state": path[-1]["to_state"] if path else initial_state,
                    "to_state": current_state,
                    "score": best_score
                })
                metrics["states_explored"] += 1
                metrics["search_depth"] = len(path)
            else:
                break
        
        return {
            "success": False,
            "path": path,
            "final_state": current_state,
            "metrics": metrics
        }
    
    async def step(self, context: PlanningContext, planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute single greedy search step"""
        return {"message": "Greedy search step", "timestamp": time.time()}


class BeamSearch(SearchPolicy):
    """Beam search policy for exploring multiple solution paths"""
    
    def __init__(self, beam_width: int = 3, max_depth: int = 10, scoring_function: Optional[str] = None):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.scoring_function = scoring_function
    
    async def search(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any],
                    transitions: List[Dict[str, Any]], context: PlanningContext,
                    planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute beam search"""
        # Initialize beam with initial state
        beam = [{
            "state": initial_state,
            "path": [],
            "score": await planner.evaluate_state(initial_state, goal_state),
            "depth": 0
        }]
        
        metrics = {
            "states_explored": 1,
            "transitions_evaluated": 0,
            "search_depth": 0,
            "beam_expansions": 0
        }
        
        for depth in range(self.max_depth):
            new_beam = []
            
            for beam_state in beam:
                current_state = beam_state["state"]
                current_path = beam_state["path"]
                
                # Check if goal reached
                if beam_state["score"] >= 0.95:  # High confidence threshold
                    return {
                        "success": True,
                        "path": current_path,
                        "final_state": current_state,
                        "metrics": metrics
                    }
                
                # Expand this beam state
                possible_transitions = await planner.get_possible_transitions(current_state)
                
                for transition in possible_transitions:
                    next_state = await planner.apply_transition(current_state, transition, context)
                    next_score = await planner.evaluate_state(next_state, goal_state)
                    
                    new_path = current_path + [{
                        "action": transition.get("action"),
                        "from_state": current_state,
                        "to_state": next_state,
                        "score": next_score
                    }]
                    
                    new_beam.append({
                        "state": next_state,
                        "path": new_path,
                        "score": next_score,
                        "depth": depth + 1
                    })
                    
                    metrics["transitions_evaluated"] += 1
                    metrics["states_explored"] += 1
            
            # Keep only top beam_width states
            new_beam.sort(key=lambda x: x["score"], reverse=True)
            beam = new_beam[:self.beam_width]
            metrics["beam_expansions"] += 1
            metrics["search_depth"] = depth + 1
            
            if not beam:
                break
        
        # Return best result from final beam
        if beam:
            best = beam[0]
            return {
                "success": best["score"] >= 0.8,
                "path": best["path"],
                "final_state": best["state"],
                "metrics": metrics
            }
        
        return {
            "success": False,
            "path": [],
            "final_state": initial_state,
            "metrics": metrics
        }
    
    async def step(self, context: PlanningContext, planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute single beam search step"""
        return {"message": "Beam search step", "timestamp": time.time()}


class MCTSSearch(SearchPolicy):
    """Monte Carlo Tree Search policy"""
    
    def __init__(self, num_simulations: int = 100, exploration_constant: float = 1.41,
                 max_simulation_depth: int = 15):
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = max_simulation_depth
    
    async def search(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any],
                    transitions: List[Dict[str, Any]], context: PlanningContext,
                    planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute MCTS search"""
        # Simplified MCTS implementation
        # In a full implementation, this would build a proper search tree
        
        best_path = []
        best_score = 0
        metrics = {
            "states_explored": 0,
            "simulations_run": 0,
            "search_depth": 0
        }
        
        for simulation in range(self.num_simulations):
            # Run simulation
            sim_result = await self._run_simulation(
                initial_state, goal_state, transitions, context, planner
            )
            
            if sim_result["score"] > best_score:
                best_score = sim_result["score"]
                best_path = sim_result["path"]
            
            metrics["simulations_run"] += 1
            metrics["states_explored"] += sim_result["states_explored"]
            metrics["search_depth"] = max(metrics["search_depth"], len(sim_result["path"]))
        
        return {
            "success": best_score >= 0.8,
            "path": best_path,
            "final_state": best_path[-1]["to_state"] if best_path else initial_state,
            "metrics": metrics
        }
    
    async def _run_simulation(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any],
                             transitions: List[Dict[str, Any]], context: PlanningContext,
                             planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Run a single MCTS simulation"""
        path = []
        current_state = initial_state
        states_explored = 1
        
        for step in range(self.max_simulation_depth):
            score = await planner.evaluate_state(current_state, goal_state)
            if score >= 0.95:
                break
            
            possible_transitions = await planner.get_possible_transitions(current_state)
            if not possible_transitions:
                break
            
            # Random selection for simulation (can be improved with UCB1)
            import random
            transition = random.choice(possible_transitions)
            
            next_state = await planner.apply_transition(current_state, transition, context)
            path.append({
                "action": transition.get("action"),
                "from_state": current_state,
                "to_state": next_state,
                "score": score
            })
            
            current_state = next_state
            states_explored += 1
        
        final_score = await planner.evaluate_state(current_state, goal_state)
        
        return {
            "path": path,
            "score": final_score,
            "states_explored": states_explored
        }
    
    async def step(self, context: PlanningContext, planner: GraphBasedPlanner) -> Dict[str, Any]:
        """Execute single MCTS step"""
        return {"message": "MCTS search step", "timestamp": time.time()}


# =============================================================================
# Planning Engine Factory
# =============================================================================

class PlanningEngine:
    """Factory and registry for planning engines"""
    
    def __init__(self, tool_registry: Dict[str, Callable]):
        self.tool_registry = tool_registry
        self.planners = {}
    
    def create_planner(self, spec: PlannerSpec) -> BasePlanner:
        """Create planner instance from specification"""
        if spec.planner_type == "react":
            return ReActPlanner(spec, self.tool_registry)
        elif spec.planner_type == "chain_of_thought":
            return ChainOfThoughtPlanner(spec, self.tool_registry)
        elif spec.planner_type == "graph_based":
            return GraphBasedPlanner(spec, self.tool_registry)
        else:
            raise ValueError(f"Unknown planner type: {spec.planner_type}")
    
    def register_planner(self, name: str, planner: BasePlanner):
        """Register planner instance"""
        self.planners[name] = planner
    
    def get_planner(self, name: str) -> Optional[BasePlanner]:
        """Get registered planner by name"""
        return self.planners.get(name)
    
    async def execute_planner(self, planner_name: str, input_data: Dict[str, Any],
                             session_id: str = "default") -> PlanningResult:
        """Execute a registered planner"""
        planner = self.get_planner(planner_name)
        if not planner:
            raise ValueError(f"Planner not found: {planner_name}")
        
        context = PlanningContext(
            planner_id=planner_name,
            session_id=session_id,
            goal=planner.spec.goal,
            initial_state=input_data,
            current_state=input_data.copy(),
            available_tools=list(self.tool_registry.keys())
        )
        
        return await planner.plan(context)


__all__ = [
    "PlanningStatus",
    "PlanningContext", 
    "PlanningResult",
    "BasePlanner",
    "ReActPlanner",
    "ChainOfThoughtPlanner",
    "GraphBasedPlanner",
    "SearchPolicy",
    "GreedySearch",
    "BeamSearch",
    "MCTSSearch", 
    "PlanningEngine"
]