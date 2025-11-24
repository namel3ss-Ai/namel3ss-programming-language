"""
Planning and Reasoning Runtime Functions

This module provides runtime functions for executing planners and planning workflows
within the generated backend. It integrates with the existing runtime infrastructure
to provide planning capabilities alongside chains, prompts, and other AI primitives.
"""

from textwrap import dedent


def render_planning_runtime_block() -> str:
    """Generate runtime code for planning and reasoning execution."""
    
    return dedent('''
# =============================================================================
# Planning and Reasoning Runtime
# =============================================================================

# Planning registries
PLANNERS: Dict[str, Dict[str, Any]] = {}
PLANNING_WORKFLOWS: Dict[str, Dict[str, Any]] = {}

# Planning engine instance (initialized lazily)
_planning_engine: Optional[Any] = None


def _init_planning_engine():
    """Initialize the planning engine with tool registry."""
    global _planning_engine
    
    if _planning_engine is not None:
        return _planning_engine
    
    try:
        from namel3ss.runtime.planning_core import PlanningEngine
        from namel3ss.runtime.planning_integration import PlannerChainAdapter
        
        # Build tool registry from runtime functions
        tool_registry = {
            # LLM connectors
            **{name: lambda payload, n=name: call_llm_connector(n, payload) 
               for name in AI_CONNECTORS.keys()},
            
            # Prompts  
            **{name: lambda payload, n=name: run_prompt(n, payload)
               for name in PROMPTS.keys()},
            
            # Chains
            **{name: lambda payload, n=name: run_chain(n, payload)
               for name in AI_CHAINS.keys()},
            
            # Tools (if available)
            **({name: spec.get("implementation") 
                for name, spec in TOOLS.items() 
                if spec.get("implementation")} if "TOOLS" in globals() else {}),
        }
        
        _planning_engine = PlanningEngine(tool_registry)
        
        # Register planners from the spec
        for name, planner_spec in PLANNERS.items():
            try:
                from namel3ss.ir.spec import PlannerSpec
                
                # Create PlannerSpec from runtime data
                spec = PlannerSpec(
                    name=name,
                    planner_type=planner_spec.get("planner_type", "generic"),
                    goal=planner_spec.get("goal", "Complete planning task"),
                    input_schema=planner_spec.get("input_schema", {}),
                    output_schema=planner_spec.get("output_schema", {}),
                    
                    # ReAct configuration
                    max_cycles=planner_spec.get("max_cycles"),
                    reasoning_prompt=planner_spec.get("reasoning_prompt"),
                    action_tools=planner_spec.get("action_tools", []),
                    success_condition=planner_spec.get("success_condition"),
                    fallback_action=planner_spec.get("fallback_action"),
                    
                    # Chain-of-Thought configuration
                    reasoning_steps=planner_spec.get("reasoning_steps", []),
                    step_prompts=planner_spec.get("step_prompts", {}),
                    step_tools=planner_spec.get("step_tools", {}),
                    dependencies=planner_spec.get("dependencies", {}),
                    
                    # Graph-based configuration
                    initial_state=planner_spec.get("initial_state"),
                    goal_state=planner_spec.get("goal_state"),
                    search_policy=planner_spec.get("search_policy"),
                    state_transitions=planner_spec.get("state_transitions", []),
                    heuristic_function=planner_spec.get("heuristic_function"),
                    max_search_time=planner_spec.get("max_search_time"),
                    
                    # Security and performance
                    allowed_tools=planner_spec.get("allowed_tools", []),
                    capabilities=planner_spec.get("capabilities", []),
                    permission_level=planner_spec.get("permission_level"),
                    timeout_seconds=planner_spec.get("timeout_seconds"),
                    max_memory_usage=planner_spec.get("max_memory_usage"),
                    
                    metadata=planner_spec.get("metadata", {})
                )
                
                planner = _planning_engine.create_planner(spec)
                _planning_engine.register_planner(name, planner)
                
            except Exception as e:
                logger.warning(f"Failed to register planner '{name}': {e}")
        
        return _planning_engine
        
    except ImportError:
        logger.warning("Planning system not available - install namel3ss[planning]")
        return None


def list_planners() -> Dict[str, Any]:
    """List all available planners and their configurations."""
    return {
        name: {
            "name": name,
            "planner_type": spec.get("planner_type", "unknown"),
            "goal": spec.get("goal", ""),
            "description": spec.get("metadata", {}).get("description", ""),
            "input_schema": spec.get("input_schema", {}),
            "output_schema": spec.get("output_schema", {}),
        }
        for name, spec in PLANNERS.items()
    }


def list_planning_workflows() -> Dict[str, Any]:
    """List all available planning workflows."""
    return {
        name: {
            "name": name,
            "workflow_type": spec.get("workflow_type", "planning"),
            "stages": spec.get("stages", []),
            "stage_dependencies": spec.get("stage_dependencies", {}),
            "input_schema": spec.get("input_schema", {}),
            "output_schema": spec.get("output_schema", {}),
        }
        for name, spec in PLANNING_WORKFLOWS.items()
    }


async def execute_planner(
    planner_name: str,
    input_data: Dict[str, Any],
    session_id: str = "default",
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Execute a planner with the given input data."""
    
    start_time = time.time()
    
    # Check if planner exists
    if planner_name not in PLANNERS:
        return {
            "success": False,
            "status": "not_found",
            "error": f"Planner '{planner_name}' not found",
            "plan": [],
            "final_state": {},
            "metrics": {"elapsed_ms": _elapsed_time(start_time)},
            "execution_trace": []
        }
    
    # Initialize planning engine
    engine = _init_planning_engine()
    if engine is None:
        return {
            "success": False,
            "status": "unavailable", 
            "error": "Planning system not available",
            "plan": [],
            "final_state": {},
            "metrics": {"elapsed_ms": _elapsed_time(start_time)},
            "execution_trace": []
        }
    
    try:
        # Execute planner
        result = await engine.execute_planner(
            planner_name=planner_name,
            input_data=input_data,
            session_id=session_id
        )
        
        return {
            "success": result.status.value == "completed",
            "status": result.status.value,
            "plan": result.plan,
            "final_state": result.final_state,
            "metrics": {
                **result.metrics,
                "elapsed_ms": _elapsed_time(start_time)
            },
            "execution_trace": result.execution_trace,
            "error": result.error
        }
        
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": str(e),
            "plan": [],
            "final_state": {},
            "metrics": {"elapsed_ms": _elapsed_time(start_time)},
            "execution_trace": []
        }


async def execute_planner_step(
    planner_name: str,
    context: Dict[str, Any],
    session_id: str = "default"
) -> Dict[str, Any]:
    """Execute a single step of a planner."""
    
    # Check if planner exists
    if planner_name not in PLANNERS:
        return {
            "step_type": "error",
            "content": {"error": f"Planner '{planner_name}' not found"},
            "timestamp": time.time()
        }
    
    # Initialize planning engine
    engine = _init_planning_engine()
    if engine is None:
        return {
            "step_type": "error",
            "content": {"error": "Planning system not available"},
            "timestamp": time.time()
        }
    
    try:
        # Get planner instance
        planner = engine.get_planner(planner_name)
        if planner is None:
            return {
                "step_type": "error",
                "content": {"error": f"Planner '{planner_name}' not initialized"},
                "timestamp": time.time()
            }
        
        # Create planning context
        from namel3ss.runtime.planning_core import PlanningContext
        
        planning_context = PlanningContext(
            planner_id=planner_name,
            session_id=session_id,
            goal=planner.spec.goal,
            initial_state=context,
            current_state=context.copy()
        )
        
        # Execute single step
        step_result = await planner.step(planning_context)
        
        return {
            "step_type": step_result.get("step_type", "unknown"),
            "content": step_result.get("content", {}),
            "timestamp": step_result.get("timestamp", time.time())
        }
        
    except Exception as e:
        return {
            "step_type": "error",
            "content": {"error": str(e)},
            "timestamp": time.time()
        }


def get_planner_status(planner_name: str) -> Dict[str, Any]:
    """Get the current status and configuration of a planner."""
    
    if planner_name not in PLANNERS:
        raise KeyError(f"Planner '{planner_name}' not found")
    
    spec = PLANNERS[planner_name]
    
    return {
        "name": planner_name,
        "planner_type": spec.get("planner_type", "unknown"),
        "goal": spec.get("goal", ""),
        "status": "ready",
        "input_schema": spec.get("input_schema", {}),
        "output_schema": spec.get("output_schema", {}),
        "configuration": {
            key: value for key, value in spec.items()
            if key not in ["input_schema", "output_schema", "metadata"]
        },
        "metadata": spec.get("metadata", {})
    }


async def execute_planning_workflow(
    workflow_name: str,
    input_data: Dict[str, Any], 
    session_id: str = "default",
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Execute a multi-stage planning workflow."""
    
    start_time = time.time()
    
    # Check if workflow exists
    if workflow_name not in PLANNING_WORKFLOWS:
        return {
            "success": False,
            "workflow_name": workflow_name,
            "error": f"Planning workflow '{workflow_name}' not found",
            "stage_results": {},
            "final_state": {},
            "execution_trace": [],
            "metrics": {"elapsed_ms": _elapsed_time(start_time)}
        }
    
    # Initialize planning engine
    engine = _init_planning_engine()
    if engine is None:
        return {
            "success": False,
            "workflow_name": workflow_name,
            "error": "Planning system not available",
            "stage_results": {},
            "final_state": {},
            "execution_trace": [],
            "metrics": {"elapsed_ms": _elapsed_time(start_time)}
        }
    
    try:
        from namel3ss.runtime.planning_integration import PlanningWorkflowOrchestrator
        
        orchestrator = PlanningWorkflowOrchestrator(engine)
        workflow_spec = PLANNING_WORKFLOWS[workflow_name]
        
        result = await orchestrator.execute_planning_workflow(
            workflow_spec=workflow_spec,
            input_data=input_data,
            session_id=session_id
        )
        
        return {
            "success": result.get("success", False),
            "workflow_name": workflow_name,
            "stage_results": result.get("stage_results", {}),
            "final_state": result.get("final_state", {}),
            "execution_trace": result.get("execution_trace", []),
            "metrics": {
                **result.get("metrics", {}),
                "elapsed_ms": _elapsed_time(start_time)
            },
            "error": result.get("error")
        }
        
    except Exception as e:
        return {
            "success": False,
            "workflow_name": workflow_name,
            "error": str(e),
            "stage_results": {},
            "final_state": {},
            "execution_trace": [],
            "metrics": {"elapsed_ms": _elapsed_time(start_time)}
        }


def _elapsed_time(start_time: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return float(round((time.time() - start_time) * 1000.0, 3))
''').strip()


__all__ = ["render_planning_runtime_block"]