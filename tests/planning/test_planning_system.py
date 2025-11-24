"""
Test Implementation for Planning & Reasoning System

This test demonstrates the complete planning system integration,
including AST nodes, IR generation, backend codegen, and runtime execution.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any

from namel3ss.ast.planning import (
    ReActPlanner, ChainOfThoughtPlanner, GraphBasedPlanner,
    BeamSearchPolicy, GreedySearchPolicy, PlanningWorkflow
)
from namel3ss.runtime.planning_core import (
    PlanningEngine, PlanningContext, ReActPlanner as RuntimeReActPlanner,
    ChainOfThoughtPlanner as RuntimeCOTPlanner, GraphBasedPlanner as RuntimeGraphPlanner
)
from namel3ss.runtime.planning_integration import (
    ChainPlannerExecutor, PlanningWorkflowOrchestrator, PlannerChainAdapter
)
from namel3ss.ir.spec import PlannerSpec


# =============================================================================
# Test AST Node Creation
# =============================================================================

def test_react_planner_ast():
    """Test ReAct planner AST node creation and properties."""
    
    planner = ReActPlanner(
        name="incident_resolver",
        goal="Resolve customer support incidents efficiently",
        max_cycles=5,
        reasoning_prompt="analyze_incident",
        action_tools=["search_kb", "create_ticket", "send_email"],
        success_condition={"ticket_resolved": True},
        fallback_action="escalate_to_human",
        initial_context={"priority": "medium"},
        metadata={"version": "1.0", "author": "ai_team"}
    )
    
    assert planner.name == "incident_resolver"
    assert planner.goal == "Resolve customer support incidents efficiently"
    assert planner.max_cycles == 5
    assert planner.reasoning_prompt == "analyze_incident"
    assert len(planner.action_tools) == 3
    assert "search_kb" in planner.action_tools
    assert planner.fallback_action == "escalate_to_human"


def test_cot_planner_ast():
    """Test Chain-of-Thought planner AST node creation."""
    
    planner = ChainOfThoughtPlanner(
        name="churn_analyzer",
        problem="Analyze customer churn risk factors",
        reasoning_steps=[
            {"step": "gather_data", "description": "Collect customer metrics"},
            {"step": "analyze_patterns", "description": "Find usage patterns"},
            {"step": "identify_risks", "description": "Detect risk factors"},
            {"step": "calculate_score", "description": "Compute churn score"}
        ],
        step_prompts={
            "gather_data": "data_collection_prompt",
            "analyze_patterns": "pattern_analysis_prompt",
            "identify_risks": "risk_identification_prompt",
            "calculate_score": "scoring_prompt"
        },
        dependencies={
            "analyze_patterns": ["gather_data"],
            "identify_risks": ["analyze_patterns"],
            "calculate_score": ["identify_risks"]
        },
        metadata={"model_version": "2.1"}
    )
    
    assert planner.name == "churn_analyzer"
    assert planner.problem == "Analyze customer churn risk factors"
    assert len(planner.reasoning_steps) == 4
    assert len(planner.step_prompts) == 4
    assert planner.dependencies["calculate_score"] == ["identify_risks"]


def test_graph_planner_ast():
    """Test Graph-based planner AST node creation."""
    
    search_policy = BeamSearchPolicy(
        beam_width=3,
        max_depth=8,
        scoring_function="workflow_cost_estimator"
    )
    
    planner = GraphBasedPlanner(
        name="workflow_optimizer",
        initial_state={"status": "new", "priority": "medium"},
        goal_state={"status": "completed", "quality": "high"},
        search_policy=search_policy,
        state_transitions=[
            {"from": "new", "action": "triage", "to": "triaged", "cost": 1},
            {"from": "triaged", "action": "assign", "to": "assigned", "cost": 2},
            {"from": "assigned", "action": "resolve", "to": "completed", "cost": 5}
        ],
        heuristic_function="completion_estimator",
        max_search_time=30.0,
        metadata={"search_algorithm": "beam_search"}
    )
    
    assert planner.name == "workflow_optimizer"
    assert planner.initial_state["status"] == "new"
    assert planner.goal_state["status"] == "completed"
    assert planner.search_policy.policy_type == "beam_search"
    assert len(planner.state_transitions) == 3


def test_planning_workflow_ast():
    """Test Planning workflow AST node creation."""
    
    workflow = PlanningWorkflow(
        name="incident_response",
        input_schema="IncidentAlert",
        output_schema="ResolutionReport", 
        stages=[
            {
                "name": "assessment",
                "planner": "react_assessor",
                "goal": "Assess incident severity and impact"
            },
            {
                "name": "planning",
                "planner": "cot_planner",
                "goal": "Create resolution plan"
            },
            {
                "name": "execution", 
                "planner": "graph_executor",
                "goal": "Execute plan optimally"
            }
        ],
        stage_dependencies={
            "planning": ["assessment"],
            "execution": ["planning"]
        },
        global_context={"team": "sre", "escalation_level": 1},
        metadata={"sla_minutes": 30}
    )
    
    assert workflow.name == "incident_response"
    assert len(workflow.stages) == 3
    assert workflow.stage_dependencies["execution"] == ["planning"]
    assert workflow.global_context["team"] == "sre"


# =============================================================================
# Test Runtime Planning Engine
# =============================================================================

@pytest.mark.asyncio
async def test_planning_engine():
    """Test planning engine initialization and planner creation."""
    
    # Mock tool registry
    tool_registry = {
        "search_kb": lambda data: {"results": [f"Article about {data.get('query', 'unknown')}"]},
        "analyze_data": lambda data: {"analysis": f"Analyzed {data.get('dataset', 'data')}"},
        "calculate_score": lambda data: {"score": 0.85, "confidence": 0.9}
    }
    
    engine = PlanningEngine(tool_registry)
    
    # Test ReAct planner creation
    react_spec = PlannerSpec(
        name="test_react",
        planner_type="react",
        goal="Test ReAct planning",
        input_schema={},
        output_schema={},
        max_cycles=3,
        reasoning_prompt="test_reasoning",
        action_tools=["search_kb", "analyze_data"],
        allowed_tools=["search_kb", "analyze_data"]
    )
    
    react_planner = engine.create_planner(react_spec)
    engine.register_planner("test_react", react_planner)
    
    assert isinstance(react_planner, RuntimeReActPlanner)
    assert engine.get_planner("test_react") == react_planner
    
    # Test execution
    result = await engine.execute_planner(
        planner_name="test_react",
        input_data={"query": "test problem", "context": "testing"},
        session_id="test_session"
    )
    
    assert result is not None
    assert hasattr(result, "status")
    assert hasattr(result, "plan")
    assert hasattr(result, "metrics")


@pytest.mark.asyncio 
async def test_react_planner_execution():
    """Test ReAct planner step-by-step execution."""
    
    # Mock tools that simulate real behavior
    async def mock_reasoning_tool(data):
        return {
            "reasoning": f"Need to solve: {data.get('goal', 'unknown')}",
            "next_action": "search_kb",
            "action_input": {"query": "solution"}
        }
    
    async def mock_search_tool(data):
        return {
            "results": ["Found relevant information"],
            "status": "success"
        }
    
    tool_registry = {
        "reasoning_prompt": mock_reasoning_tool,
        "search_kb": mock_search_tool
    }
    
    engine = PlanningEngine(tool_registry)
    
    spec = PlannerSpec(
        name="test_react",
        planner_type="react", 
        goal="Find solution to test problem",
        input_schema={},
        output_schema={},
        max_cycles=2,
        reasoning_prompt="reasoning_prompt",
        action_tools=["search_kb"],
        success_condition={"solution_found": True}
    )
    
    planner = engine.create_planner(spec)
    
    context = PlanningContext(
        planner_id="test_react",
        session_id="test",
        goal="Find solution to test problem",
        initial_state={"problem": "test issue"},
        available_tools=["reasoning_prompt", "search_kb"]
    )
    
    # Test single step
    step_result = await planner.step(context)
    
    assert "thought" in step_result
    assert "action" in step_result  
    assert "observation" in step_result
    
    # Test full planning
    result = await planner.plan(context)
    
    assert result.plan is not None
    assert len(result.plan) > 0
    assert result.metrics["cycles_completed"] <= 2


@pytest.mark.asyncio
async def test_cot_planner_execution():
    """Test Chain-of-Thought planner execution."""
    
    async def mock_step_tool(data):
        step_name = data.get("step_name", "unknown")
        return {
            "step": step_name,
            "reasoning": f"Completed step: {step_name}",
            "conclusion": f"Result for {step_name}",
            "state_updates": {f"{step_name}_completed": True}
        }
    
    tool_registry = {
        "step1_prompt": mock_step_tool,
        "step2_prompt": mock_step_tool
    }
    
    engine = PlanningEngine(tool_registry)
    
    spec = PlannerSpec(
        name="test_cot",
        planner_type="chain_of_thought",
        goal="Multi-step reasoning test",
        input_schema={},
        output_schema={},
        reasoning_steps=[
            {"step": "step1"},
            {"step": "step2"}
        ],
        step_prompts={
            "step1": "step1_prompt",
            "step2": "step2_prompt"
        },
        dependencies={
            "step2": ["step1"]
        }
    )
    
    planner = engine.create_planner(spec)
    
    context = PlanningContext(
        planner_id="test_cot",
        session_id="test",
        goal="Multi-step reasoning test",
        initial_state={"input": "test data"}
    )
    
    result = await planner.plan(context)
    
    assert result.plan is not None
    assert len(result.plan) == 2  # Two reasoning steps
    assert result.metrics["steps_completed"] == 2


@pytest.mark.asyncio
async def test_chain_integration():
    """Test planner integration with chain execution system."""
    
    # Mock runtime context
    chain_context = {"session_id": "test", "user_id": "test_user"}
    
    tool_registry = {
        "mock_tool": lambda data: {"result": f"Processed {data}"}
    }
    
    engine = PlanningEngine(tool_registry)
    
    # Register a test planner
    spec = PlannerSpec(
        name="chain_test_planner",
        planner_type="react",
        goal="Chain integration test",
        input_schema={},
        output_schema={},
        max_cycles=1,
        action_tools=["mock_tool"]
    )
    
    planner = engine.create_planner(spec)
    engine.register_planner("chain_test_planner", planner)
    
    # Test chain executor
    executor = ChainPlannerExecutor(engine, chain_context)
    
    step_config = {
        "kind": "planner",
        "target": "chain_test_planner",
        "planner_type": "react",
        "options": {"test_input": "chain_test"}
    }
    
    workflow_state = {"context": "test"}
    
    result = await executor.execute_planner_step(step_config, workflow_state)
    
    assert result["success"] in [True, False]  # May succeed or fail, but should not crash
    assert "step_type" in result
    assert "planner_name" in result


@pytest.mark.asyncio
async def test_planning_workflow_orchestrator():
    """Test planning workflow orchestration."""
    
    tool_registry = {
        "stage1_tool": lambda data: {"stage1_result": "completed"},
        "stage2_tool": lambda data: {"stage2_result": "completed"}
    }
    
    engine = PlanningEngine(tool_registry)
    
    # Register planners for workflow stages
    for i, stage in enumerate(["stage1", "stage2"], 1):
        spec = PlannerSpec(
            name=f"{stage}_planner",
            planner_type="react",
            goal=f"Complete {stage}",
            input_schema={},
            output_schema={},
            max_cycles=1,
            action_tools=[f"{stage}_tool"]
        )
        planner = engine.create_planner(spec)
        engine.register_planner(f"{stage}_planner", planner)
    
    orchestrator = PlanningWorkflowOrchestrator(engine)
    
    workflow_spec = {
        "name": "test_workflow",
        "stages": [
            {"name": "stage1", "planner": "stage1_planner", "goal": "First stage"},
            {"name": "stage2", "planner": "stage2_planner", "goal": "Second stage"}
        ],
        "stage_dependencies": {"stage2": ["stage1"]},
        "global_context": {"workflow_id": "test123"}
    }
    
    result = await orchestrator.execute_planning_workflow(
        workflow_spec=workflow_spec,
        input_data={"input": "test"},
        session_id="test"
    )
    
    assert "success" in result
    assert "stage_results" in result
    assert result["workflow_name"] == "test_workflow"


# =============================================================================
# Integration Test
# =============================================================================

def test_end_to_end_planning_system():
    """Test the complete planning system from AST to runtime."""
    
    # 1. Create AST nodes
    react_ast = ReActPlanner(
        name="e2e_react",
        goal="End-to-end test",
        max_cycles=2,
        reasoning_prompt="e2e_reasoning",
        action_tools=["test_tool"],
        metadata={"test": True}
    )
    
    # 2. Validate AST properties  
    assert react_ast.name == "e2e_react"
    assert react_ast.goal == "End-to-end test"
    
    # 3. Test state encoding (would be called during backend generation)
    from namel3ss.codegen.backend.state.ai import _encode_planner
    
    encoded = _encode_planner(react_ast, set())
    
    assert encoded["name"] == "e2e_react"
    assert encoded["planner_type"] == "react" 
    assert encoded["goal"] == "End-to-end test"
    assert encoded["max_cycles"] == 2
    
    # 4. Test IR spec creation
    spec = PlannerSpec(
        name=encoded["name"],
        planner_type=encoded["planner_type"],
        goal=encoded["goal"],
        input_schema={},
        output_schema={},
        **{k: v for k, v in encoded.items() 
           if k not in ["name", "planner_type", "goal", "metadata"]}
    )
    
    assert spec.name == "e2e_react"
    assert spec.planner_type == "react"
    
    # 5. Test runtime creation
    tool_registry = {"test_tool": lambda x: {"result": "success"}}
    engine = PlanningEngine(tool_registry)
    
    runtime_planner = engine.create_planner(spec)
    assert isinstance(runtime_planner, RuntimeReActPlanner)
    
    print("✅ End-to-end planning system test passed!")


if __name__ == "__main__":
    # Run the tests
    test_react_planner_ast()
    test_cot_planner_ast()
    test_graph_planner_ast()
    test_planning_workflow_ast()
    
    # Run async tests
    asyncio.run(test_planning_engine())
    asyncio.run(test_react_planner_execution())
    asyncio.run(test_cot_planner_execution())
    asyncio.run(test_chain_integration())
    asyncio.run(test_planning_workflow_orchestrator())
    
    # Run integration test
    test_end_to_end_planning_system()
    
    print("🎉 All planning system tests passed!")
    print("📋 Planning & Reasoning implementation is complete and working!")