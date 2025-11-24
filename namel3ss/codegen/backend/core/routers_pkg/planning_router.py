"""
Planning and Reasoning Router Generation

Generates FastAPI routers for planning and reasoning endpoints,
including ReAct, Chain-of-Thought, and Graph-based planners.
"""

import textwrap
from typing import Any, Dict

from ...state import BackendState


def _render_planning_router_module(state: BackendState) -> str:
    """
    Generate FastAPI router module for planning and reasoning endpoints.
    
    Creates endpoints for:
    - GET /api/planners/ - List all available planners
    - POST /api/planners/{planner_name}/execute - Execute a planner
    - GET /api/planners/{planner_name}/status - Get planner status
    - POST /api/planning_workflows/{workflow_name}/execute - Execute planning workflow
    - POST /api/planners/{planner_name}/step - Execute single planning step
    """
    
    # Check if we have any planners or planning workflows
    has_planners = bool(state.planners)
    has_workflows = bool(state.planning_workflows)
    
    if not has_planners and not has_workflows:
        return ""  # No planning router needed
    
    router_code = '''
"""Generated FastAPI router for planning and reasoning endpoints."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..helpers import rate_limit_dependency, router_dependencies
from ..runtime import (
    execute_planner,
    execute_planning_workflow,
    execute_planner_step,
    get_planner_status,
    list_planners,
    list_planning_workflows,
)

router = APIRouter(tags=["planning"], dependencies=router_dependencies())


# Planning request/response models
class PlannerExecuteRequest(BaseModel):
    input_data: Dict[str, Any]
    session_id: Optional[str] = "default"
    timeout: Optional[float] = None
    

class PlannerExecuteResponse(BaseModel):
    success: bool
    status: str
    plan: List[Dict[str, Any]]
    final_state: Dict[str, Any]
    metrics: Dict[str, Any]
    execution_trace: List[Dict[str, Any]]
    error: Optional[str] = None


class PlannerStepRequest(BaseModel):
    context: Dict[str, Any]
    session_id: Optional[str] = "default"


class PlannerStepResponse(BaseModel):
    step_type: str
    content: Dict[str, Any]
    timestamp: float


class PlanningWorkflowRequest(BaseModel):
    input_data: Dict[str, Any] 
    session_id: Optional[str] = "default"
    timeout: Optional[float] = None


class PlanningWorkflowResponse(BaseModel):
    success: bool
    workflow_name: str
    stage_results: Dict[str, Any]
    final_state: Dict[str, Any]
    execution_trace: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    error: Optional[str] = None


# Planner endpoints
@router.get("/api/planners/")
async def list_available_planners() -> Dict[str, Any]:
    """List all available planners and their configurations."""
    return {
        "planners": list_planners(),
        "planning_workflows": list_planning_workflows(),
    }


@router.post(
    "/api/planners/{planner_name}/execute",
    response_model=PlannerExecuteResponse,
    dependencies=[rate_limit_dependency("ai")],
)
async def execute_planner_endpoint(
    planner_name: str, 
    request: PlannerExecuteRequest
) -> PlannerExecuteResponse:
    """Execute a planner with the given input data."""
    try:
        result = await execute_planner(
            planner_name=planner_name,
            input_data=request.input_data,
            session_id=request.session_id,
            timeout=request.timeout
        )
        
        return PlannerExecuteResponse(
            success=result.get("success", False),
            status=result.get("status", "unknown"),
            plan=result.get("plan", []),
            final_state=result.get("final_state", {}),
            metrics=result.get("metrics", {}),
            execution_trace=result.get("execution_trace", []),
            error=result.get("error")
        )
        
    except KeyError:
        raise HTTPException(
            status_code=404, 
            detail=f"Planner '{planner_name}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Planner execution failed: {str(e)}"
        )


@router.post(
    "/api/planners/{planner_name}/step", 
    response_model=PlannerStepResponse,
    dependencies=[rate_limit_dependency("ai")],
)
async def execute_planner_step_endpoint(
    planner_name: str,
    request: PlannerStepRequest
) -> PlannerStepResponse:
    """Execute a single step of a planner."""
    try:
        result = await execute_planner_step(
            planner_name=planner_name,
            context=request.context,
            session_id=request.session_id
        )
        
        return PlannerStepResponse(
            step_type=result.get("step_type", "unknown"),
            content=result.get("content", {}),
            timestamp=result.get("timestamp", 0.0)
        )
        
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Planner '{planner_name}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Planner step execution failed: {str(e)}"
        )


@router.get("/api/planners/{planner_name}/status")
async def get_planner_status_endpoint(planner_name: str) -> Dict[str, Any]:
    """Get the current status and configuration of a planner."""
    try:
        return get_planner_status(planner_name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Planner '{planner_name}' not found"
        )


# Planning workflow endpoints
@router.post(
    "/api/planning_workflows/{workflow_name}/execute",
    response_model=PlanningWorkflowResponse,
    dependencies=[rate_limit_dependency("ai")],
)
async def execute_planning_workflow_endpoint(
    workflow_name: str,
    request: PlanningWorkflowRequest
) -> PlanningWorkflowResponse:
    """Execute a multi-stage planning workflow."""
    try:
        result = await execute_planning_workflow(
            workflow_name=workflow_name,
            input_data=request.input_data,
            session_id=request.session_id,
            timeout=request.timeout
        )
        
        return PlanningWorkflowResponse(
            success=result.get("success", False),
            workflow_name=workflow_name,
            stage_results=result.get("stage_results", {}),
            final_state=result.get("final_state", {}),
            execution_trace=result.get("execution_trace", []),
            metrics=result.get("metrics", {}),
            error=result.get("error")
        )
        
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Planning workflow '{workflow_name}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Planning workflow execution failed: {str(e)}"
        )


# Planning utilities endpoints
@router.get("/api/planners/{planner_name}/schema")
async def get_planner_schema(planner_name: str) -> Dict[str, Any]:
    """Get the input/output schema for a planner."""
    try:
        status = get_planner_status(planner_name)
        return {
            "planner_name": planner_name,
            "planner_type": status.get("planner_type", "unknown"),
            "input_schema": status.get("input_schema", {}),
            "output_schema": status.get("output_schema", {}),
            "goal": status.get("goal", ""),
            "description": status.get("description", "")
        }
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Planner '{planner_name}' not found"
        )


__all__ = ["router"]
'''

    return textwrap.dedent(router_code).strip() + "\n"


__all__ = ["_render_planning_router_module"]