"""
Planning Integration with Chain Execution System

This module integrates the planning system with the existing chain execution 
infrastructure, enabling planners to be used as steps within workflows.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable
import logging

from ..runtime.planning_core import (
    PlanningEngine, PlanningContext, PlanningResult, PlanningStatus
)
from ..ir.spec import PlannerSpec, ChainSpec

logger = logging.getLogger(__name__)


class ChainPlannerExecutor:
    """
    Executes planners within chain workflows.
    
    Integrates with the existing ChainStep execution framework to allow
    planners to be invoked as "planner" kind steps alongside prompts, 
    tools, and other workflow primitives.
    """
    
    def __init__(self, planning_engine: PlanningEngine, chain_context: Dict[str, Any]):
        self.planning_engine = planning_engine
        self.chain_context = chain_context
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute_planner_step(self, step_config: Dict[str, Any], 
                                   workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a planner step within a chain workflow.
        
        Args:
            step_config: ChainStep configuration with kind="planner"
            workflow_state: Current workflow state and variables
            
        Returns:
            Dict containing planner results and updated state
        """
        try:
            planner_name = step_config.get("target")
            if not planner_name:
                raise ValueError("Planner step missing 'target' field")
            
            # Prepare planner input
            planner_input = self._prepare_planner_input(step_config, workflow_state)
            
            # Execute planner
            session_id = self.chain_context.get("session_id", "default")
            result = await self.planning_engine.execute_planner(
                planner_name=planner_name,
                input_data=planner_input,
                session_id=session_id
            )
            
            # Process results
            return self._process_planner_result(result, step_config, workflow_state)
            
        except Exception as e:
            self.logger.error(f"Planner step execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "step_type": "planner",
                "planner_name": step_config.get("target", "unknown"),
                "timestamp": time.time()
            }
    
    def _prepare_planner_input(self, step_config: Dict[str, Any], 
                              workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for planner execution"""
        
        # Get planner options from step config
        options = step_config.get("options", {})
        
        # Resolve variable references in options
        resolved_options = self._resolve_variables(options, workflow_state)
        
        # Include additional planning context if specified
        planning_context = step_config.get("planning_context", {})
        resolved_context = self._resolve_variables(planning_context, workflow_state)
        
        # Combine into planner input
        planner_input = {
            **resolved_options,
            **resolved_context,
            "_workflow_state": workflow_state,
            "_step_config": step_config
        }
        
        return planner_input
    
    def _resolve_variables(self, data: Any, workflow_state: Dict[str, Any]) -> Any:
        """
        Resolve variable references in data structure.
        
        Supports simple $variable syntax for referencing workflow state.
        """
        if isinstance(data, dict):
            return {key: self._resolve_variables(value, workflow_state) 
                   for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_variables(item, workflow_state) for item in data]
        elif isinstance(data, str) and data.startswith("$"):
            variable_name = data[1:]
            return workflow_state.get(variable_name, data)
        else:
            return data
    
    def _process_planner_result(self, result: PlanningResult, step_config: Dict[str, Any],
                               workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process planner result and update workflow state"""
        
        # Extract key information from planning result
        processed_result = {
            "success": result.status == PlanningStatus.COMPLETED,
            "status": result.status.value,
            "plan": result.plan,
            "final_state": result.final_state,
            "metrics": result.metrics,
            "execution_trace": result.execution_trace,
            "error": result.error,
            "step_type": "planner",
            "planner_name": step_config.get("target"),
            "planner_type": step_config.get("planner_type"),
            "timestamp": time.time()
        }
        
        # Apply output mapping if configured
        output_mapping = step_config.get("output_mapping", {})
        if output_mapping:
            for output_key, state_key in output_mapping.items():
                if output_key in result.final_state:
                    workflow_state[state_key] = result.final_state[output_key]
                elif output_key == "plan":
                    workflow_state[state_key] = result.plan
                elif output_key == "metrics":
                    workflow_state[state_key] = result.metrics
        
        # Store complete result if no specific mapping
        if not output_mapping:
            result_key = step_config.get("name", f"planner_result_{int(time.time())}")
            workflow_state[result_key] = processed_result
        
        return processed_result


class PlanningWorkflowOrchestrator:
    """
    Orchestrates complex planning workflows that combine multiple planners.
    
    Manages high-level planning workflows that chain different types of
    planners and coordinate their execution with proper dependency handling.
    """
    
    def __init__(self, planning_engine: PlanningEngine):
        self.planning_engine = planning_engine
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute_planning_workflow(self, workflow_spec: Dict[str, Any],
                                       input_data: Dict[str, Any],
                                       session_id: str = "default") -> Dict[str, Any]:
        """Execute a multi-stage planning workflow"""
        
        workflow_name = workflow_spec.get("name", "unnamed_planning_workflow")
        stages = workflow_spec.get("stages", [])
        stage_dependencies = workflow_spec.get("stage_dependencies", {})
        global_context = workflow_spec.get("global_context", {})
        
        self.logger.info(f"Executing planning workflow: {workflow_name}")
        
        # Initialize workflow state
        workflow_state = {
            **input_data,
            **global_context,
            "_workflow_name": workflow_name,
            "_session_id": session_id,
            "_start_time": time.time()
        }
        
        # Track stage execution
        completed_stages = set()
        stage_results = {}
        execution_trace = []
        
        try:
            # Execute stages in dependency order
            while len(completed_stages) < len(stages):
                progress_made = False
                
                for stage in stages:
                    stage_name = stage.get("name", stage.get("planner", "unnamed_stage"))
                    
                    if stage_name in completed_stages:
                        continue
                    
                    # Check dependencies
                    dependencies = stage_dependencies.get(stage_name, [])
                    if all(dep in completed_stages for dep in dependencies):
                        # Execute this stage
                        stage_result = await self._execute_planning_stage(
                            stage, workflow_state, session_id
                        )
                        
                        stage_results[stage_name] = stage_result
                        completed_stages.add(stage_name)
                        progress_made = True
                        
                        execution_trace.append({
                            "stage": stage_name,
                            "timestamp": time.time(),
                            "result": stage_result,
                            "dependencies": dependencies
                        })
                        
                        # Update workflow state with stage results
                        self._integrate_stage_result(stage_result, workflow_state, stage)
                
                if not progress_made:
                    raise Exception("Circular dependency detected in planning workflow stages")
            
            # Workflow completed successfully
            return {
                "success": True,
                "workflow_name": workflow_name,
                "stage_results": stage_results,
                "final_state": workflow_state,
                "execution_trace": execution_trace,
                "metrics": {
                    "stages_completed": len(completed_stages),
                    "total_execution_time": time.time() - workflow_state["_start_time"],
                    "success_rate": len(completed_stages) / len(stages)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Planning workflow '{workflow_name}' failed: {str(e)}")
            return {
                "success": False,
                "workflow_name": workflow_name,
                "error": str(e),
                "stage_results": stage_results,
                "final_state": workflow_state,
                "execution_trace": execution_trace,
                "metrics": {
                    "stages_completed": len(completed_stages),
                    "total_execution_time": time.time() - workflow_state["_start_time"],
                    "success_rate": len(completed_stages) / len(stages) if stages else 0
                }
            }
    
    async def _execute_planning_stage(self, stage: Dict[str, Any], 
                                     workflow_state: Dict[str, Any],
                                     session_id: str) -> Dict[str, Any]:
        """Execute a single planning stage"""
        
        planner_name = stage.get("planner")
        goal = stage.get("goal", "Complete planning stage")
        
        if not planner_name:
            raise ValueError(f"Planning stage missing 'planner' field: {stage}")
        
        # Prepare stage input
        stage_input = {
            "goal": goal,
            **workflow_state,
            "_stage_config": stage
        }
        
        # Execute planner for this stage
        result = await self.planning_engine.execute_planner(
            planner_name=planner_name,
            input_data=stage_input,
            session_id=session_id
        )
        
        return {
            "planner_name": planner_name,
            "goal": goal,
            "status": result.status.value,
            "success": result.status == PlanningStatus.COMPLETED,
            "plan": result.plan,
            "final_state": result.final_state,
            "metrics": result.metrics,
            "error": result.error
        }
    
    def _integrate_stage_result(self, stage_result: Dict[str, Any], 
                               workflow_state: Dict[str, Any],
                               stage_config: Dict[str, Any]):
        """Integrate stage result into workflow state"""
        
        # Apply output mapping if configured
        output_mapping = stage_config.get("output_mapping", {})
        if output_mapping and stage_result.get("final_state"):
            for output_key, state_key in output_mapping.items():
                if output_key in stage_result["final_state"]:
                    workflow_state[state_key] = stage_result["final_state"][output_key]
        
        # Store stage result with stage name prefix
        stage_name = stage_config.get("name", stage_config.get("planner", "stage"))
        workflow_state[f"{stage_name}_result"] = stage_result
        
        # Merge final state if no conflicts
        if stage_result.get("final_state"):
            for key, value in stage_result["final_state"].items():
                if not key.startswith("_") and key not in workflow_state:
                    workflow_state[key] = value


class PlannerChainAdapter:
    """
    Adapter that makes planners compatible with existing chain infrastructure.
    
    Provides a unified interface for invoking planners within the existing
    chain execution system without requiring changes to the core chain runtime.
    """
    
    def __init__(self, planning_engine: PlanningEngine):
        self.planning_engine = planning_engine
        self.chain_executor = None
        self.workflow_orchestrator = PlanningWorkflowOrchestrator(planning_engine)
    
    def register_with_chain_runtime(self, chain_runtime):
        """Register planner adapter with the chain runtime system"""
        self.chain_executor = ChainPlannerExecutor(
            self.planning_engine, 
            chain_runtime.context
        )
        
        # Register planner step handler
        chain_runtime.register_step_handler("planner", self._handle_planner_step)
        
        # Register planning workflow handler  
        chain_runtime.register_step_handler("planning_workflow", self._handle_planning_workflow_step)
    
    async def _handle_planner_step(self, step_config: Dict[str, Any],
                                  workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle planner step execution within chain"""
        return await self.chain_executor.execute_planner_step(step_config, workflow_state)
    
    async def _handle_planning_workflow_step(self, step_config: Dict[str, Any],
                                           workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle planning workflow execution within chain"""
        workflow_spec = step_config.get("workflow_spec", {})
        session_id = workflow_state.get("_session_id", "default")
        
        result = await self.workflow_orchestrator.execute_planning_workflow(
            workflow_spec=workflow_spec,
            input_data=workflow_state,
            session_id=session_id
        )
        
        return result
    
    def create_planner_chain_step(self, planner_name: str, planner_type: str,
                                 options: Dict[str, Any] = None,
                                 output_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Create a chain step configuration for a planner"""
        return {
            "kind": "planner",
            "target": planner_name,
            "planner_type": planner_type,
            "options": options or {},
            "output_mapping": output_mapping or {},
            "stop_on_error": True
        }
    
    def create_planning_workflow_step(self, workflow_name: str,
                                     stages: List[Dict[str, Any]],
                                     stage_dependencies: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Create a chain step configuration for a planning workflow"""
        return {
            "kind": "planning_workflow",
            "target": workflow_name,
            "workflow_spec": {
                "name": workflow_name,
                "stages": stages,
                "stage_dependencies": stage_dependencies or {}
            },
            "stop_on_error": True
        }


# =============================================================================
# Planning Security Integration
# =============================================================================

class PlanningSecurityManager:
    """
    Manages security policies for planning operations.
    
    Integrates with existing security framework to ensure planners
    respect permission levels, capability requirements, and resource limits.
    """
    
    def __init__(self, security_context: Dict[str, Any]):
        self.security_context = security_context
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_planner_execution(self, planner_spec: PlannerSpec,
                                  user_context: Dict[str, Any]) -> bool:
        """Validate that user can execute the planner"""
        
        # Check permission level
        required_permission = planner_spec.permission_level
        user_permission = user_context.get("permission_level")
        
        if required_permission and not self._check_permission_level(user_permission, required_permission):
            self.logger.warning(f"Insufficient permission for planner {planner_spec.name}")
            return False
        
        # Check required capabilities
        required_capabilities = planner_spec.capabilities
        user_capabilities = user_context.get("capabilities", [])
        
        if not all(cap in user_capabilities for cap in required_capabilities):
            self.logger.warning(f"Missing capabilities for planner {planner_spec.name}")
            return False
        
        # Check tool access
        allowed_tools = planner_spec.allowed_tools
        if allowed_tools:
            available_tools = user_context.get("available_tools", [])
            if not all(tool in available_tools for tool in allowed_tools):
                self.logger.warning(f"Missing tool access for planner {planner_spec.name}")
                return False
        
        return True
    
    def _check_permission_level(self, user_level: str, required_level: str) -> bool:
        """Check if user permission level meets requirement"""
        # Simplified permission hierarchy
        levels = ["guest", "user", "admin", "system"]
        
        try:
            user_idx = levels.index(user_level) if user_level in levels else -1
            required_idx = levels.index(required_level) if required_level in levels else len(levels)
            return user_idx >= required_idx
        except ValueError:
            return False
    
    def apply_resource_limits(self, planning_context: PlanningContext,
                             planner_spec: PlannerSpec):
        """Apply resource limits to planning execution"""
        
        # Set timeout based on spec and security policy
        max_timeout = self.security_context.get("max_planning_timeout", 300)  # 5 minutes
        spec_timeout = planner_spec.timeout_seconds or 60
        
        planning_context.metadata["timeout"] = min(spec_timeout, max_timeout)
        
        # Set memory limits
        max_memory = self.security_context.get("max_planning_memory", 100 * 1024 * 1024)  # 100MB
        spec_memory = planner_spec.max_memory_usage or max_memory
        
        planning_context.metadata["memory_limit"] = min(spec_memory, max_memory)


__all__ = [
    "ChainPlannerExecutor",
    "PlanningWorkflowOrchestrator", 
    "PlannerChainAdapter",
    "PlanningSecurityManager"
]