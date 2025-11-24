"""AI resource encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from namel3ss.plugins.utils import normalize_plugin_category

from .expressions import _collect_template_markers, _encode_value, _expression_to_runtime, _expression_to_source
from .utils import _validate_chain_memory_options

if TYPE_CHECKING:
    from ....ast import (
        AIModel,
        Chain,
        ChainStep,
        Connector,
        Memory,
        OutputField,
        OutputFieldType,
        OutputSchema,
        Prompt,
        PromptField,
        Template,
        WorkflowForBlock,
        WorkflowIfBlock,
        WorkflowNode,
        WorkflowWhileBlock,
    )
    from ....ast.agents import LLMDefinition, ToolDefinition
    from ....ast.rag import IndexDefinition, RagPipelineDefinition


def _encode_ai_connector(connector: "Connector", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an AI connector definition."""
    config_encoded = _encode_value(connector.config, env_keys)
    if isinstance(config_encoded, dict):
        config_payload = config_encoded
    else:
        config_payload = connector.config
    encoded: Dict[str, Any] = {
        "name": connector.name,
        "type": connector.connector_type,
        "category": normalize_plugin_category(connector.connector_type),
        "config": config_payload,
    }
    if connector.provider:
        encoded["provider"] = connector.provider
    if connector.description:
        encoded["description"] = connector.description
    return encoded


def _encode_template(template: "Template", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a template definition."""
    metadata_encoded = _encode_value(template.metadata, env_keys)
    if isinstance(metadata_encoded, dict):
        metadata_payload = metadata_encoded
    else:
        metadata_payload = template.metadata
    return {
        "name": template.name,
        "prompt": template.prompt,
        "metadata": metadata_payload,
    }


def _encode_memory(memory: "Memory", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a memory definition."""
    config_encoded = _encode_value(memory.config, env_keys)
    if not isinstance(config_encoded, dict):
        config_encoded = {"value": config_encoded} if config_encoded is not None else {}
    metadata_encoded = _encode_value(memory.metadata, env_keys)
    if not isinstance(metadata_encoded, dict):
        metadata_encoded = {"value": metadata_encoded} if metadata_encoded is not None else {}
    payload: Dict[str, Any] = {
        "name": memory.name,
        "scope": memory.scope,
        "kind": memory.kind,
        "config": config_encoded,
        "metadata": metadata_encoded,
    }
    if memory.max_items is not None:
        payload["max_items"] = int(memory.max_items)
    return payload


def _encode_ai_model(model: "AIModel", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an AI model definition."""
    config_payload = _encode_value(model.config, env_keys)
    metadata_value = _encode_value(model.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    return {
        "name": model.name,
        "provider": model.provider,
        "model": model.model_name,
        "config": config_payload if isinstance(config_payload, dict) else model.config,
        "description": model.description,
        "metadata": metadata_value,
    }


def _encode_prompt(prompt: "Prompt", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a prompt definition."""
    _collect_template_markers(prompt.template, env_keys)
    parameters_value = _encode_value(prompt.parameters, env_keys)
    if not isinstance(parameters_value, dict):
        parameters_value = {"value": parameters_value} if parameters_value is not None else {}
    metadata_value = _encode_value(prompt.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    
    # Encode structured prompt args
    args_list = []
    if prompt.args:
        for arg in prompt.args:
            arg_dict = {
                "name": arg.name,
                "type": arg.arg_type,
                "required": arg.required,
            }
            if arg.default is not None:
                arg_dict["default"] = _encode_value(arg.default, env_keys)
            if arg.description:
                arg_dict["description"] = arg.description
            args_list.append(arg_dict)
    
    # Encode structured output schema
    output_schema_dict = None
    if prompt.output_schema:
        output_schema_dict = _encode_output_schema(prompt.output_schema, env_keys)
    
    result = {
        "name": prompt.name,
        "model": prompt.model,
        "template": prompt.template,
        "input": [_encode_prompt_field(field, env_keys) for field in prompt.input_fields],
        "output": [_encode_prompt_field(field, env_keys) for field in prompt.output_fields],
        "parameters": parameters_value,
        "metadata": metadata_value,
        "description": prompt.description,
    }
    
    # Add structured prompt fields if present
    if args_list:
        result["args"] = args_list
    if output_schema_dict:
        result["output_schema"] = output_schema_dict
    
    return result


def _encode_prompt_field(field: "PromptField", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a prompt field definition."""
    # Accept both PromptField instances and simple dicts
    if isinstance(field, dict):
        metadata_value = _encode_value(field.get("metadata", {}), env_keys)
        if not isinstance(metadata_value, dict):
            metadata_value = {"value": metadata_value} if metadata_value is not None else {}
        payload: Dict[str, Any] = {
            "name": field.get("name") or field.get("field") or "field",
            "type": field.get("type", "string"),
            "required": field.get("required", True),
            "enum": list(field.get("enum", []) or []),
            "description": field.get("description"),
            "metadata": metadata_value,
        }
        if "default" in field:
            payload["default"] = _encode_value(field.get("default"), env_keys)
        return payload
    
    metadata_value = _encode_value(getattr(field, "metadata", {}), env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    payload2: Dict[str, Any] = {
        "name": getattr(field, "name", "field"),
        "type": getattr(field, "field_type", getattr(field, "type", "string")),
        "required": getattr(field, "required", True),
        "enum": list(getattr(field, "enum", []) or []),
        "description": getattr(field, "description", None),
        "metadata": metadata_value,
    }
    if getattr(field, "default", None) is not None:
        payload2["default"] = _encode_value(getattr(field, "default"), env_keys)
    return payload2


def _encode_output_field_type(field_type: "OutputFieldType") -> Dict[str, Any]:
    """Encode an OutputFieldType to a dictionary for runtime."""
    result = {
        "base_type": field_type.base_type,
        "nullable": field_type.nullable,
    }
    
    if field_type.enum_values:
        result["enum_values"] = field_type.enum_values
    
    if field_type.element_type:
        result["element_type"] = _encode_output_field_type(field_type.element_type)
    
    if field_type.nested_fields:
        result["nested_fields"] = [
            {
                "name": field.name,
                "field_type": _encode_output_field_type(field.field_type),
                "required": field.required,
                "description": field.description,
            }
            for field in field_type.nested_fields
        ]
    
    return result


def _encode_output_schema(schema: "OutputSchema", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an OutputSchema to a dictionary for runtime."""
    return {
        "fields": [
            {
                "name": field.name,
                "field_type": _encode_output_field_type(field.field_type),
                "required": field.required,
                "description": field.description,
            }
            for field in schema.fields
        ]
    }


def _encode_llm(llm: "LLMDefinition", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a first-class LLM definition for the backend runtime."""
    config_value = _encode_value(llm.config, env_keys)
    if not isinstance(config_value, dict):
        config_value = {"value": config_value} if config_value is not None else {}
    metadata_value = _encode_value(llm.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    payload: Dict[str, Any] = {
        "name": llm.name,
        "provider": llm.provider,
        "model": llm.model,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "config": config_value,
        "metadata": metadata_value,
    }
    if llm.top_p is not None:
        payload["top_p"] = llm.top_p
    if llm.frequency_penalty is not None:
        payload["frequency_penalty"] = llm.frequency_penalty
    if llm.presence_penalty is not None:
        payload["presence_penalty"] = llm.presence_penalty
    return payload


def _encode_tool(tool: "ToolDefinition", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a first-class Tool definition for the backend runtime."""
    # Encode parameters as input_schema
    input_schema_value = _encode_value(getattr(tool, "parameters", {}), env_keys)
    if not isinstance(input_schema_value, dict):
        input_schema_value = {"value": input_schema_value} if input_schema_value is not None else {}
    
    # Encode returns as output_schema
    output_schema_value = _encode_value(getattr(tool, "returns", {}), env_keys)
    if not isinstance(output_schema_value, dict):
        output_schema_value = {"value": output_schema_value} if output_schema_value is not None else {}
    
    metadata_value = _encode_value(tool.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    
    # Extract security metadata
    required_capabilities = list(getattr(tool, "required_capabilities", []) or [])
    permission_level = getattr(tool, "permission_level", None)
    timeout_seconds = getattr(tool, "timeout_seconds", None)
    rate_limit_per_minute = getattr(tool, "rate_limit_per_minute", None)
    security_config = getattr(tool, "security_config", None)
    
    payload: Dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": input_schema_value,
        "output_schema": output_schema_value,
        "implementation": getattr(tool, "implementation", {}),
        "examples": getattr(tool, "examples", []),
        # Security metadata
        "required_capabilities": required_capabilities,
        "permission_level": permission_level,
        "timeout_seconds": timeout_seconds,
        "rate_limit_per_minute": rate_limit_per_minute,
        "security_config": security_config,
        "metadata": metadata_value,
    }
    
    return payload


def _encode_chain(chain: "Chain", env_keys: Set[str], memory_names: Set[str]) -> Dict[str, Any]:
    """Encode a chain definition."""
    encoded_steps = [
        _encode_workflow_node(node, env_keys, memory_names, chain.name) for node in chain.steps
    ]
    metadata_encoded = _encode_value(chain.metadata, env_keys)
    if not isinstance(metadata_encoded, dict):
        metadata_encoded = {"value": metadata_encoded}
    return {
        "name": chain.name,
        "input_key": chain.input_key,
        "steps": encoded_steps,
        "metadata": metadata_encoded,
    }


def _encode_workflow_node(
    node: "WorkflowNode",
    env_keys: Set[str],
    memory_names: Set[str],
    chain_name: str,
) -> Dict[str, Any]:
    """Encode a workflow node (step, if, for, while)."""
    from ....ast import ChainStep, WorkflowForBlock, WorkflowIfBlock, WorkflowWhileBlock
    
    if isinstance(node, ChainStep):
        return _encode_chain_step(node, env_keys, memory_names, chain_name)
    if isinstance(node, WorkflowIfBlock):
        payload: Dict[str, Any] = {
            "type": "if",
            "condition": _expression_to_runtime(node.condition),
            "condition_source": _expression_to_source(node.condition),
            "then": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.then_steps],
            "elif": [
                {
                    "condition": _expression_to_runtime(branch_condition),
                    "condition_source": _expression_to_source(branch_condition),
                    "steps": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in branch_steps],
                }
                for branch_condition, branch_steps in node.elif_steps
            ],
            "else": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.else_steps],
        }
        return payload
    if isinstance(node, WorkflowForBlock):
        payload: Dict[str, Any] = {
            "type": "for",
            "loop_var": node.loop_var,
            "source_kind": node.source_kind,
            "body": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.body],
        }
        if node.source_name:
            payload["source_name"] = node.source_name
        if node.source_expression is not None:
            payload["source_expression"] = _expression_to_runtime(node.source_expression)
            payload["source_expression_source"] = _expression_to_source(node.source_expression)
        if node.max_iterations:
            payload["max_iterations"] = int(node.max_iterations)
        return payload
    if isinstance(node, WorkflowWhileBlock):
        payload = {
            "type": "while",
            "condition": _expression_to_runtime(node.condition),
            "condition_source": _expression_to_source(node.condition),
            "body": [_encode_workflow_node(child, env_keys, memory_names, chain_name) for child in node.body],
        }
        if node.max_iterations:
            payload["max_iterations"] = int(node.max_iterations)
        return payload
    raise TypeError(f"Unsupported workflow node '{type(node).__name__}' in chain '{chain_name}'")


def _encode_chain_step(
    step: "ChainStep",
    env_keys: Set[str],
    memory_names: Set[str],
    chain_name: str,
) -> Dict[str, Any]:
    """Encode a chain step."""
    options_encoded = _encode_value(step.options, env_keys)
    if not isinstance(options_encoded, dict):
        options_encoded = {"value": options_encoded}
    _validate_chain_memory_options(step.options, memory_names, chain_name, step.kind, step.target)
    payload: Dict[str, Any] = {
        "type": "step",
        "kind": step.kind,
        "target": step.target,
        "options": options_encoded,
        "stop_on_error": bool(step.stop_on_error),
    }
    if step.name:
        payload["name"] = step.name
    if step.evaluation:
        evaluation_payload: Dict[str, Any] = {
            "evaluators": list(step.evaluation.evaluators),
        }
        if step.evaluation.guardrail:
            evaluation_payload["guardrail"] = step.evaluation.guardrail
        payload["evaluation"] = evaluation_payload
    
    # Planning-specific fields
    if hasattr(step, "planner_type") and step.planner_type:
        payload["planner_type"] = step.planner_type
    if hasattr(step, "planning_context") and step.planning_context:
        planning_context_encoded = _encode_value(step.planning_context, env_keys)
        if isinstance(planning_context_encoded, dict):
            payload["planning_context"] = planning_context_encoded
        
    return payload


def _encode_planner(planner: Any, env_keys: Set[str]) -> Dict[str, Any]:
    """
    Encode a planner definition for backend generation.
    
    Handles ReAct, Chain-of-Thought, and Graph-based planners.
    """
    from ....ast.planning import ReActPlanner, ChainOfThoughtPlanner, GraphBasedPlanner
    
    base_payload: Dict[str, Any] = {
        "name": planner.name,
        "metadata": _encode_value(planner.metadata, env_keys) if hasattr(planner, "metadata") else {}
    }
    
    if isinstance(planner, ReActPlanner):
        base_payload.update({
            "planner_type": "react",
            "goal": planner.goal,
            "max_cycles": planner.max_cycles,
            "reasoning_prompt": planner.reasoning_prompt,
            "action_tools": list(planner.action_tools),
            "fallback_action": planner.fallback_action,
            "initial_context": _encode_value(planner.initial_context, env_keys),
        })
        
        if planner.success_condition:
            # Encode success condition expression
            base_payload["success_condition"] = _expression_to_runtime(planner.success_condition)
            base_payload["success_condition_source"] = _expression_to_source(planner.success_condition)
    
    elif isinstance(planner, ChainOfThoughtPlanner):
        base_payload.update({
            "planner_type": "chain_of_thought",
            "problem": planner.problem,
            "reasoning_steps": list(planner.reasoning_steps),
            "step_prompts": dict(planner.step_prompts),
            "step_tools": {k: list(v) for k, v in planner.step_tools.items()},
            "dependencies": {k: list(v) for k, v in planner.dependencies.items()},
            "intermediate_validation": bool(planner.intermediate_validation),
        })
    
    elif isinstance(planner, GraphBasedPlanner):
        base_payload.update({
            "planner_type": "graph_based", 
            "initial_state": _encode_value(planner.initial_state, env_keys),
            "goal_state": _encode_value(planner.goal_state, env_keys),
            "state_transitions": list(planner.state_transitions),
            "heuristic_function": planner.heuristic_function,
            "max_search_time": planner.max_search_time,
        })
        
        # Encode search policy
        if planner.search_policy:
            search_policy_payload = {
                "policy_type": planner.search_policy.policy_type,
                "parameters": dict(planner.search_policy.parameters)
            }
            base_payload["search_policy"] = search_policy_payload
    
    else:
        # Generic planner fallback
        base_payload.update({
            "planner_type": "generic",
            "goal": getattr(planner, "goal", "Complete planning task"),
        })
    
    return base_payload


def _encode_planning_workflow(workflow: Any, env_keys: Set[str]) -> Dict[str, Any]:
    """
    Encode a planning workflow definition.
    
    Handles multi-stage planning workflows that coordinate multiple planners.
    """
    from ....ast.planning import PlanningWorkflow
    
    if not isinstance(workflow, PlanningWorkflow):
        # Generic workflow fallback
        return {
            "name": getattr(workflow, "name", "unnamed_workflow"),
            "workflow_type": "planning",
            "stages": [],
            "metadata": _encode_value(getattr(workflow, "metadata", {}), env_keys)
        }
    
    return {
        "name": workflow.name,
        "workflow_type": "planning",
        "input_schema": workflow.input_schema,
        "output_schema": workflow.output_schema,
        "stages": list(workflow.stages),
        "stage_dependencies": {k: list(v) for k, v in workflow.stage_dependencies.items()},
        "global_context": _encode_value(workflow.global_context, env_keys),
        "error_handling": _encode_value(workflow.error_handling, env_keys),
        "metadata": _encode_value(workflow.metadata, env_keys)
    }


def _encode_index(index: "IndexDefinition", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a RAG index definition for the backend runtime."""
    config_value = _encode_value(index.config, env_keys)
    if not isinstance(config_value, dict):
        config_value = {"value": config_value} if config_value is not None else {}
    metadata_value = _encode_value(index.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    
    payload: Dict[str, Any] = {
        "name": index.name,
        "source_dataset": index.source_dataset,
        "embedding_model": index.embedding_model,
        "chunk_size": index.chunk_size,
        "overlap": index.overlap,
        "backend": index.backend,
        "config": config_value,
        "metadata": metadata_value,
    }
    
    if index.namespace is not None:
        payload["namespace"] = index.namespace
    if index.collection is not None:
        payload["collection"] = index.collection
    if index.table_name is not None:
        payload["table_name"] = index.table_name
    if index.metadata_fields is not None:
        payload["metadata_fields"] = index.metadata_fields
    
    # Multimodal fields
    if hasattr(index, 'extract_images') and index.extract_images:
        payload["extract_images"] = index.extract_images
    if hasattr(index, 'extract_audio') and index.extract_audio:
        payload["extract_audio"] = index.extract_audio
    if hasattr(index, 'image_model') and index.image_model:
        payload["image_model"] = index.image_model
    if hasattr(index, 'audio_model') and index.audio_model:
        payload["audio_model"] = index.audio_model
    
    return payload


def _encode_rag_pipeline(pipeline: "RagPipelineDefinition", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a RAG pipeline definition for the backend runtime."""
    config_value = _encode_value(pipeline.config, env_keys)
    if not isinstance(config_value, dict):
        config_value = {"value": config_value} if config_value is not None else {}
    metadata_value = _encode_value(pipeline.metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    filters_value = _encode_value(pipeline.filters, env_keys)
    if not isinstance(filters_value, dict):
        filters_value = {"value": filters_value} if filters_value is not None else {}
    
    payload: Dict[str, Any] = {
        "name": pipeline.name,
        "query_encoder": pipeline.query_encoder,
        "index": pipeline.index,
        "top_k": pipeline.top_k,
        "distance_metric": pipeline.distance_metric,
        "config": config_value,
        "metadata": metadata_value,
    }
    
    if pipeline.reranker is not None:
        payload["reranker"] = pipeline.reranker
    if pipeline.filters is not None:
        payload["filters"] = filters_value
    
    # Hybrid search fields
    if hasattr(pipeline, 'enable_hybrid') and pipeline.enable_hybrid:
        payload["enable_hybrid"] = pipeline.enable_hybrid
    if hasattr(pipeline, 'sparse_model') and pipeline.sparse_model:
        payload["sparse_model"] = pipeline.sparse_model
    if hasattr(pipeline, 'dense_weight') and pipeline.dense_weight is not None:
        payload["dense_weight"] = pipeline.dense_weight
    if hasattr(pipeline, 'sparse_weight') and pipeline.sparse_weight is not None:
        payload["sparse_weight"] = pipeline.sparse_weight
    if hasattr(pipeline, 'reranker_type') and pipeline.reranker_type:
        payload["reranker_type"] = pipeline.reranker_type
    
    return payload
