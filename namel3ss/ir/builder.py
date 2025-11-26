"""
IR builder - converts AST to runtime-agnostic intermediate representation.

This module transforms parsed Namel3ss AST into IR specifications that
runtime adapters can consume. It extracts semantic information without
making assumptions about the target runtime.

PHASE 1 IMPLEMENTATION:
-----------------------
In Phase 1, we introduce the IR layer while keeping existing behavior.
The builder uses the existing BackendState as a bridge, extracting data
from it to build the IR. Future phases will bypass BackendState entirely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from namel3ss.ast import App
from namel3ss.security.config import get_security_config

from .spec import (
    BackendIR,
    FrontendIR,
    EndpointIR,
    AgentSpec,
    PromptSpec,
    ToolSpec,
    DatasetSpec,
    FrameSpec,
    MemorySpec,
    ChainSpec,
    InsightSpec,
    PageSpec,
    ComponentSpec,
    RouteSpec,
    TypeSpec,
    SchemaField,
    HTTPMethod,
    MemoryScope,
    CacheStrategy,
    DataBindingSpec,
    UpdateChannelSpec,
    # Data display component IR specs
    IRDataTable,
    IRColumnConfig,
    IRToolbarConfig,
    IRDataList,
    IRListItemConfig,
    IRStatSummary,
    IRSparklineConfig,
    IRTimeline,
    IRTimelineItem,
    IRAvatarGroup,
    IRAvatarItem,
    IRDataChart,
    IRChartConfig,
)


def build_backend_ir(app: App) -> BackendIR:
    """
    Build backend IR from parsed application AST.
    
    This is the main entry point for converting language AST to
    runtime-agnostic IR.
    
    Phase 1: Uses existing BackendState as bridge (no behavior change)
    Future: Will extract directly from AST
    
    Args:
        app: Parsed application AST
        
    Returns:
        BackendIR: Complete backend specification
        
    Example:
        >>> from namel3ss import Parser
        >>> parser = Parser(source_code)
        >>> module = parser.parse()
        >>> app = module.body[0]
        >>> ir = build_backend_ir(app)
    """
    # Phase 1: Use existing BackendState as bridge
    from namel3ss.codegen.backend.state import build_backend_state
    
    state = build_backend_state(app)
    
    # Convert BackendState to IR
    ir = BackendIR(
        app_name=app.name,
        app_version="1.0.0",
        ir_version="0.1.0",
    )
    
    # Extract prompts from state
    ir.prompts = _extract_prompts_from_state(state)
    
    # Extract agents from state
    ir.agents = _extract_agents_from_state(state)
    
    # Extract tools from state
    ir.tools = _extract_tools_from_state(state)
    
    # Extract datasets from state
    ir.datasets = _extract_datasets_from_state(state)
    
    # Extract frames from state
    ir.frames = _extract_frames_from_state(state)
    
    # Extract memory from state
    ir.memory = _extract_memory_from_state(state)
    
    # Extract chains from state
    ir.chains = _extract_chains_from_state(state)
    
    # Extract insights from state
    ir.insights = _extract_insights_from_state(state)
    
    # Extract endpoints from state
    ir.endpoints = _extract_endpoints_from_state(state)
    
    # Extract update channels for realtime datasets
    ir.update_channels = _extract_update_channels(state, ir.datasets)
    
    # Extract configuration
    if state.connectors:
        ir.database_config = {
            "connectors": state.connectors
        }
    
    # Extract security configuration from global config
    try:
        security_config = get_security_config()
        ir.security_config = {
            "default_environment": security_config.default_environment,
            "audit_log_enabled": security_config.audit_log_path is not None,
            "fail_mode": security_config.fail_mode,
        }
    except Exception:
        # Security config not available, use defaults
        ir.security_config = None
    
    # Build security mappings from agents and tools
    ir.agent_tool_mappings = {}
    ir.capability_requirements = {}
    ir.permission_levels = {}
    
    for agent in ir.agents:
        if agent.allowed_tools:
            ir.agent_tool_mappings[agent.name] = agent.allowed_tools
        if agent.permission_level:
            ir.permission_levels[agent.name] = agent.permission_level
    
    for tool in ir.tools:
        if tool.required_capabilities:
            ir.capability_requirements[tool.name] = tool.required_capabilities
        if tool.permission_level:
            ir.permission_levels[tool.name] = tool.permission_level
    
    # Store additional metadata
    ir.metadata = {
        "ai_models": state.ai_models,
        "ai_connectors": state.ai_connectors,
        "experiments": state.experiments,
        "training_jobs": state.training_jobs,
        "tuning_jobs": state.tuning_jobs,
        "crud_resources": state.crud_resources,
        "evaluators": state.evaluators,
        "metrics": state.metrics,
        "guardrails": state.guardrails,
        "eval_suites": state.eval_suites,
        "queries": state.queries,
        "knowledge_modules": state.knowledge_modules,
        # PHASE 2: Store original app for runtime adapters
        # This is temporary until runtimes consume IR directly
        "_original_app": app,
    }
    
    return ir


def build_frontend_ir(app: App) -> FrontendIR:
    """
    Build frontend IR from parsed application AST.
    
    Args:
        app: Parsed application AST
        
    Returns:
        FrontendIR: Complete frontend specification
    """
    from namel3ss.codegen.backend.state import build_backend_state
    
    state = build_backend_state(app)
    
    ir = FrontendIR(
        app_name=app.name,
        app_version="1.0.0",
        ir_version="0.1.0",
    )
    
    # Extract pages from state
    ir.pages = _extract_pages_from_state(state)
    
    # Extract routes from state
    ir.routes = _extract_routes_from_state(state)
    
    # PHASE 2: Store original app for runtime adapters
    ir.metadata["_original_app"] = app
    
    return ir


# =============================================================================
# Internal conversion functions (Phase 1: Extract from BackendState)
# =============================================================================

def _extract_prompts_from_state(state) -> List[PromptSpec]:
    """Extract prompts from BackendState"""
    prompts = []
    for name, prompt_data in state.prompts.items():
        prompts.append(PromptSpec(
            name=name,
            input_schema=_dict_to_type_spec(prompt_data.get("input_schema", {})),
            output_schema=_dict_to_type_spec(prompt_data.get("output_schema", {})),
            template=prompt_data.get("template", ""),
            model=prompt_data.get("model", ""),
            temperature=prompt_data.get("temperature", 0.7),
            max_tokens=prompt_data.get("max_tokens"),
            system_message=prompt_data.get("system", ""),
            examples=prompt_data.get("examples", []),
            memory_refs=prompt_data.get("memory_refs", []),
            metadata=prompt_data,
        ))
    return prompts


def _extract_agents_from_state(state) -> List[AgentSpec]:
    """Extract agents from BackendState"""
    agents = []
    for name, agent_data in state.agents.items():
        # Extract security metadata
        allowed_tools = agent_data.get("tools", [])
        capabilities = agent_data.get("capabilities", [])
        permission_level = agent_data.get("permission_level")
        
        # Extract security policy if present
        security_policy = None
        security_config = agent_data.get("security_config")
        if security_config:
            security_policy = {
                "max_tokens_per_request": getattr(security_config, "max_tokens_per_request", None),
                "rate_limit_per_minute": getattr(security_config, "rate_limit_per_minute", None),
                "timeout_seconds": getattr(security_config, "timeout_seconds", None),
            }
        
        agents.append(AgentSpec(
            name=name,
            nodes=agent_data.get("nodes", []),
            edges=agent_data.get("edges", []),
            entry_point=agent_data.get("entry_point", ""),
            handoff_logic=agent_data.get("handoff_logic", {}),
            state_schema=_dict_to_type_spec(agent_data.get("state_schema", {})),
            # Security metadata
            allowed_tools=allowed_tools,
            capabilities=capabilities,
            permission_level=permission_level,
            security_policy=security_policy,
            metadata=agent_data,
        ))
    return agents


def _extract_tools_from_state(state) -> List[ToolSpec]:
    """Extract tools from BackendState"""
    tools = []
    for name, tool_data in state.tools.items():
        # Extract security metadata
        required_capabilities = tool_data.get("required_capabilities", [])
        permission_level = tool_data.get("permission_level")
        timeout_seconds = tool_data.get("timeout_seconds")
        rate_limit_per_minute = tool_data.get("rate_limit_per_minute")
        
        tools.append(ToolSpec(
            name=name,
            description=tool_data.get("description", ""),
            input_schema=_dict_to_type_spec(tool_data.get("input_schema", {})),
            output_schema=_dict_to_type_spec(tool_data.get("output_schema", {})),
            implementation_type=tool_data.get("type", "python"),
            implementation_ref=tool_data.get("implementation", ""),
            # Security metadata
            required_capabilities=required_capabilities,
            permission_level=permission_level,
            timeout_seconds=timeout_seconds,
            rate_limit_per_minute=rate_limit_per_minute,
            metadata=tool_data,
        ))
    return tools


def _extract_datasets_from_state(state) -> List[DatasetSpec]:
    """Extract datasets from BackendState with binding metadata"""
    datasets = []
    
    # Access original app if available to get dataset AST nodes
    original_app = state.__dict__.get("_original_app")
    dataset_ast_map = {}
    if original_app:
        # Build map of dataset name -> AST node
        for item in getattr(original_app, "body", []):
            if hasattr(item, "__class__") and item.__class__.__name__ == "Dataset":
                dataset_ast_map[item.name] = item
    
    for name, dataset_data in state.datasets.items():
        # Extract access policy from AST if available
        access_policy = None
        primary_key = None
        realtime_enabled = False
        
        if name in dataset_ast_map:
            dataset_ast = dataset_ast_map[name]
            if hasattr(dataset_ast, "access_policy") and dataset_ast.access_policy:
                ap = dataset_ast.access_policy
                access_policy = {
                    "read_only": ap.read_only,
                    "allow_create": ap.allow_create,
                    "allow_update": ap.allow_update,
                    "allow_delete": ap.allow_delete,
                    "required_capabilities": ap.required_capabilities,
                }
                primary_key = ap.primary_key
            
            # Check if dataset should support realtime
            if hasattr(dataset_ast, "reactive") and dataset_ast.reactive:
                realtime_enabled = True
        
        datasets.append(DatasetSpec(
            name=name,
            source_type=dataset_data.get("source_type", "sql"),
            source_config=dataset_data.get("source_config", {}),
            schema=[],  # TODO: Extract schema fields
            transformations=dataset_data.get("transformations", []),
            cache_policy=dataset_data.get("cache_policy"),
            refresh_policy=dataset_data.get("refresh_policy"),
            access_policy=access_policy,
            primary_key=primary_key,
            realtime_enabled=realtime_enabled,
            metadata=dataset_data,
        ))
    return datasets


def _extract_frames_from_state(state) -> List[FrameSpec]:
    """Extract frames from BackendState"""
    frames = []
    for name, frame_data in state.frames.items():
        frames.append(FrameSpec(
            name=name,
            columns=[],  # TODO: Extract column fields
            source_dataset=frame_data.get("source_dataset"),
            constraints=frame_data.get("constraints", []),
            indexes=frame_data.get("indexes", []),
            relationships=frame_data.get("relationships", []),
            metadata=frame_data,
        ))
    return frames


def _extract_memory_from_state(state) -> List[MemorySpec]:
    """Extract memory from BackendState"""
    memory = []
    for name, memory_data in state.memories.items():
        scope_str = memory_data.get("scope", "session")
        scope = MemoryScope(scope_str) if scope_str in MemoryScope.__members__.values() else MemoryScope.SESSION
        
        memory.append(MemorySpec(
            name=name,
            scope=scope,
            kind=memory_data.get("kind", "conversation"),
            max_items=memory_data.get("max_items"),
            ttl=memory_data.get("ttl"),
            embedding_model=memory_data.get("embedding_model"),
            metadata=memory_data,
        ))
    return memory


def _extract_chains_from_state(state) -> List[ChainSpec]:
    """Extract chains from BackendState"""
    chains = []
    for name, chain_data in state.chains.items():
        chains.append(ChainSpec(
            name=name,
            steps=chain_data.get("steps", []),
            input_schema=_dict_to_type_spec(chain_data.get("input_schema", {})),
            output_schema=_dict_to_type_spec(chain_data.get("output_schema", {})),
            error_handling=chain_data.get("error_handling", {}),
            metadata=chain_data,
        ))
    return chains


def _extract_insights_from_state(state) -> List[InsightSpec]:
    """Extract insights from BackendState"""
    insights = []
    for name, insight_data in state.insights.items():
        insights.append(InsightSpec(
            name=name,
            query=insight_data.get("query", ""),
            dataset_ref=insight_data.get("dataset", ""),
            aggregations=insight_data.get("aggregations", []),
            filters=insight_data.get("filters", []),
            metadata=insight_data,
        ))
    return insights


def _extract_update_channels(state, datasets: List[DatasetSpec]) -> List[UpdateChannelSpec]:
    """Extract update channels for realtime-enabled datasets"""
    channels = []
    
    for dataset in datasets:
        if dataset.realtime_enabled:
            # Create update channel for this dataset
            channels.append(UpdateChannelSpec(
                name=f"dataset:{dataset.name}:changes",
                dataset_name=dataset.name,
                event_types=["create", "update", "delete"],
                transport="websocket",
                requires_auth=True,
                required_capabilities=dataset.access_policy.get("required_capabilities", []) if dataset.access_policy else [],
                redis_channel=f"namel3ss:dataset:{dataset.name}:updates",
            ))
    
    return channels


def _extract_endpoints_from_state(state) -> List[EndpointIR]:
    """Extract API endpoints from BackendState"""
    endpoints = []
    
    # Prompts become endpoints
    for name, prompt_data in state.prompts.items():
        endpoints.append(EndpointIR(
            path=f"/api/prompts/{name}",
            method=HTTPMethod.POST,
            input_schema=_dict_to_type_spec(prompt_data.get("input_schema", {})),
            output_schema=_dict_to_type_spec(prompt_data.get("output_schema", {})),
            handler_type="prompt",
            handler_ref=f"prompts.{name}",
            description=prompt_data.get("description", ""),
        ))
    
    # Agents become endpoints
    for name, agent_data in state.agents.items():
        endpoints.append(EndpointIR(
            path=f"/api/agents/{name}",
            method=HTTPMethod.POST,
            input_schema=_dict_to_type_spec(agent_data.get("input_schema", {})),
            output_schema=_dict_to_type_spec(agent_data.get("output_schema", {})),
            handler_type="agent",
            handler_ref=f"agents.{name}",
            description=agent_data.get("description", ""),
        ))
    
    # Datasets become endpoints - GET for listing/querying
    # Access original app to get dataset access policies
    original_app = state.__dict__.get("_original_app")
    dataset_ast_map = {}
    if original_app:
        for item in getattr(original_app, "body", []):
            if hasattr(item, "__class__") and item.__class__.__name__ == "Dataset":
                dataset_ast_map[item.name] = item
    
    for name, dataset_data in state.datasets.items():
        # GET endpoint for reading data (always available)
        endpoints.append(EndpointIR(
            path=f"/api/datasets/{name}",
            method=HTTPMethod.GET,
            input_schema=TypeSpec(kind="object"),  # Query params for pagination, sort, filter
            output_schema=TypeSpec(kind="object", fields=[
                SchemaField("data", TypeSpec(kind="array")),
                SchemaField("total", TypeSpec(kind="integer")),
                SchemaField("page", TypeSpec(kind="integer")),
                SchemaField("page_size", TypeSpec(kind="integer")),
            ]),
            handler_type="dataset",
            handler_ref=f"datasets.{name}",
            description=f"List and query {name} dataset with pagination, sorting, and filtering",
        ))
        
        # Check if dataset supports writes
        access_policy = None
        if name in dataset_ast_map:
            dataset_ast = dataset_ast_map[name]
            if hasattr(dataset_ast, "access_policy"):
                access_policy = dataset_ast.access_policy
        
        # POST endpoint for creating records
        if access_policy and access_policy.allow_create:
            endpoints.append(EndpointIR(
                path=f"/api/datasets/{name}",
                method=HTTPMethod.POST,
                input_schema=TypeSpec(kind="object"),  # Record data
                output_schema=TypeSpec(kind="object"),  # Created record with ID
                handler_type="dataset",
                handler_ref=f"datasets.{name}.create",
                description=f"Create new record in {name} dataset",
                auth_required=True,
                allowed_capabilities=access_policy.required_capabilities,
            ))
        
        # PATCH endpoint for updating records
        if access_policy and access_policy.allow_update:
            endpoints.append(EndpointIR(
                path=f"/api/datasets/{name}/{{id}}",
                method=HTTPMethod.PATCH,
                input_schema=TypeSpec(kind="object"),  # Partial record data
                output_schema=TypeSpec(kind="object"),  # Updated record
                handler_type="dataset",
                handler_ref=f"datasets.{name}.update",
                description=f"Update record in {name} dataset",
                auth_required=True,
                allowed_capabilities=access_policy.required_capabilities,
            ))
        
        # DELETE endpoint for deleting records
        if access_policy and access_policy.allow_delete:
            endpoints.append(EndpointIR(
                path=f"/api/datasets/{name}/{{id}}",
                method=HTTPMethod.DELETE,
                input_schema=TypeSpec(kind="object"),
                output_schema=TypeSpec(kind="object", fields=[
                    SchemaField("success", TypeSpec(kind="boolean")),
                ]),
                handler_type="dataset",
                handler_ref=f"datasets.{name}.delete",
                description=f"Delete record from {name} dataset",
                auth_required=True,
                allowed_capabilities=access_policy.required_capabilities,
            ))
    
    return endpoints


def _extract_pages_from_state(state) -> List[PageSpec]:
    """Extract pages from BackendState with component binding metadata"""
    pages = []
    
    # Access original app to get page AST with ShowTable/ShowChart/ShowForm nodes
    original_app = state.__dict__.get("_original_app")
    
    for page_spec in state.pages:
        components = []
        
        # Extract components from original AST if available
        if original_app:
            components = _extract_components_from_page_ast(original_app, page_spec.name, state)
        
        pages.append(PageSpec(
            name=page_spec.name,
            slug=page_spec.slug,
            title=page_spec.name,
            components=components,
            layout=page_spec.layout.get("type", "default") if page_spec.layout else "default",
            metadata={
                "route": page_spec.route,
                "api_path": page_spec.api_path,
                "reactive": page_spec.reactive,
                "refresh_policy": page_spec.refresh_policy,
            },
        ))
    return pages


def _extract_components_from_page_ast(app, page_name: str, state) -> List[ComponentSpec]:
    """Extract ComponentSpec from page AST nodes"""
    components = []
    
    # Find the page in the app AST
    for item in getattr(app, "body", []):
        if hasattr(item, "__class__") and item.__class__.__name__ == "Page" and item.name == page_name:
            # Process page body statements
            for stmt in getattr(item, "body", []):
                component = _statement_to_component_spec(stmt, state)
                if component:
                    components.append(component)
            break
    
    return components


def _statement_to_component_spec(stmt, state) -> Optional[ComponentSpec]:
    """Convert AST statement to ComponentSpec"""
    stmt_type = stmt.__class__.__name__
    
    if stmt_type == "ShowTable":
        return _show_table_to_component(stmt, state)
    elif stmt_type == "ShowChart":
        return _show_chart_to_component(stmt, state)
    elif stmt_type == "ShowForm":
        return _show_form_to_component(stmt, state)
    # Data display components
    elif stmt_type == "ShowDataTable":
        return _show_data_table_to_component(stmt, state)
    elif stmt_type == "ShowDataList":
        return _show_data_list_to_component(stmt, state)
    elif stmt_type == "ShowStatSummary":
        return _show_stat_summary_to_component(stmt, state)
    elif stmt_type == "ShowTimeline":
        return _show_timeline_to_component(stmt, state)
    elif stmt_type == "ShowAvatarGroup":
        return _show_avatar_group_to_component(stmt, state)
    elif stmt_type == "ShowDataChart":
        return _show_data_chart_to_component(stmt, state)
    # Layout primitives
    elif stmt_type == "StackLayout":
        return _stack_layout_to_component(stmt, state)
    elif stmt_type == "GridLayout":
        return _grid_layout_to_component(stmt, state)
    elif stmt_type == "SplitLayout":
        return _split_layout_to_component(stmt, state)
    elif stmt_type == "TabsLayout":
        return _tabs_layout_to_component(stmt, state)
    elif stmt_type == "AccordionLayout":
        return _accordion_layout_to_component(stmt, state)
    # Add other statement types as needed
    
    return None


def _show_table_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowTable AST node to ComponentSpec with binding"""
    from namel3ss.ast.pages import DataBindingConfig
    
    # Build binding spec if binding config present
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        
        # Get dataset spec to find schema fields
        dataset_name = stmt.source
        dataset_spec = None
        for ds in state.datasets.values():
            if ds.get("name") == dataset_name or dataset_name in state.datasets:
                dataset_spec = ds
                break
        
        # Extract sortable/filterable fields from columns or schema
        sortable_fields = stmt.columns if stmt.columns else []
        filterable_fields = sortable_fields.copy()
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=binding_config.page_size,
            enable_sorting=binding_config.enable_sorting,
            sortable_fields=sortable_fields,
            enable_filtering=binding_config.enable_filtering,
            filterable_fields=filterable_fields,
            enable_search=binding_config.enable_search,
            searchable_fields=filterable_fields if binding_config.enable_search else [],
            editable=binding_config.editable,
            enable_create=binding_config.enable_create,
            enable_update=binding_config.enable_update,
            enable_delete=binding_config.enable_delete,
            create_endpoint=f"/api/datasets/{dataset_name}" if binding_config.enable_create else None,
            update_endpoint=f"/api/datasets/{dataset_name}/{{id}}" if binding_config.enable_update else None,
            delete_endpoint=f"/api/datasets/{dataset_name}/{{id}}" if binding_config.enable_delete else None,
            subscribe_to_changes=binding_config.subscribe_to_changes,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval,
            cache_ttl=binding_config.cache_ttl,
            optimistic_updates=binding_config.optimistic_updates,
            field_mapping=binding_config.field_mapping,
        )
    
    return ComponentSpec(
        name=stmt.title,
        type="table",
        props={
            "title": stmt.title,
            "source": stmt.source,
            "source_type": stmt.source_type,
            "columns": stmt.columns,
            "filter_by": stmt.filter_by,
            "sort_by": stmt.sort_by,
            "style": stmt.style,
        },
        data_source=stmt.source,
        binding=binding_spec,
    )


def _show_chart_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowChart AST node to ComponentSpec with binding"""
    from namel3ss.ast.pages import DataBindingConfig
    
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        dataset_name = stmt.source
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=binding_config.page_size,
            enable_sorting=False,  # Charts typically don't need interactive sorting
            sortable_fields=[],
            enable_filtering=binding_config.enable_filtering,
            filterable_fields=[],
            enable_search=False,
            searchable_fields=[],
            editable=False,  # Charts are read-only
            subscribe_to_changes=binding_config.subscribe_to_changes,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval,
            cache_ttl=binding_config.cache_ttl,
        )
    
    return ComponentSpec(
        name=stmt.heading,
        type="chart",
        props={
            "heading": stmt.heading,
            "source": stmt.source,
            "source_type": stmt.source_type,
            "chart_type": stmt.chart_type,
            "x": stmt.x,
            "y": stmt.y,
            "color": stmt.color,
            "encodings": stmt.encodings,
            "style": stmt.style,
        },
        data_source=stmt.source,
        binding=binding_spec,
    )


def _show_form_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowForm AST node to ComponentSpec with binding"""
    from namel3ss.ast.pages import DataBindingConfig
    
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        
        # For forms, bound_dataset specifies the dataset to load/save from
        dataset_name = getattr(stmt, "bound_dataset", None)
        if dataset_name:
            binding_spec = DataBindingSpec(
                dataset_name=dataset_name,
                endpoint_path=f"/api/datasets/{dataset_name}",
                page_size=1,  # Forms typically work with single records
                enable_sorting=False,
                sortable_fields=[],
                enable_filtering=False,
                filterable_fields=[],
                enable_search=False,
                searchable_fields=[],
                editable=binding_config.editable,
                enable_create=binding_config.enable_create,
                enable_update=binding_config.enable_update,
                enable_delete=binding_config.enable_delete,
                create_endpoint=f"/api/datasets/{dataset_name}" if binding_config.enable_create else None,
                update_endpoint=f"/api/datasets/{dataset_name}/{{id}}" if binding_config.enable_update else None,
                delete_endpoint=f"/api/datasets/{dataset_name}/{{id}}" if binding_config.enable_delete else None,
                optimistic_updates=binding_config.optimistic_updates,
                field_mapping=binding_config.field_mapping,
            )
    
    return ComponentSpec(
        name=stmt.title,
        type="form",
        props={
            "title": stmt.title,
            "fields": [{"name": f.name, "type": f.field_type} for f in stmt.fields],
            "on_submit_ops": stmt.on_submit_ops,
            "styles": stmt.styles,
        },
        data_source=getattr(stmt, "bound_dataset", None),
        binding=binding_spec,
    )


def _extract_routes_from_state(state) -> List[RouteSpec]:
    """Extract routes from BackendState"""
    routes = []
    for page_spec in state.pages:
        routes.append(RouteSpec(
            path=page_spec.route,
            page_ref=page_spec.name,
            auth_required=False,  # TODO: Extract from page metadata
        ))
    return routes


def _dict_to_type_spec(data: Dict[str, Any]) -> TypeSpec:
    """Convert dictionary representation to TypeSpec"""
    if not data:
        return TypeSpec(kind="object")
    
    kind = data.get("type", "object")
    
    if kind == "object" and "properties" in data:
        fields = []
        for field_name, field_data in data.get("properties", {}).items():
            fields.append(SchemaField(
                name=field_name,
                type_spec=_dict_to_type_spec(field_data),
                required=field_name in data.get("required", []),
            ))
        return TypeSpec(kind=kind, fields=fields)
    
    elif kind == "array" and "items" in data:
        return TypeSpec(
            kind=kind,
            item_type=_dict_to_type_spec(data["items"])
        )
    
    else:
        return TypeSpec(kind=kind)


# =============================================================================
# Data Display Component Converters
# =============================================================================

def _show_data_table_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowDataTable AST node to IRDataTable"""
    from namel3ss.ast.pages import DataBindingConfig
    
    # Build binding spec
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        dataset_name = stmt.source
        
        # Extract sortable/filterable fields from columns
        sortable_fields = [col.id for col in stmt.columns if col.sortable] if stmt.columns else []
        filterable_fields = [col.id for col in stmt.columns] if stmt.columns else []
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=stmt.page_size,
            enable_sorting=True,
            sortable_fields=sortable_fields,
            enable_filtering=True,
            filterable_fields=filterable_fields,
            enable_search=stmt.toolbar.search is not None if stmt.toolbar else False,
            searchable_fields=filterable_fields if stmt.toolbar and stmt.toolbar.search else [],
            editable=binding_config.editable if binding_config else False,
            enable_create=binding_config.enable_create if binding_config else False,
            enable_update=binding_config.enable_update if binding_config else False,
            enable_delete=binding_config.enable_delete if binding_config else False,
            create_endpoint=f"/api/datasets/{dataset_name}" if binding_config and binding_config.enable_create else None,
            update_endpoint=f"/api/datasets/{dataset_name}/{{id}}" if binding_config and binding_config.enable_update else None,
            delete_endpoint=f"/api/datasets/{dataset_name}/{{id}}" if binding_config and binding_config.enable_delete else None,
            subscribe_to_changes=binding_config.subscribe_to_changes if binding_config else False,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config and binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval if binding_config else None,
            cache_ttl=binding_config.cache_ttl if binding_config else None,
            optimistic_updates=binding_config.optimistic_updates if binding_config else True,
        )
    
    # Convert columns
    ir_columns = []
    if stmt.columns:
        for col in stmt.columns:
            ir_columns.append(IRColumnConfig(
                id=col.id,
                label=col.label,
                field=col.field,
                width=col.width,
                align=col.align,
                sortable=col.sortable,
                format=col.format,
                transform=col.transform,
                render_template=col.render_template,
            ))
    
    # Convert toolbar
    ir_toolbar = None
    if stmt.toolbar:
        ir_toolbar = IRToolbarConfig(
            search=stmt.toolbar.search,
            filters=stmt.toolbar.filters,
            bulk_actions=[_conditional_action_to_dict(a) for a in stmt.toolbar.bulk_actions] if stmt.toolbar.bulk_actions else [],
            actions=[_conditional_action_to_dict(a) for a in stmt.toolbar.actions] if stmt.toolbar.actions else [],
        )
    
    # Convert row actions
    row_actions = [_conditional_action_to_dict(a) for a in stmt.row_actions] if stmt.row_actions else []
    
    # Convert empty state
    empty_state = _empty_state_to_dict(stmt.empty_state) if stmt.empty_state else None
    
    ir_table = IRDataTable(
        title=stmt.title,
        source_type=stmt.source_type,
        source=stmt.source,
        columns=ir_columns,
        row_actions=row_actions,
        toolbar=ir_toolbar,
        filter_by=stmt.filter_by,
        sort_by=stmt.sort_by,
        default_sort=stmt.default_sort,
        page_size=stmt.page_size,
        enable_pagination=stmt.enable_pagination,
        empty_state=empty_state,
        binding=binding_spec,
        layout=_layout_meta_to_dict(stmt.layout) if stmt.layout else None,
        style=stmt.style,
    )
    
    # Wrap in ComponentSpec for compatibility
    return ComponentSpec(
        name=stmt.title,
        type="data_table",
        props={
            "ir_spec": ir_table,
        },
        data_source=stmt.source,
        binding=binding_spec,
        metadata={"ir_type": "IRDataTable"},
    )


def _show_data_list_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowDataList AST node to IRDataList"""
    from namel3ss.ast.pages import DataBindingConfig
    
    # Build binding spec
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        dataset_name = stmt.source
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=stmt.page_size,
            enable_sorting=False,
            sortable_fields=[],
            enable_filtering=True,
            filterable_fields=[],
            enable_search=stmt.enable_search,
            searchable_fields=[],
            editable=False,
            subscribe_to_changes=binding_config.subscribe_to_changes if binding_config else False,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config and binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval if binding_config else None,
            cache_ttl=binding_config.cache_ttl if binding_config else None,
        )
    
    # Convert item config
    ir_item = None
    if stmt.item:
        ir_item = IRListItemConfig(
            avatar=stmt.item.avatar,
            title=stmt.item.title,
            subtitle=stmt.item.subtitle,
            metadata=stmt.item.metadata,
            actions=[_conditional_action_to_dict(a) for a in stmt.item.actions] if stmt.item.actions else [],
            badge=stmt.item.badge,
            icon=stmt.item.icon,
            state_class=stmt.item.state_class,
        )
    
    # Convert empty state
    empty_state = _empty_state_to_dict(stmt.empty_state) if stmt.empty_state else None
    
    ir_list = IRDataList(
        title=stmt.title,
        source_type=stmt.source_type,
        source=stmt.source,
        item=ir_item,
        variant=stmt.variant,
        dividers=stmt.dividers,
        filter_by=stmt.filter_by,
        enable_search=stmt.enable_search,
        search_placeholder=stmt.search_placeholder,
        page_size=stmt.page_size,
        enable_pagination=stmt.enable_pagination,
        empty_state=empty_state,
        binding=binding_spec,
        layout=_layout_meta_to_dict(stmt.layout) if stmt.layout else None,
        style=stmt.style,
    )
    
    return ComponentSpec(
        name=stmt.title,
        type="data_list",
        props={
            "ir_spec": ir_list,
        },
        data_source=stmt.source,
        binding=binding_spec,
        metadata={"ir_type": "IRDataList"},
    )


def _show_stat_summary_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowStatSummary AST node to IRStatSummary"""
    from namel3ss.ast.pages import DataBindingConfig
    
    # Build binding spec
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        dataset_name = stmt.source
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=1,  # Stat summaries typically fetch single values
            enable_sorting=False,
            sortable_fields=[],
            enable_filtering=False,
            filterable_fields=[],
            enable_search=False,
            searchable_fields=[],
            editable=False,
            subscribe_to_changes=binding_config.subscribe_to_changes if binding_config else False,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config and binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval if binding_config else None,
            cache_ttl=binding_config.cache_ttl if binding_config else None,
        )
    
    # Convert sparkline config
    ir_sparkline = None
    if stmt.sparkline:
        ir_sparkline = IRSparklineConfig(
            data_source=stmt.sparkline.data_source,
            x_field=stmt.sparkline.x_field,
            y_field=stmt.sparkline.y_field,
            color=stmt.sparkline.color,
            variant=stmt.sparkline.variant,
        )
    
    ir_stat = IRStatSummary(
        label=stmt.label,
        source_type=stmt.source_type,
        source=stmt.source,
        value=stmt.value,
        format=stmt.format,
        prefix=stmt.prefix,
        suffix=stmt.suffix,
        delta=stmt.delta,
        trend=stmt.trend,
        comparison_period=stmt.comparison_period,
        sparkline=ir_sparkline,
        color=stmt.color,
        icon=stmt.icon,
        binding=binding_spec,
        layout=_layout_meta_to_dict(stmt.layout) if stmt.layout else None,
        style=stmt.style,
    )
    
    return ComponentSpec(
        name=stmt.label,
        type="stat_summary",
        props={
            "ir_spec": ir_stat,
        },
        data_source=stmt.source,
        binding=binding_spec,
        metadata={"ir_type": "IRStatSummary"},
    )


def _show_timeline_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowTimeline AST node to IRTimeline"""
    from namel3ss.ast.pages import DataBindingConfig
    
    # Build binding spec
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        dataset_name = stmt.source
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=stmt.page_size,
            enable_sorting=True,
            sortable_fields=[],
            enable_filtering=True,
            filterable_fields=[],
            enable_search=False,
            searchable_fields=[],
            editable=False,
            subscribe_to_changes=binding_config.subscribe_to_changes if binding_config else False,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config and binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval if binding_config else None,
            cache_ttl=binding_config.cache_ttl if binding_config else None,
        )
    
    # Convert item config
    ir_item = None
    if stmt.item:
        ir_item = IRTimelineItem(
            timestamp=stmt.item.timestamp,
            title=stmt.item.title,
            description=stmt.item.description,
            icon=stmt.item.icon,
            status=stmt.item.status,
            color=stmt.item.color,
            actions=[_conditional_action_to_dict(a) for a in stmt.item.actions] if stmt.item.actions else [],
        )
    
    # Convert empty state
    empty_state = _empty_state_to_dict(stmt.empty_state) if stmt.empty_state else None
    
    ir_timeline = IRTimeline(
        title=stmt.title,
        source_type=stmt.source_type,
        source=stmt.source,
        item=ir_item,
        variant=stmt.variant,
        show_timestamps=stmt.show_timestamps,
        group_by_date=stmt.group_by_date,
        filter_by=stmt.filter_by,
        sort_by=stmt.sort_by,
        page_size=stmt.page_size,
        enable_pagination=stmt.enable_pagination,
        empty_state=empty_state,
        binding=binding_spec,
        layout=_layout_meta_to_dict(stmt.layout) if stmt.layout else None,
        style=stmt.style,
    )
    
    return ComponentSpec(
        name=stmt.title,
        type="timeline",
        props={
            "ir_spec": ir_timeline,
        },
        data_source=stmt.source,
        binding=binding_spec,
        metadata={"ir_type": "IRTimeline"},
    )


def _show_avatar_group_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowAvatarGroup AST node to IRAvatarGroup"""
    from namel3ss.ast.pages import DataBindingConfig
    
    # Build binding spec
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        dataset_name = stmt.source
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=stmt.max_visible + 10,  # Fetch a few extra for "+N more"
            enable_sorting=False,
            sortable_fields=[],
            enable_filtering=True,
            filterable_fields=[],
            enable_search=False,
            searchable_fields=[],
            editable=False,
            subscribe_to_changes=binding_config.subscribe_to_changes if binding_config else False,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config and binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval if binding_config else None,
            cache_ttl=binding_config.cache_ttl if binding_config else None,
        )
    
    # Convert item config
    ir_item = None
    if stmt.item:
        ir_item = IRAvatarItem(
            name=stmt.item.name,
            image_url=stmt.item.image_url,
            initials=stmt.item.initials,
            color=stmt.item.color,
            status=stmt.item.status,
            tooltip=stmt.item.tooltip,
        )
    
    ir_avatar_group = IRAvatarGroup(
        title=stmt.title,
        source_type=stmt.source_type,
        source=stmt.source,
        item=ir_item,
        max_visible=stmt.max_visible,
        size=stmt.size,
        variant=stmt.variant,
        filter_by=stmt.filter_by,
        binding=binding_spec,
        layout=_layout_meta_to_dict(stmt.layout) if stmt.layout else None,
        style=stmt.style,
    )
    
    return ComponentSpec(
        name=stmt.title or "Avatar Group",
        type="avatar_group",
        props={
            "ir_spec": ir_avatar_group,
        },
        data_source=stmt.source,
        binding=binding_spec,
        metadata={"ir_type": "IRAvatarGroup"},
    )


def _show_data_chart_to_component(stmt, state) -> ComponentSpec:
    """Convert ShowDataChart AST node to IRDataChart"""
    from namel3ss.ast.pages import DataBindingConfig
    
    # Build binding spec
    binding_spec = None
    if hasattr(stmt, "binding") and stmt.binding:
        binding_config: DataBindingConfig = stmt.binding
        dataset_name = stmt.source
        
        binding_spec = DataBindingSpec(
            dataset_name=dataset_name,
            endpoint_path=f"/api/datasets/{dataset_name}",
            page_size=1000,  # Charts can handle more data
            enable_sorting=False,
            sortable_fields=[],
            enable_filtering=True,
            filterable_fields=[],
            enable_search=False,
            searchable_fields=[],
            editable=False,
            subscribe_to_changes=binding_config.subscribe_to_changes if binding_config else False,
            websocket_topic=f"dataset:{dataset_name}:changes" if binding_config and binding_config.subscribe_to_changes else None,
            polling_interval=binding_config.refresh_interval if binding_config else None,
            cache_ttl=binding_config.cache_ttl if binding_config else None,
        )
    
    # Convert chart config
    ir_config = None
    if stmt.config:
        ir_config = IRChartConfig(
            variant=stmt.config.variant,
            x_field=stmt.config.x_field,
            y_fields=stmt.config.y_fields,
            group_by=stmt.config.group_by,
            stacked=stmt.config.stacked,
            smooth=stmt.config.smooth,
            fill=stmt.config.fill,
            legend=stmt.config.legend,
            tooltip=stmt.config.tooltip,
            x_axis=stmt.config.x_axis,
            y_axis=stmt.config.y_axis,
            colors=stmt.config.colors,
            color_scheme=stmt.config.color_scheme,
        )
    
    # Convert empty state
    empty_state = _empty_state_to_dict(stmt.empty_state) if stmt.empty_state else None
    
    ir_chart = IRDataChart(
        title=stmt.title,
        source_type=stmt.source_type,
        source=stmt.source,
        config=ir_config,
        filter_by=stmt.filter_by,
        sort_by=stmt.sort_by,
        empty_state=empty_state,
        binding=binding_spec,
        layout=_layout_meta_to_dict(stmt.layout) if stmt.layout else None,
        style=stmt.style,
        height=stmt.height,
    )
    
    return ComponentSpec(
        name=stmt.title,
        type="data_chart",
        props={
            "ir_spec": ir_chart,
        },
        data_source=stmt.source,
        binding=binding_spec,
        metadata={"ir_type": "IRDataChart"},
    )


# Helper functions for conversions

def _conditional_action_to_dict(action) -> Dict[str, Any]:
    """Convert ConditionalAction AST node to dictionary"""
    return {
        "label": action.label,
        "action_type": action.action_type,
        "action_target": action.action_target,
        "params": action.params,
        "condition": action.condition,
        "style": action.style,
        "icon": action.icon,
        "confirm": action.confirm,
    }


def _empty_state_to_dict(empty_state) -> Dict[str, Any]:
    """Convert EmptyStateConfig AST node to dictionary"""
    if not empty_state:
        return None
    return {
        "icon": empty_state.icon,
        "title": empty_state.title,
        "message": empty_state.message,
        "action": empty_state.action,
    }


def _layout_meta_to_dict(layout_meta) -> Dict[str, Any]:
    """Convert LayoutMeta AST node to dictionary"""
    if not layout_meta:
        return None
    return {
        "direction": layout_meta.direction,
        "spacing": layout_meta.spacing,
        "width": layout_meta.width,
        "height": layout_meta.height,
        "variant": layout_meta.variant,
        "align": layout_meta.align,
        "emphasis": layout_meta.emphasis,
        "extras": layout_meta.extras,
    }


# =============================================================================
# Layout Primitive Converters
# =============================================================================

def _stack_layout_to_component(stmt, state) -> ComponentSpec:
    """Convert StackLayout AST node to ComponentSpec with children"""
    from .spec import IRStackLayout
    
    # Recursively convert children
    children = []
    for child_stmt in stmt.children:
        child_spec = _statement_to_component_spec(child_stmt, state)
        if child_spec:
            children.append(child_spec)
    
    layout_ir = IRStackLayout(
        direction=stmt.direction,
        gap=stmt.gap,
        align=stmt.align,
        justify=stmt.justify,
        wrap=stmt.wrap,
        children=children,
        style=stmt.style,
        layout_meta=stmt.layout if hasattr(stmt, 'layout') else None,
    )
    
    return ComponentSpec(
        name=f"stack_{id(stmt)}",
        type="stack",
        props={
            "direction": stmt.direction,
            "gap": stmt.gap,
            "align": stmt.align,
            "justify": stmt.justify,
            "wrap": stmt.wrap,
        },
        children=children,
        layout=layout_ir,
    )


def _grid_layout_to_component(stmt, state) -> ComponentSpec:
    """Convert GridLayout AST node to ComponentSpec with children"""
    from .spec import IRGridLayout
    
    # Recursively convert children
    children = []
    for child_stmt in stmt.children:
        child_spec = _statement_to_component_spec(child_stmt, state)
        if child_spec:
            children.append(child_spec)
    
    layout_ir = IRGridLayout(
        columns=stmt.columns,
        min_column_width=stmt.min_column_width,
        gap=stmt.gap,
        responsive=stmt.responsive,
        children=children,
        style=stmt.style,
        layout_meta=stmt.layout if hasattr(stmt, 'layout') else None,
    )
    
    return ComponentSpec(
        name=f"grid_{id(stmt)}",
        type="grid",
        props={
            "columns": stmt.columns,
            "minColumnWidth": stmt.min_column_width,
            "gap": stmt.gap,
            "responsive": stmt.responsive,
        },
        children=children,
        layout=layout_ir,
    )


def _split_layout_to_component(stmt, state) -> ComponentSpec:
    """Convert SplitLayout AST node to ComponentSpec with left/right children"""
    from .spec import IRSplitLayout
    
    # Recursively convert left children
    left_children = []
    for child_stmt in stmt.left:
        child_spec = _statement_to_component_spec(child_stmt, state)
        if child_spec:
            left_children.append(child_spec)
    
    # Recursively convert right children
    right_children = []
    for child_stmt in stmt.right:
        child_spec = _statement_to_component_spec(child_stmt, state)
        if child_spec:
            right_children.append(child_spec)
    
    layout_ir = IRSplitLayout(
        left=left_children,
        right=right_children,
        ratio=stmt.ratio,
        resizable=stmt.resizable,
        orientation=stmt.orientation,
        style=stmt.style,
        layout_meta=stmt.layout if hasattr(stmt, 'layout') else None,
    )
    
    return ComponentSpec(
        name=f"split_{id(stmt)}",
        type="split",
        props={
            "ratio": stmt.ratio,
            "resizable": stmt.resizable,
            "orientation": stmt.orientation,
        },
        children=left_children + right_children,  # Combine for ComponentSpec
        layout=layout_ir,
    )


def _tabs_layout_to_component(stmt, state) -> ComponentSpec:
    """Convert TabsLayout AST node to ComponentSpec with tab children"""
    from .spec import IRTabsLayout, IRTabItem
    
    # Convert tabs
    tabs_ir = []
    all_children = []
    for tab in stmt.tabs:
        # Recursively convert tab content
        tab_children = []
        for child_stmt in tab.content:
            child_spec = _statement_to_component_spec(child_stmt, state)
            if child_spec:
                tab_children.append(child_spec)
                all_children.append(child_spec)
        
        tab_ir = IRTabItem(
            id=tab.id,
            label=tab.label,
            icon=tab.icon,
            badge=tab.badge,
            content=tab_children,
        )
        tabs_ir.append(tab_ir)
    
    layout_ir = IRTabsLayout(
        tabs=tabs_ir,
        default_tab=stmt.default_tab,
        persist_state=stmt.persist_state,
        style=stmt.style,
        layout_meta=stmt.layout if hasattr(stmt, 'layout') else None,
    )
    
    return ComponentSpec(
        name=f"tabs_{id(stmt)}",
        type="tabs",
        props={
            "defaultTab": stmt.default_tab,
            "persistState": stmt.persist_state,
        },
        children=all_children,
        layout=layout_ir,
    )


def _accordion_layout_to_component(stmt, state) -> ComponentSpec:
    """Convert AccordionLayout AST node to ComponentSpec with accordion items"""
    from .spec import IRAccordionLayout, IRAccordionItem
    
    # Convert items
    items_ir = []
    all_children = []
    for item in stmt.items:
        # Recursively convert item content
        item_children = []
        for child_stmt in item.content:
            child_spec = _statement_to_component_spec(child_stmt, state)
            if child_spec:
                item_children.append(child_spec)
                all_children.append(child_spec)
        
        item_ir = IRAccordionItem(
            id=item.id,
            title=item.title,
            description=item.description,
            icon=item.icon,
            default_open=item.default_open,
            content=item_children,
        )
        items_ir.append(item_ir)
    
    layout_ir = IRAccordionLayout(
        items=items_ir,
        multiple=stmt.multiple,
        style=stmt.style,
        layout_meta=stmt.layout if hasattr(stmt, 'layout') else None,
    )
    
    return ComponentSpec(
        name=f"accordion_{id(stmt)}",
        type="accordion",
        props={
            "multiple": stmt.multiple,
        },
        children=all_children,
        layout=layout_ir,
    )


__all__ = [
    "build_backend_ir",
    "build_frontend_ir",
]
