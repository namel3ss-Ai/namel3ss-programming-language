"""
Code generation for logic query HTTP endpoints.

Generates FastAPI routers that expose queries as REST endpoints with
parameter binding and result streaming.
"""

from __future__ import annotations

from typing import List

from namel3ss.ast import App
from namel3ss.ast.logic import LogicQuery


def _render_queries_router_module(app: App) -> str:
    """
    Generate a FastAPI router module for logic queries.
    
    Creates endpoints like:
    - GET /queries/ - List all queries
    - POST /queries/{query_name}/execute - Execute a query with parameters
    """
    if not app.queries:
        return ""
    
    lines = [
        '"""Query execution endpoints."""',
        "",
        "import logging",
        "from typing import Any, Dict, List, Optional",
        "from fastapi import APIRouter, HTTPException",
        "from pydantic import BaseModel",
        "",
        "from ..runtime import APP",
        "from ..runtime.logic_engine import LogicEngineConfig",
        "from ..runtime.logic_adapters import AdapterRegistry",
        "from ..runtime.query_compiler import QueryContext, QueryCompiler",
        "from ..runtime.dataset_adapter_factory import create_adapter_registry",
        "from ..runtime.logic_deserializer import load_queries_from_runtime, load_knowledge_modules_from_runtime",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "router = APIRouter(prefix=\"/queries\", tags=[\"queries\"])",
        "",
    ]
    
    # Serialize query metadata (goals as strings since LogicStruct isn't JSON serializable)
    lines.extend([
        "# Query metadata",
        "QUERIES = {",
    ])
    
    for query in app.queries:
        lines.append(f'    "{query.name}": {{')
        lines.append(f'        "name": "{query.name}",')
        lines.append(f'        "knowledge_sources": {query.knowledge_sources},')
        # Convert goals to string representations
        goal_strs = [f'"{g.functor}/{len(g.args)}"' for g in query.goals]
        lines.append(f'        "goals": [{", ".join(goal_strs)}],')
        if query.variables:
            lines.append(f'        "variables": {query.variables},')
        if query.limit:
            lines.append(f'        "limit": {query.limit},')
        lines.append("    },")
    
    lines.extend([
        "}",
        "",
    ])
    
    # Add request/response models
    lines.extend([
        "class QueryExecuteRequest(BaseModel):",
        '    """Request body for query execution."""',
        "    parameters: Dict[str, Any] = {}",
        "    limit: Optional[int] = None",
        "    offset: Optional[int] = 0",
        "    max_depth: Optional[int] = 100",
        "    max_steps: Optional[int] = 10000",
        "    timeout_seconds: Optional[float] = 10.0",
        "",
        "",
        "class QueryExecuteResponse(BaseModel):",
        '    """Response for query execution."""',
        "    query_name: str",
        "    results: List[Dict[str, Any]]",
        "    count: int",
        "    limited: bool",
        "    offset: int",
        "",
        "",
    ])
    
    # List queries endpoint
    lines.extend([
        "@router.get(\"/\")",
        "async def list_queries():",
        '    """List all available queries."""',
        "    return {",
        '        "queries": list(QUERIES.keys()),',
        '        "metadata": QUERIES,',
        "    }",
        "",
        "",
    ])
    
    # Execute query endpoint with full implementation
    lines.extend([
        "@router.post(\"/{query_name}/execute\", response_model=QueryExecuteResponse)",
        "async def execute_query(query_name: str, request: QueryExecuteRequest):",
        '    """Execute a query with optional parameters."""',
        '    # Validate query exists',
        '    if query_name not in QUERIES:',
        '        raise HTTPException(status_code=404, detail=f"Query {query_name!r} not found")',
        "",
        "    try:",
        "        # Load queries and knowledge modules from runtime",
        "        from ..runtime import APP_QUERIES, KNOWLEDGE_MODULES",
        "",
        "        # Deserialize AST objects from runtime dicts",
        "        queries = load_queries_from_runtime(APP_QUERIES)",
        "        knowledge_modules = load_knowledge_modules_from_runtime(KNOWLEDGE_MODULES)",
        "",
        "        # Find the query AST",
        "        query_ast = None",
        "        for q in queries:",
        "            if q.name == query_name:",
        "                query_ast = q",
        "                break",
        "",
        "        if query_ast is None:",
        '            raise HTTPException(status_code=404, detail=f"Query {query_name!r} AST not found")',
        "",
        "        # Build adapter registry from datasets",
        "        adapter_registry = create_adapter_registry(APP)",
        "",
        "        # Build query context",
        "        knowledge_map = {km.name: km for km in knowledge_modules}",
        "        context = QueryContext(",
        "            knowledge_modules=knowledge_map,",
        "            adapter_registry=adapter_registry,",
        "        )",
        "",
        "        # Configure engine with safety limits",
        "        engine_config = LogicEngineConfig(",
        "            max_depth=min(request.max_depth or 100, 1000),",
        "            max_steps=min(request.max_steps or 10000, 100000),",
        "            timeout_seconds=min(request.timeout_seconds or 10.0, 60.0),",
        "        )",
        "",
        "        # Compile query",
        "        compiler = QueryCompiler(context, engine_config=engine_config)",
        "        compiled_query = compiler.compile_query(query_ast)",
        "",
        "        # Validate and bind parameters",
        "        if query_ast.variables:",
        "            for var in query_ast.variables:",
        "                if var not in request.parameters:",
        '                    raise HTTPException(',
        "                        status_code=400,",
        f'                        detail=f"Missing required parameter: {{var!r}}"',
        "                    )",
        "",
        "        # Apply limit from request",
        "        effective_limit = request.limit or query_ast.limit",
        "        if effective_limit is not None:",
        "            compiled_query.limit = effective_limit",
        "",
        "        # Execute query",
        "        all_results = compiled_query.execute_all()",
        "",
        "        # Apply pagination",
        "        offset = request.offset or 0",
        "        paginated_results = all_results[offset:]",
        "        if effective_limit is not None:",
        "            paginated_results = paginated_results[:effective_limit]",
        "",
        "        # Check if results were limited",
        "        limited = (",
        "            effective_limit is not None",
        "            and len(all_results) > offset + len(paginated_results)",
        "        )",
        "",
        "        # Bind parameters to results if specified",
        "        if request.parameters:",
        "            for result in paginated_results:",
        "                for key, value in request.parameters.items():",
        "                    if key not in result:",
        "                        result[key] = value",
        "",
        "        return QueryExecuteResponse(",
        "            query_name=query_name,",
        "            results=paginated_results,",
        "            count=len(paginated_results),",
        "            limited=limited,",
        "            offset=offset,",
        "        )",
        "",
        "    except HTTPException:",
        "        raise",
        "    except Exception as e:",
        "        logger.exception(f\"Error executing query {query_name!r}\")",
        '        raise HTTPException(',
        "            status_code=500,",
        '            detail=f"Query execution failed: {type(e).__name__}"',
        "        )",
        "",
    ]
        "        )",
        "",
        "    except HTTPException:",
        "        raise",
        "    except Exception as e:",
        "        logger.exception(f\"Error executing query {query_name!r}\")",
        '        raise HTTPException(',
        "            status_code=500,",
        '            detail=f"Query execution failed: {type(e).__name__}"',
        "        )",
        "",
    ])
    
    return "\n".join(lines)


def _generate_query_execution_code(query: LogicQuery) -> List[str]:
    """
    Generate Python code to execute a specific query.
    
    Returns lines of code that:
    1. Set up QueryContext with knowledge modules and adapters
    2. Compile the query
    3. Execute and return results
    """
    lines = []
    
    # Setup context
    lines.extend([
        "# Setup query context",
        "from app import APP_INSTANCE  # Access to parsed app",
        "from ..runtime.query_compiler import QueryContext, QueryCompiler",
        "from ..runtime.logic_engine import LogicEngineConfig",
        "",
        "knowledge_map = {km.name: km for km in APP_INSTANCE.knowledge_modules}",
        "adapter_registry = AdapterRegistry()",
        "",
        "# TODO: Register dataset adapters based on app.datasets",
        "",
        "context = QueryContext(",
        "    knowledge_modules=knowledge_map,",
        "    adapter_registry=adapter_registry,",
        ")",
        "",
    ])
    
    # Configure engine
    lines.extend([
        "# Configure engine with request parameters",
        "engine_config = LogicEngineConfig(",
        "    max_depth=request.max_depth,",
        "    max_steps=request.max_steps,",
        "    timeout_seconds=request.timeout_seconds,",
        ")",
        "",
    ])
    
    # Compile and execute
    lines.extend([
        "# Compile query",
        "compiler = QueryCompiler(context, engine_config=engine_config)",
        f'query_ast = next(q for q in APP_INSTANCE.queries if q.name == "{query.name}")',
        "compiled = compiler.compile_query(query_ast)",
        "",
        "# Apply limit from request if provided",
        "if request.limit is not None:",
        "    compiled.limit = request.limit",
        "",
        "# Execute query",
        "results = compiled.execute_all()",
        "",
        "return QueryExecuteResponse(",
        f'    query_name="{query.name}",',
        "    results=results,",
        "    count=len(results),",
        "    limited=request.limit is not None and len(results) >= request.limit,",
        ")",
    ])
    
    return lines


__all__ = [
    "_render_queries_router_module",
    "_generate_query_execution_code",
]
