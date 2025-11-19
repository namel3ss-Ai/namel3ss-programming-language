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
        "from typing import Any, Dict, List, Optional",
        "from fastapi import APIRouter, HTTPException",
        "from pydantic import BaseModel",
        "",
        "from ..runtime.logic_engine import LogicEngineConfig",
        "from ..runtime.logic_adapters import AdapterRegistry, DatasetAdapter",
        "from ..runtime.query_compiler import QueryContext, QueryCompiler",
        "",
        "router = APIRouter(prefix=\"/queries\", tags=[\"queries\"])",
        "",
    ]
    
    # Add query metadata
    lines.extend([
        "# Query metadata",
        "QUERIES = {",
    ])
    
    for query in app.queries:
        lines.append(f'    "{query.name}": {{')
        lines.append(f'        "name": "{query.name}",')
        lines.append(f'        "knowledge_sources": {query.knowledge_sources},')
        lines.append(f'        "goals": {[str(g) for g in query.goals]},')
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
    
    # Execute query endpoint
    lines.extend([
        "@router.post(\"/{query_name}/execute\", response_model=QueryExecuteResponse)",
        "async def execute_query(query_name: str, request: QueryExecuteRequest):",
        '    """Execute a query with optional parameters."""',
        '    if query_name not in QUERIES:',
        '        raise HTTPException(status_code=404, detail=f"Query not found: {query_name}")',
        "",
        "    # Load query from AST (in real implementation, load from app)",
        "    # For now, return mock results",
        "    # TODO: Integrate with QueryCompiler and execute actual query",
        "",
        "    return QueryExecuteResponse(",
        "        query_name=query_name,",
        "        results=[],",
        "        count=0,",
        "        limited=False,",
        "    )",
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
