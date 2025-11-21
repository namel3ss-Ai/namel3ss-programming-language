"""
Graph execution API endpoints with production-grade validation and instrumentation.

Provides endpoints for executing visual agent graphs with full OpenTelemetry
tracing, cost tracking, and error handling.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from n3_server.api.auth import get_current_user  # Placeholder for auth
from n3_server.converter import EnhancedN3ASTConverter, GraphJSON, ConversionError
from n3_server.db.models import Project, User
from n3_server.db.session import get_db
from n3_server.execution.executor import GraphExecutor, ExecutionContext, ExecutionSpan as ExecSpan
from n3_server.execution.registry import RuntimeRegistry, RegistryError
from namel3ss.llm.registry import get_llm, LLMRegistryError


router = APIRouter(prefix="/execution", tags=["execution"])
tracer = trace.get_tracer(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class ExecutionOptions(BaseModel):
    """Options for graph execution."""
    
    max_steps: int = Field(default=100, ge=1, le=1000, description="Maximum steps to execute")
    timeout_ms: Optional[int] = Field(default=None, ge=1000, description="Execution timeout in milliseconds")
    trace_level: str = Field(default="full", description="Trace detail level: full, summary, none")
    
    @field_validator("trace_level")
    @classmethod
    def validate_trace_level(cls, v: str) -> str:
        if v not in ["full", "summary", "none"]:
            raise ValueError("trace_level must be one of: full, summary, none")
        return v


class ExecutionRequest(BaseModel):
    """Request to execute a graph."""
    
    entry: str = Field(default="start", description="Entry node ID or 'start'")
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data for execution")
    options: Optional[ExecutionOptions] = Field(default=None, description="Execution options")


class SpanAttributeResponse(BaseModel):
    """Attributes for an execution span."""
    
    model: Optional[str] = None
    temperature: Optional[float] = None
    tokens_prompt: Optional[int] = Field(None, alias="tokensPrompt")
    tokens_completion: Optional[int] = Field(None, alias="tokensCompletion")
    cost: Optional[float] = None
    top_k: Optional[int] = Field(None, alias="topK")
    reranker: Optional[str] = None
    tool_name: Optional[str] = Field(None, alias="toolName")
    error: Optional[str] = None
    
    class Config:
        populate_by_name = True


class ExecutionSpanResponse(BaseModel):
    """A traced execution span."""
    
    span_id: str = Field(alias="spanId")
    parent_span_id: Optional[str] = Field(None, alias="parentSpanId")
    name: str
    type: str
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    duration_ms: float = Field(alias="durationMs")
    status: str
    attributes: SpanAttributeResponse
    input_data: Optional[Any] = Field(None, alias="inputData")
    output_data: Optional[Any] = Field(None, alias="outputData")
    
    class Config:
        populate_by_name = True


class ExecutionMetrics(BaseModel):
    """Aggregated execution metrics."""
    
    total_duration_ms: float = Field(alias="totalDurationMs")
    total_tokens_prompt: int = Field(alias="totalTokensPrompt")
    total_tokens_completion: int = Field(alias="totalTokensCompletion")
    total_cost: float = Field(alias="totalCost")
    span_count: int = Field(alias="spanCount")
    
    class Config:
        populate_by_name = True


class ExecutionResponse(BaseModel):
    """Response from graph execution."""
    
    execution_id: str = Field(alias="executionId")
    project_id: str = Field(alias="projectId")
    status: str  # "success", "error", "timeout"
    result: Any
    trace: List[ExecutionSpanResponse]
    metrics: ExecutionMetrics
    error: Optional[str] = None
    
    class Config:
        populate_by_name = True


class ValidationError(BaseModel):
    """Graph validation error."""
    
    node_id: Optional[str] = Field(None, alias="nodeId")
    message: str
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        populate_by_name = True


class ValidationResponse(BaseModel):
    """Response from graph validation."""
    
    valid: bool
    errors: List[ValidationError]


# ============================================================================
# Helper Functions
# ============================================================================

async def build_llm_registry(project: Project) -> Dict[str, Any]:
    """
    Build LLM registry from project configuration.
    
    In production, this would:
    - Load API keys from secure storage (Vault, AWS Secrets Manager)
    - Support multiple providers (OpenAI, Anthropic, Azure, etc.)
    - Handle provider-specific configuration
    
    For now, we use a simple approach with environment-based keys.
    """
    import os
    
    llm_registry = {}
    
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Register commonly used models
    if openai_api_key:
        try:
            llm_registry["gpt-4"] = get_llm("openai", model="gpt-4", api_key=openai_api_key)
            llm_registry["gpt-4-turbo"] = get_llm("openai", model="gpt-4-turbo-preview", api_key=openai_api_key)
            llm_registry["gpt-3.5-turbo"] = get_llm("openai", model="gpt-3.5-turbo", api_key=openai_api_key)
        except LLMRegistryError:
            pass  # Model not available
    
    if anthropic_api_key:
        try:
            llm_registry["claude-3-opus"] = get_llm("anthropic", model="claude-3-opus-20240229", api_key=anthropic_api_key)
            llm_registry["claude-3-sonnet"] = get_llm("anthropic", model="claude-3-sonnet-20240229", api_key=anthropic_api_key)
            llm_registry["claude-3-haiku"] = get_llm("anthropic", model="claude-3-haiku-20240307", api_key=anthropic_api_key)
        except LLMRegistryError:
            pass
    
    return llm_registry


async def build_tool_registry() -> Dict[str, Any]:
    """
    Build tool registry.
    
    In production, this would:
    - Load tool definitions from database
    - Support custom Python functions
    - Handle tool dependencies and security
    
    For now, return empty registry. Tools can be added as needed.
    """
    return {}


def calculate_metrics(spans: List[ExecSpan]) -> ExecutionMetrics:
    """Calculate aggregated metrics from execution spans."""
    total_duration_ms = max((s.end_time - s.start_time).total_seconds() * 1000 for s in spans) if spans else 0
    total_tokens_prompt = sum(s.attributes.tokens_prompt or 0 for s in spans)
    total_tokens_completion = sum(s.attributes.tokens_completion or 0 for s in spans)
    total_cost = sum(s.attributes.cost or 0 for s in spans)
    
    return ExecutionMetrics(
        total_duration_ms=total_duration_ms,
        total_tokens_prompt=total_tokens_prompt,
        total_tokens_completion=total_tokens_completion,
        total_cost=total_cost,
        span_count=len(spans),
    )


def convert_span_to_response(span: ExecSpan) -> ExecutionSpanResponse:
    """Convert internal ExecutionSpan to API response format."""
    return ExecutionSpanResponse(
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        name=span.name,
        type=span.type.value,
        start_time=span.start_time.isoformat(),
        end_time=span.end_time.isoformat(),
        duration_ms=span.duration_ms,
        status=span.status,
        attributes=SpanAttributeResponse(
            model=span.attributes.model,
            temperature=span.attributes.temperature,
            tokens_prompt=span.attributes.tokens_prompt,
            tokens_completion=span.attributes.tokens_completion,
            cost=span.attributes.cost,
            top_k=span.attributes.top_k,
            reranker=span.attributes.reranker,
            tool_name=span.attributes.tool_name,
            error=span.attributes.error,
        ),
        input_data=span.input_data,
        output_data=span.output_data,
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/graphs/{project_id}/execute", response_model=ExecutionResponse)
async def execute_graph(
    project_id: str,
    request: ExecutionRequest,
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(get_current_user),  # Enable when auth is ready
):
    """
    Execute a graph with full instrumentation and validation.
    
    This endpoint:
    1. Validates project access (TODO: enable auth)
    2. Loads and validates graph structure with Pydantic
    3. Converts graph to N3 AST with EnhancedN3ASTConverter
    4. Builds RuntimeRegistry with LLMs, tools, agents, RAG pipelines
    5. Executes with GraphExecutor and full OpenTelemetry tracing
    6. Returns results, trace, and metrics
    
    Args:
        project_id: Project ID containing the graph
        request: Execution request with entry point and input data
        db: Database session
        current_user: Authenticated user (TODO: enable)
    
    Returns:
        ExecutionResponse with result, trace, and metrics
    
    Raises:
        HTTPException 404: Project not found
        HTTPException 400: Validation or execution error
        HTTPException 500: Internal server error
    """
    execution_id = str(uuid4())
    
    with tracer.start_as_current_span("execute_graph") as span:
        span.set_attribute("execution.id", execution_id)
        span.set_attribute("execution.project_id", project_id)
        span.set_attribute("execution.entry", request.entry)
        
        try:
            # 1. Load project
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project '{project_id}' not found"
                )
            
            # TODO: Validate user has access to project
            # if not has_project_access(current_user, project):
            #     raise HTTPException(status_code=403, detail="Access denied")
            
            # 2. Load and validate graph JSON
            graph_data = project.graph_data or {}
            if not graph_data.get("nodes"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Project has no graph data"
                )
            
            # Prepare graph JSON for validation
            graph_json_dict = {
                "projectId": project_id,
                "name": project.name,
                "nodes": graph_data.get("nodes", []),
                "edges": graph_data.get("edges", []),
                "variables": graph_data.get("variables", {}),
            }
            
            try:
                graph_json = GraphJSON.model_validate(graph_json_dict)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid graph structure: {str(e)}"
                )
            
            # 3. Convert graph to AST with validation
            converter = EnhancedN3ASTConverter()
            
            try:
                chain_ast, conversion_context = await converter.convert_graph_to_chain(graph_json)
                span.set_attribute("conversion.nodes", len(graph_json.nodes))
                span.set_attribute("conversion.edges", len(graph_json.edges))
            except ConversionError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Graph conversion failed at node '{e.node_id}': {e.details}"
                )
            
            # 4. Build runtime registry
            llm_registry = await build_llm_registry(project)
            tool_registry = await build_tool_registry()
            
            if not llm_registry:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No LLM providers configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY."
                )
            
            try:
                runtime_registry = await RuntimeRegistry.from_conversion_context(
                    context=conversion_context,
                    llm_registry=llm_registry,
                    tool_registry=tool_registry,
                )
                span.set_attribute("registry.agents", len(runtime_registry.agents))
                span.set_attribute("registry.prompts", len(runtime_registry.prompts))
                span.set_attribute("registry.rag_pipelines", len(runtime_registry.rag_pipelines))
            except RegistryError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Runtime registry build failed: {str(e)}"
                )
            
            # 5. Create execution context
            execution_context = ExecutionContext(
                project_id=project_id,
                entry_node=request.entry,
                input_data=request.input,
                options=request.options.model_dump() if request.options else {},
            )
            
            # 6. Execute with GraphExecutor
            executor = GraphExecutor(registry=runtime_registry)
            
            try:
                result = await executor.execute_chain(
                    chain=chain_ast,
                    input_data=request.input,
                    context=execution_context,
                )
                
                # 7. Convert spans to response format
                trace_spans = [
                    convert_span_to_response(s)
                    for s in execution_context.spans
                ]
                
                # 8. Calculate metrics
                metrics = calculate_metrics(execution_context.spans)
                
                span.set_attribute("execution.status", "success")
                span.set_attribute("execution.spans", len(trace_spans))
                span.set_attribute("execution.cost", metrics.total_cost)
                span.set_status(Status(StatusCode.OK))
                
                return ExecutionResponse(
                    execution_id=execution_id,
                    project_id=project_id,
                    status="success",
                    result=result,
                    trace=trace_spans,
                    metrics=metrics,
                    error=None,
                )
                
            except Exception as e:
                # Execution error - still return partial trace
                trace_spans = [
                    convert_span_to_response(s)
                    for s in execution_context.spans
                ]
                metrics = calculate_metrics(execution_context.spans)
                
                span.set_attribute("execution.status", "error")
                span.set_attribute("execution.error", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                return ExecutionResponse(
                    execution_id=execution_id,
                    project_id=project_id,
                    status="error",
                    result=None,
                    trace=trace_spans,
                    metrics=metrics,
                    error=str(e),
                )
        
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )


@router.post("/graphs/{project_id}/validate", response_model=ValidationResponse)
async def validate_graph(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(get_current_user),  # Enable when auth is ready
):
    """
    Validate graph structure without executing.
    
    Useful for:
    - Pre-flight validation in editor
    - Checking graph before saving
    - Identifying structural issues
    
    Args:
        project_id: Project ID containing the graph
        db: Database session
        current_user: Authenticated user (TODO: enable)
    
    Returns:
        ValidationResponse with validation status and errors
    """
    with tracer.start_as_current_span("validate_graph") as span:
        span.set_attribute("validation.project_id", project_id)
        
        try:
            # Load project
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project '{project_id}' not found"
                )
            
            # Load graph data
            graph_data = project.graph_data or {}
            if not graph_data.get("nodes"):
                return ValidationResponse(
                    valid=False,
                    errors=[ValidationError(message="No graph data found")]
                )
            
            # Prepare graph JSON
            graph_json_dict = {
                "projectId": project_id,
                "name": project.name,
                "nodes": graph_data.get("nodes", []),
                "edges": graph_data.get("edges", []),
                "variables": graph_data.get("variables", {}),
            }
            
            # Validate structure
            errors = []
            
            try:
                graph_json = GraphJSON.model_validate(graph_json_dict)
            except Exception as e:
                errors.append(ValidationError(
                    message=f"Invalid graph structure: {str(e)}"
                ))
                return ValidationResponse(valid=False, errors=errors)
            
            # Validate conversion
            converter = EnhancedN3ASTConverter()
            validation_errors = converter.validate_graph(graph_json)
            
            for err in validation_errors:
                errors.append(ValidationError(
                    node_id=err.get("node_id"),
                    message=err.get("message", "Unknown error"),
                    details=err.get("details"),
                ))
            
            is_valid = len(errors) == 0
            
            span.set_attribute("validation.valid", is_valid)
            span.set_attribute("validation.error_count", len(errors))
            span.set_status(Status(StatusCode.OK))
            
            return ValidationResponse(
                valid=is_valid,
                errors=errors,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Validation failed: {str(e)}"
            )
