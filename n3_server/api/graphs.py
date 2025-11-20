from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from n3_server.db.session import get_db
from n3_server.db.models import Project
from n3_server.converter import N3ASTConverter
from n3_server.execution import GraphExecutor, ExecutionContext

router = APIRouter()
tracer = trace.get_tracer(__name__)


class NodeData(BaseModel):
    id: str
    type: str
    label: str
    data: dict[str, Any]
    position: dict[str, float] | None = None


class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    label: str | None = None
    conditionExpr: str | None = None


class ChainInfo(BaseModel):
    id: str
    name: str


class AgentInfo(BaseModel):
    id: str
    name: str


class GraphResponse(BaseModel):
    projectId: str
    name: str
    chains: list[ChainInfo]
    agents: list[AgentInfo]
    activeRootId: str
    nodes: list[NodeData]
    edges: list[EdgeData]
    metadata: dict[str, Any]


class GraphUpdatePayload(BaseModel):
    nodes: list[NodeData]
    edges: list[EdgeData]
    metadata: dict[str, Any]


class ExecutionRequest(BaseModel):
    entry: str
    input: dict[str, Any]
    options: dict[str, Any] | None = None


class SpanAttribute(BaseModel):
    model: str | None = None
    temperature: float | None = None
    tokensPrompt: int | None = None
    tokensCompletion: int | None = None
    cost: float | None = None


class ExecutionSpan(BaseModel):
    spanId: str
    parentSpanId: str | None
    name: str
    type: str
    startTime: str
    endTime: str
    durationMs: float
    status: str
    attributes: SpanAttribute
    input: Any | None = None
    output: Any | None = None


class ExecutionResult(BaseModel):
    result: Any
    trace: list[ExecutionSpan]


@router.get("/{project_id}", response_model=GraphResponse)
async def get_graph(project_id: str, db: AsyncSession = Depends(get_db)):
    """Get graph data for a project."""
    with tracer.start_as_current_span("get_graph"):
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        graph_data = project.graph_data or {}
        
        return GraphResponse(
            projectId=project.id,
            name=project.name,
            chains=graph_data.get("chains", []),
            agents=graph_data.get("agents", []),
            activeRootId=graph_data.get("activeRootId", ""),
            nodes=graph_data.get("nodes", []),
            edges=graph_data.get("edges", []),
            metadata=project.metadata or {},
        )


@router.put("/{project_id}")
async def update_graph(
    project_id: str,
    payload: GraphUpdatePayload,
    db: AsyncSession = Depends(get_db),
):
    """Update graph data for a project."""
    with tracer.start_as_current_span("update_graph"):
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project.graph_data = {
            "nodes": [n.model_dump() for n in payload.nodes],
            "edges": [e.model_dump() for e in payload.edges],
            "chains": project.graph_data.get("chains", []),
            "agents": project.graph_data.get("agents", []),
            "activeRootId": project.graph_data.get("activeRootId", ""),
        }
        project.metadata = payload.metadata
        
        await db.commit()
        
        return {"status": "ok"}


@router.post("/{project_id}/execute", response_model=ExecutionResult)
async def execute_graph(
    project_id: str,
    request: ExecutionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Execute a graph with N3 runtime and OpenTelemetry instrumentation."""
    with tracer.start_as_current_span("execute_graph") as span:
        span.set_attribute("project_id", project_id)
        span.set_attribute("entry", request.entry)
        
        result_db = await db.execute(select(Project).where(Project.id == project_id))
        project = result_db.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Convert graph JSON back to N3 AST
        converter = N3ASTConverter()
        graph_data = project.graph_data or {}
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        try:
            # Convert graph to N3 Chain
            chain = converter.graph_json_to_chain(nodes, edges, request.entry)
            
            # Create execution context
            exec_context = ExecutionContext(
                project_id=project_id,
                entry_node=request.entry,
                input_data=request.input,
                options=request.options or {},
            )
            
            # Execute chain with instrumentation
            executor = GraphExecutor()
            result = await executor.execute_chain(chain, request.input, exec_context)
            
            # Convert execution spans to API response format
            trace_spans = [
                ExecutionSpan(
                    spanId=s.span_id,
                    parentSpanId=s.parent_span_id,
                    name=s.name,
                    type=s.type.value,
                    startTime=s.start_time.isoformat(),
                    endTime=s.end_time.isoformat(),
                    durationMs=s.duration_ms,
                    status=s.status,
                    attributes=SpanAttribute(
                        model=s.attributes.model,
                        temperature=s.attributes.temperature,
                        tokensPrompt=s.attributes.tokens_prompt,
                        tokensCompletion=s.attributes.tokens_completion,
                        cost=s.attributes.cost,
                    ),
                    input=s.input_data,
                    output=s.output_data,
                )
                for s in exec_context.spans
            ]
            
            span.set_status(Status(StatusCode.OK))
            
            return ExecutionResult(
                result=result,
                trace=trace_spans,
            )
            
        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=400, detail=f"Graph execution failed: {str(e)}")
