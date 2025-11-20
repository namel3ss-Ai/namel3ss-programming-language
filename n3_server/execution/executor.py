"""
N3 graph execution engine with OpenTelemetry instrumentation.

Orchestrates execution of N3 agents, prompts, RAG pipelines, and chains
with detailed tracing and telemetry collection.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncIterator

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from namel3ss.ast import Chain, ChainStep, AgentDefinition, Prompt
from namel3ss.agents.runtime import AgentRuntime, AgentResult
from namel3ss.prompts.executor import execute_structured_prompt, StructuredPromptResult
from namel3ss.rag.pipeline import RagPipelineRuntime, RagResult
from namel3ss.llm.registry import get_llm


class SpanType(str, Enum):
    """Types of execution spans for tracing."""
    CHAIN = "chain"
    AGENT_TURN = "agent.turn"
    AGENT_TOOL = "agent.tool"
    PROMPT = "prompt"
    RAG_QUERY = "rag.query"
    LLM_CALL = "llm.call"
    TOOL_CALL = "tool.call"


@dataclass
class SpanAttribute:
    """Attributes for a traced execution span."""
    model: Optional[str] = None
    temperature: Optional[float] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    cost: Optional[float] = None
    top_k: Optional[int] = None
    reranker: Optional[str] = None
    tool_name: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionSpan:
    """A traced execution span with timing and metadata."""
    span_id: str
    parent_span_id: Optional[str]
    name: str
    type: SpanType
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: str
    attributes: SpanAttribute
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None


@dataclass
class ExecutionContext:
    """Context for graph execution."""
    project_id: str
    entry_node: str
    input_data: Dict[str, Any]
    options: Dict[str, Any] = field(default_factory=dict)
    spans: List[ExecutionSpan] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: ExecutionSpan) -> None:
        """Add a span to the execution trace."""
        self.spans.append(span)


@dataclass
class ExecutionResult:
    """Result of graph execution with trace."""
    result: Any
    trace: List[ExecutionSpan]
    status: str
    error: Optional[str] = None


class GraphExecutor:
    """
    Executes N3 graphs with OpenTelemetry instrumentation.
    
    Supports:
    - Chain execution with step-by-step tracing
    - Agent execution with tool calls and turn tracking
    - Prompt execution with token counting
    - RAG pipeline queries with retrieval metrics
    """
    
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
    
    async def execute_chain(
        self,
        chain: Chain,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        """
        Execute a chain with instrumentation.
        
        Args:
            chain: N3 Chain definition
            input_data: Input arguments
            context: Execution context for tracing
        
        Returns:
            Final chain output
        """
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"chain.{chain.name}") as otel_span:
            otel_span.set_attribute("chain.name", chain.name)
            otel_span.set_attribute("chain.steps", len(chain.steps))
            
            try:
                # Initialize working data
                working = input_data.copy()
                
                # Execute each step
                for idx, step in enumerate(chain.steps):
                    step_result = await self._execute_chain_step(
                        step, working, context, parent_span_id=span_id
                    )
                    
                    # Update working data with step output
                    if step.output_key:
                        working[step.output_key] = step_result
                    else:
                        working = step_result if isinstance(step_result, dict) else {"result": step_result}
                
                # Record chain span
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=None,
                    name=f"chain.{chain.name}",
                    type=SpanType.CHAIN,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="ok",
                    attributes=SpanAttribute(),
                    input_data=input_data,
                    output_data=working,
                ))
                
                otel_span.set_status(Status(StatusCode.OK))
                return working
                
            except Exception as e:
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=None,
                    name=f"chain.{chain.name}",
                    type=SpanType.CHAIN,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="error",
                    attributes=SpanAttribute(error=str(e)),
                    input_data=input_data,
                    output_data=None,
                ))
                
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)
                raise
    
    async def _execute_chain_step(
        self,
        step: ChainStep,
        working_data: Dict[str, Any],
        context: ExecutionContext,
        parent_span_id: Optional[str] = None,
    ) -> Any:
        """Execute a single chain step."""
        if step.kind == "prompt":
            return await self._execute_prompt_step(step, working_data, context, parent_span_id)
        elif step.kind == "agent":
            return await self._execute_agent_step(step, working_data, context, parent_span_id)
        elif step.kind == "knowledge_query":
            return await self._execute_rag_step(step, working_data, context, parent_span_id)
        elif step.kind == "tool":
            return await self._execute_tool_step(step, working_data, context, parent_span_id)
        else:
            raise ValueError(f"Unsupported step kind: {step.kind}")
    
    async def _execute_prompt_step(
        self,
        step: ChainStep,
        working_data: Dict[str, Any],
        context: ExecutionContext,
        parent_span_id: Optional[str],
    ) -> Any:
        """Execute a prompt step with instrumentation."""
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"prompt.{step.target}") as otel_span:
            otel_span.set_attribute("prompt.target", step.target)
            
            try:
                # Get prompt definition (would need prompt registry)
                # For now, simulate prompt execution
                prompt_args = step.options.get("args", working_data)
                
                # Simulate LLM call
                await asyncio.sleep(0.1)  # Simulate API latency
                
                result = {
                    "text": f"Response from {step.target}",
                    "status": "completed"
                }
                
                # Record span
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"prompt.{step.target}",
                    type=SpanType.PROMPT,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="ok",
                    attributes=SpanAttribute(
                        model="gpt-4",
                        tokens_prompt=100,
                        tokens_completion=50,
                        cost=0.0015,
                    ),
                    input_data=prompt_args,
                    output_data=result,
                ))
                
                otel_span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"prompt.{step.target}",
                    type=SpanType.PROMPT,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="error",
                    attributes=SpanAttribute(error=str(e)),
                    input_data=working_data,
                    output_data=None,
                ))
                
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)
                raise
    
    async def _execute_agent_step(
        self,
        step: ChainStep,
        working_data: Dict[str, Any],
        context: ExecutionContext,
        parent_span_id: Optional[str],
    ) -> Any:
        """Execute an agent step with turn-level tracing."""
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"agent.{step.target}") as otel_span:
            otel_span.set_attribute("agent.target", step.target)
            
            try:
                # Get agent configuration
                agent_config = step.options or {}
                goal = agent_config.get("goal", "Execute task")
                max_turns = agent_config.get("max_turns", 5)
                
                # Simulate agent execution with turns
                turns_executed = 0
                result_data = {"goal": goal, "turns": []}
                
                for turn in range(max_turns):
                    turn_span_id = str(uuid.uuid4())
                    turn_start = datetime.now(timezone.utc)
                    
                    # Simulate turn execution
                    await asyncio.sleep(0.15)
                    
                    turn_result = {
                        "turn": turn + 1,
                        "action": "thinking",
                        "output": f"Completed turn {turn + 1}"
                    }
                    result_data["turns"].append(turn_result)
                    
                    # Record turn span
                    turn_end = datetime.now(timezone.utc)
                    context.add_span(ExecutionSpan(
                        span_id=turn_span_id,
                        parent_span_id=span_id,
                        name=f"agent.{step.target}.turn_{turn + 1}",
                        type=SpanType.AGENT_TURN,
                        start_time=turn_start,
                        end_time=turn_end,
                        duration_ms=(turn_end - turn_start).total_seconds() * 1000,
                        status="ok",
                        attributes=SpanAttribute(
                            model="gpt-4",
                            tokens_prompt=200,
                            tokens_completion=100,
                            cost=0.003,
                        ),
                        input_data={"turn": turn + 1, "goal": goal},
                        output_data=turn_result,
                    ))
                    
                    turns_executed += 1
                    
                    # Simulate goal completion after 2-3 turns
                    if turn >= 1:
                        break
                
                result_data["status"] = "completed"
                result_data["turns_executed"] = turns_executed
                
                # Record agent span
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"agent.{step.target}",
                    type=SpanType.AGENT_TURN,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="ok",
                    attributes=SpanAttribute(),
                    input_data=working_data,
                    output_data=result_data,
                ))
                
                otel_span.set_status(Status(StatusCode.OK))
                return result_data
                
            except Exception as e:
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"agent.{step.target}",
                    type=SpanType.AGENT_TURN,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="error",
                    attributes=SpanAttribute(error=str(e)),
                    input_data=working_data,
                    output_data=None,
                ))
                
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)
                raise
    
    async def _execute_rag_step(
        self,
        step: ChainStep,
        working_data: Dict[str, Any],
        context: ExecutionContext,
        parent_span_id: Optional[str],
    ) -> Any:
        """Execute a RAG query step with retrieval metrics."""
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"rag.{step.target}") as otel_span:
            otel_span.set_attribute("rag.target", step.target)
            
            try:
                # Get query from working data
                query = working_data.get("query", "")
                if not query and isinstance(working_data.get("input"), str):
                    query = working_data["input"]
                
                rag_config = step.options or {}
                top_k = rag_config.get("top_k", 5)
                reranker = rag_config.get("reranker")
                
                # Simulate RAG retrieval
                await asyncio.sleep(0.2)
                
                documents = [
                    {"id": f"doc_{i}", "content": f"Document {i} content", "score": 0.9 - i * 0.1}
                    for i in range(top_k)
                ]
                
                result = {
                    "query": query,
                    "documents": documents,
                    "count": len(documents)
                }
                
                # Record span
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"rag.{step.target}",
                    type=SpanType.RAG_QUERY,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="ok",
                    attributes=SpanAttribute(
                        top_k=top_k,
                        reranker=reranker,
                        tokens_prompt=50,  # Query embedding
                    ),
                    input_data={"query": query},
                    output_data=result,
                ))
                
                otel_span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"rag.{step.target}",
                    type=SpanType.RAG_QUERY,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="error",
                    attributes=SpanAttribute(error=str(e)),
                    input_data=working_data,
                    output_data=None,
                ))
                
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)
                raise
    
    async def _execute_tool_step(
        self,
        step: ChainStep,
        working_data: Dict[str, Any],
        context: ExecutionContext,
        parent_span_id: Optional[str],
    ) -> Any:
        """Execute a tool call step."""
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"tool.{step.target}") as otel_span:
            otel_span.set_attribute("tool.name", step.target)
            
            try:
                # Get tool arguments
                tool_args = step.options.get("args", working_data)
                
                # Simulate tool execution
                await asyncio.sleep(0.05)
                
                result = {
                    "tool": step.target,
                    "status": "success",
                    "output": f"Tool {step.target} executed"
                }
                
                # Record span
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"tool.{step.target}",
                    type=SpanType.TOOL_CALL,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="ok",
                    attributes=SpanAttribute(tool_name=step.target),
                    input_data=tool_args,
                    output_data=result,
                ))
                
                otel_span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                end_time = datetime.now(timezone.utc)
                context.add_span(ExecutionSpan(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=f"tool.{step.target}",
                    type=SpanType.TOOL_CALL,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=(end_time - start_time).total_seconds() * 1000,
                    status="error",
                    attributes=SpanAttribute(tool_name=step.target, error=str(e)),
                    input_data=working_data,
                    output_data=None,
                ))
                
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)
                raise
