"""
N3 graph execution engine with OpenTelemetry instrumentation.

Orchestrates execution of N3 agents, prompts, RAG pipelines, and chains
with detailed tracing and telemetry collection.
"""

from __future__ import annotations

import asyncio
import logging
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
from n3_server.execution.registry import RuntimeRegistry, RegistryError

logger = logging.getLogger(__name__)


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
    
    This executor uses real runtime components (AgentRuntime, PromptExecutor,
    RagPipelineRuntime) provided via the RuntimeRegistry.
    """
    
    def __init__(self, registry: RuntimeRegistry):
        """
        Initialize executor with runtime registry.
        
        Args:
            registry: RuntimeRegistry with instantiated components
        """
        self.registry = registry
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
                    # ChainStep doesn't have output_key, always merge result
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
        """
        Execute a prompt step with real PromptExecutor.
        
        Uses execute_structured_prompt from namel3ss.prompts.executor to
        execute the prompt with validation and instrumentation.
        """
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"prompt.{step.target}") as otel_span:
            otel_span.set_attribute("prompt.target", step.target)
            
            try:
                # Get prompt definition from registry
                prompt_def = self.registry.get_prompt(step.target)
                otel_span.set_attribute("prompt.name", prompt_def.name)
                
                # Get LLM for prompt
                llm = self.registry.get_llm(prompt_def.model)
                otel_span.set_attribute("prompt.model", prompt_def.model)
                
                # Prepare arguments (merge step options with working data)
                prompt_args = step.options.get("args", {})
                prompt_args.update(working_data)
                
                # Execute with real prompt executor
                logger.info(
                    f"Executing prompt '{step.target}' with model '{prompt_def.model}'"
                )
                result = await execute_structured_prompt(
                    prompt_def=prompt_def,
                    llm=llm,
                    args=prompt_args,
                    retry_on_validation_error=True,
                    max_retries=2,
                )
                
                # Record span with real metrics
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
                        model=result.model,
                        tokens_prompt=result.prompt_tokens,
                        tokens_completion=result.completion_tokens,
                        cost=self._estimate_cost(
                            result.model,
                            result.prompt_tokens,
                            result.completion_tokens
                        ),
                    ),
                    input_data=prompt_args,
                    output_data=result.output,
                ))
                
                otel_span.set_attribute("prompt.tokens_prompt", result.prompt_tokens)
                otel_span.set_attribute("prompt.tokens_completion", result.completion_tokens)
                otel_span.set_attribute("prompt.latency_ms", result.latency_ms)
                otel_span.set_status(Status(StatusCode.OK))
                
                logger.info(
                    f"Prompt '{step.target}' completed: "
                    f"{result.prompt_tokens}+{result.completion_tokens} tokens, "
                    f"{result.latency_ms:.1f}ms"
                )
                
                return result.output
                
            except RegistryError as e:
                logger.error(f"Registry error executing prompt '{step.target}': {e}")
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
            
            except Exception as e:
                logger.error(f"Error executing prompt '{step.target}': {e}")
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
        """
        Execute an agent step with real AgentRuntime.
        
        Uses AgentRuntime from registry to execute the agent with turn-level
        tracing and token counting.
        """
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"agent.{step.target}") as otel_span:
            otel_span.set_attribute("agent.target", step.target)
            
            try:
                # Get agent runtime from registry
                agent_runtime = self.registry.get_agent(step.target)
                otel_span.set_attribute("agent.name", agent_runtime.name)
                
                # Get agent configuration
                agent_config = step.options or {}
                user_input = agent_config.get("input") or working_data.get("input", "")
                goal = agent_config.get("goal")
                max_turns = agent_config.get("max_turns", agent_runtime.max_turns)
                
                logger.info(
                    f"Executing agent '{step.target}' with max_turns={max_turns}"
                )
                
                # Execute with real agent runtime
                result: AgentResult = await agent_runtime.execute(
                    user_input=user_input,
                    goal=goal,
                    max_turns=max_turns,
                )
                
                # Record turn-level spans
                total_tokens_prompt = 0
                total_tokens_completion = 0
                
                for turn_idx, turn in enumerate(result.turns, start=1):
                    turn_span_id = str(uuid.uuid4())
                    
                    # Estimate turn timing (AgentResult doesn't have duration_ms, use metadata or default)
                    total_duration_ms = result.metadata.get('duration_ms', 100.0 * len(result.turns))
                    turn_duration_ms = total_duration_ms / len(result.turns)
                    turn_start = start_time
                    turn_end = datetime.fromtimestamp(
                        start_time.timestamp() + (turn_idx * turn_duration_ms / 1000),
                        tz=timezone.utc
                    )
                    
                    # Count tokens for turn
                    from namel3ss.agents.runtime import estimate_messages_tokens
                    turn_prompt_tokens = estimate_messages_tokens(turn.messages)
                    turn_completion_tokens = turn_prompt_tokens  # Estimate both from messages
                    total_tokens_prompt += turn_prompt_tokens
                    total_tokens_completion += turn_completion_tokens
                    
                    # Extract user and assistant messages
                    user_msg = next((m for m in turn.messages if m.role == "user"), None)
                    assistant_msg = next((m for m in turn.messages if m.role == "assistant"), None)
                    
                    context.add_span(ExecutionSpan(
                        span_id=turn_span_id,
                        parent_span_id=span_id,
                        name=f"agent.{step.target}.turn_{turn_idx}",
                        type=SpanType.AGENT_TURN,
                        start_time=turn_start,
                        end_time=turn_end,
                        duration_ms=turn_duration_ms,
                        status="ok",
                        attributes=SpanAttribute(
                            model=agent_runtime.llm.model_name,
                            tokens_prompt=turn_prompt_tokens,
                            tokens_completion=turn_completion_tokens,
                            cost=self._estimate_cost(
                                agent_runtime.llm.model_name,
                                turn_prompt_tokens,
                                turn_completion_tokens
                            ),
                        ),
                        input_data={
                            "turn": turn_idx,
                            "role": user_msg.role if user_msg else "user",
                            "content": user_msg.content if user_msg else "",
                        },
                        output_data={
                            "role": assistant_msg.role if assistant_msg else "assistant",
                            "content": assistant_msg.content if assistant_msg else "",
                            "tool_calls": len(turn.tool_calls),
                        },
                    ))
                
                # Prepare result data
                result_data = {
                    "status": result.status,
                    "turns_executed": len(result.turns),
                    "final_response": result.final_response,
                    "error": result.error,
                }
                
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
                    status="ok" if result.status == "completed" else "error",
                    attributes=SpanAttribute(
                        model=agent_runtime.llm.model_name,
                        tokens_prompt=total_tokens_prompt,
                        tokens_completion=total_tokens_completion,
                        cost=self._estimate_cost(
                            agent_runtime.llm.model_name,
                            total_tokens_prompt,
                            total_tokens_completion
                        ),
                    ),
                    input_data={"input": user_input, "goal": goal},
                    output_data=result_data,
                ))
                
                otel_span.set_attribute("agent.turns", len(result.turns))
                otel_span.set_attribute("agent.tokens_prompt", total_tokens_prompt)
                otel_span.set_attribute("agent.tokens_completion", total_tokens_completion)
                otel_span.set_status(Status(StatusCode.OK))
                
                # Calculate total duration from metadata or estimate
                duration_ms = result.metadata.get('duration_ms', total_duration_ms)
                logger.info(
                    f"Agent '{step.target}' completed: {len(result.turns)} turns, "
                    f"{total_tokens_prompt}+{total_tokens_completion} tokens, "
                    f"{duration_ms:.1f}ms"
                )
                
                return result_data
                
            except RegistryError as e:
                logger.error(f"Registry error executing agent '{step.target}': {e}")
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
                
            except Exception as e:
                logger.error(f"Error executing agent '{step.target}': {e}")
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
        """
        Execute a RAG query step with real RagPipelineRuntime.
        
        Uses RagPipelineRuntime from registry to execute retrieval with
        embeddings, vector search, and optional reranking.
        """
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_as_current_span(f"rag.{step.target}") as otel_span:
            otel_span.set_attribute("rag.target", step.target)
            
            try:
                # Get RAG pipeline from registry
                rag_pipeline = self.registry.get_rag_pipeline(step.target)
                otel_span.set_attribute("rag.name", rag_pipeline.name)
                otel_span.set_attribute("rag.encoder", rag_pipeline.query_encoder)
                
                # Get query from working data
                query = working_data.get("query", "")
                if not query and isinstance(working_data.get("input"), str):
                    query = working_data["input"]
                
                if not query:
                    raise ValueError("No query provided in working data")
                
                # Get RAG configuration
                rag_config = step.options or {}
                top_k = rag_config.get("top_k", rag_pipeline.top_k)
                
                logger.info(
                    f"Executing RAG query on '{step.target}' with top_k={top_k}"
                )
                
                # Execute query with real RAG pipeline
                rag_result: RagResult = await rag_pipeline.execute_query(
                    query=query,
                    top_k=top_k,
                )
                
                # Convert documents to dict format
                documents = [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "score": doc.score,
                        "metadata": doc.metadata,
                    }
                    for doc in rag_result.documents
                ]
                
                result = {
                    "query": rag_result.query,
                    "documents": documents,
                    "count": len(documents),
                    "metadata": rag_result.metadata,
                }
                
                # Record span with metrics
                end_time = datetime.now(timezone.utc)
                retrieval_time_ms = rag_result.metadata.get("retrieval_time_ms", 0)
                rerank_time_ms = rag_result.metadata.get("rerank_time_ms", 0)
                
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
                        reranker=rag_pipeline.reranker,
                        model=rag_pipeline.query_encoder,
                    ),
                    input_data={"query": query, "top_k": top_k},
                    output_data=result,
                ))
                
                otel_span.set_attribute("rag.documents_retrieved", len(documents))
                otel_span.set_attribute("rag.retrieval_time_ms", retrieval_time_ms)
                if rerank_time_ms:
                    otel_span.set_attribute("rag.rerank_time_ms", rerank_time_ms)
                otel_span.set_status(Status(StatusCode.OK))
                
                logger.info(
                    f"RAG query on '{step.target}' completed: {len(documents)} docs, "
                    f"retrieval={retrieval_time_ms:.1f}ms"
                )
                
                return result
                
            except RegistryError as e:
                logger.error(f"Registry error executing RAG '{step.target}': {e}")
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
                
            except Exception as e:
                logger.error(f"Error executing RAG '{step.target}': {e}")
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
    
    def _estimate_cost(self, model: str, tokens_prompt: int, tokens_completion: int) -> float:
        """
        Estimate API cost based on token usage and model pricing.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            tokens_prompt: Number of prompt tokens
            tokens_completion: Number of completion tokens
        
        Returns:
            Estimated cost in USD
        """
        # Pricing per 1K tokens (as of 2024)
        PRICING = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
        }
        
        # Normalize model name
        model_lower = model.lower()
        for pricing_key in PRICING:
            if pricing_key in model_lower:
                pricing = PRICING[pricing_key]
                cost_prompt = (tokens_prompt / 1000) * pricing["prompt"]
                cost_completion = (tokens_completion / 1000) * pricing["completion"]
                return cost_prompt + cost_completion
        
        # Default pricing for unknown models (use GPT-4 as baseline)
        logger.warning(f"Unknown model '{model}', using default pricing")
        cost_prompt = (tokens_prompt / 1000) * 0.03
        cost_completion = (tokens_completion / 1000) * 0.06
        return cost_prompt + cost_completion
