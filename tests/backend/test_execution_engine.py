"""
Tests for N3 execution engine with OpenTelemetry tracing.
"""

import pytest
from datetime import datetime, timezone

from namel3ss.ast import Chain, ChainStep
from n3_server.execution import (
    GraphExecutor,
    ExecutionContext,
    ExecutionResult,
    SpanType,
)


@pytest.mark.asyncio
async def test_execute_simple_chain():
    """Test executing a simple chain with prompt steps."""
    # Create a simple chain
    chain = Chain(
        name="test_chain",
        steps=[
            ChainStep(
                kind="prompt",
                target="greeting_prompt",
                options={"args": {"name": "Alice"}},
                output_key="greeting",
            ),
            ChainStep(
                kind="prompt",
                target="summary_prompt",
                options={"args": {"text": "Hello"}},
                output_key="summary",
            ),
        ],
        input_key="input",
        output_key="summary",
    )
    
    # Create execution context
    context = ExecutionContext(
        project_id="test-project",
        entry_node="start",
        input_data={"name": "Alice", "text": "Hello world"},
    )
    
    # Execute
    executor = GraphExecutor()
    result = await executor.execute_chain(chain, context.input_data, context)
    
    # Verify result
    assert result is not None
    assert "greeting" in result or "summary" in result
    
    # Verify trace spans
    assert len(context.spans) >= 3  # Chain + 2 prompts
    
    # Check chain span
    chain_spans = [s for s in context.spans if s.type == SpanType.CHAIN]
    assert len(chain_spans) == 1
    assert chain_spans[0].status == "ok"
    
    # Check prompt spans
    prompt_spans = [s for s in context.spans if s.type == SpanType.PROMPT]
    assert len(prompt_spans) == 2
    assert all(s.status == "ok" for s in prompt_spans)
    assert all(s.attributes.tokens_prompt is not None for s in prompt_spans)
    assert all(s.attributes.tokens_completion is not None for s in prompt_spans)


@pytest.mark.asyncio
async def test_execute_agent_step():
    """Test executing a chain with an agent step."""
    chain = Chain(
        name="agent_chain",
        steps=[
            ChainStep(
                kind="agent",
                target="research_agent",
                options={
                    "goal": "Research topic",
                    "max_turns": 3,
                },
                output_key="research_result",
            ),
        ],
        input_key="input",
        output_key="research_result",
    )
    
    context = ExecutionContext(
        project_id="test-project",
        entry_node="start",
        input_data={"query": "AI trends"},
    )
    
    executor = GraphExecutor()
    result = await executor.execute_chain(chain, context.input_data, context)
    
    # Verify result
    assert result is not None
    
    # Verify agent spans
    agent_spans = [s for s in context.spans if s.type == SpanType.AGENT_TURN]
    assert len(agent_spans) >= 2  # Parent + turns
    
    # Check turn spans have token counts
    turn_spans = [s for s in agent_spans if "turn_" in s.name]
    assert len(turn_spans) >= 1
    assert all(s.attributes.tokens_prompt is not None for s in turn_spans)


@pytest.mark.asyncio
async def test_execute_rag_step():
    """Test executing a RAG query step."""
    chain = Chain(
        name="rag_chain",
        steps=[
            ChainStep(
                kind="knowledge_query",
                target="docs_index",
                options={
                    "top_k": 5,
                    "reranker": "cohere",
                },
                output_key="documents",
            ),
        ],
        input_key="input",
        output_key="documents",
    )
    
    context = ExecutionContext(
        project_id="test-project",
        entry_node="start",
        input_data={"query": "What is RAG?"},
    )
    
    executor = GraphExecutor()
    result = await executor.execute_chain(chain, context.input_data, context)
    
    # Verify result
    assert result is not None
    
    # Verify RAG spans
    rag_spans = [s for s in context.spans if s.type == SpanType.RAG_QUERY]
    assert len(rag_spans) == 1
    assert rag_spans[0].status == "ok"
    assert rag_spans[0].attributes.top_k == 5
    assert rag_spans[0].attributes.reranker == "cohere"


@pytest.mark.asyncio
async def test_execute_tool_step():
    """Test executing a tool call step."""
    chain = Chain(
        name="tool_chain",
        steps=[
            ChainStep(
                kind="tool",
                target="calculator",
                options={"args": {"operation": "add", "a": 5, "b": 3}},
                output_key="result",
            ),
        ],
        input_key="input",
        output_key="result",
    )
    
    context = ExecutionContext(
        project_id="test-project",
        entry_node="start",
        input_data={"operation": "add"},
    )
    
    executor = GraphExecutor()
    result = await executor.execute_chain(chain, context.input_data, context)
    
    # Verify result
    assert result is not None
    
    # Verify tool spans
    tool_spans = [s for s in context.spans if s.type == SpanType.TOOL_CALL]
    assert len(tool_spans) == 1
    assert tool_spans[0].status == "ok"
    assert tool_spans[0].attributes.tool_name == "calculator"


@pytest.mark.asyncio
async def test_multi_step_chain_with_tracing():
    """Test a complex multi-step chain with full tracing."""
    chain = Chain(
        name="complex_chain",
        steps=[
            ChainStep(
                kind="knowledge_query",
                target="docs_index",
                options={"top_k": 3},
                output_key="docs",
            ),
            ChainStep(
                kind="prompt",
                target="summarize_prompt",
                options={"args": {"documents": []}},
                output_key="summary",
            ),
            ChainStep(
                kind="agent",
                target="review_agent",
                options={"goal": "Review summary", "max_turns": 2},
                output_key="review",
            ),
        ],
        input_key="input",
        output_key="review",
    )
    
    context = ExecutionContext(
        project_id="test-project",
        entry_node="start",
        input_data={"query": "Explain transformers"},
    )
    
    executor = GraphExecutor()
    result = await executor.execute_chain(chain, context.input_data, context)
    
    # Verify result
    assert result is not None
    
    # Verify comprehensive tracing
    assert len(context.spans) >= 5  # Chain + RAG + Prompt + Agent + turns
    
    # Check span hierarchy
    chain_span = [s for s in context.spans if s.type == SpanType.CHAIN][0]
    child_spans = [s for s in context.spans if s.parent_span_id == chain_span.span_id]
    assert len(child_spans) >= 3
    
    # Verify timing
    total_duration = sum(s.duration_ms for s in context.spans if s.parent_span_id is None)
    assert total_duration > 0


@pytest.mark.asyncio
async def test_execution_context_variables():
    """Test that execution context maintains variables across steps."""
    chain = Chain(
        name="var_chain",
        steps=[
            ChainStep(
                kind="prompt",
                target="step1",
                options={},
                output_key="var1",
            ),
            ChainStep(
                kind="prompt",
                target="step2",
                options={},
                output_key="var2",
            ),
        ],
        input_key="input",
        output_key="var2",
    )
    
    context = ExecutionContext(
        project_id="test-project",
        entry_node="start",
        input_data={"initial": "data"},
    )
    context.variables["custom_var"] = "custom_value"
    
    executor = GraphExecutor()
    result = await executor.execute_chain(chain, context.input_data, context)
    
    # Verify context preserved
    assert "custom_var" in context.variables
    assert context.variables["custom_var"] == "custom_value"


@pytest.mark.asyncio
async def test_span_attributes_completeness():
    """Test that spans capture all required attributes."""
    chain = Chain(
        name="attr_chain",
        steps=[
            ChainStep(
                kind="prompt",
                target="test_prompt",
                options={},
                output_key="result",
            ),
        ],
        input_key="input",
        output_key="result",
    )
    
    context = ExecutionContext(
        project_id="test-project",
        entry_node="start",
        input_data={"test": "data"},
    )
    
    executor = GraphExecutor()
    await executor.execute_chain(chain, context.input_data, context)
    
    # Check span attributes
    prompt_span = [s for s in context.spans if s.type == SpanType.PROMPT][0]
    
    # Verify timing
    assert isinstance(prompt_span.start_time, datetime)
    assert isinstance(prompt_span.end_time, datetime)
    assert prompt_span.duration_ms > 0
    assert prompt_span.end_time > prompt_span.start_time
    
    # Verify attributes
    assert prompt_span.attributes.model is not None
    assert prompt_span.attributes.tokens_prompt is not None
    assert prompt_span.attributes.tokens_completion is not None
    assert prompt_span.attributes.cost is not None
    
    # Verify data
    assert prompt_span.input_data is not None
    assert prompt_span.output_data is not None
