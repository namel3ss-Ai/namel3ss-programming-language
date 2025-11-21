"""Tests for runtime registry and graph executor integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from n3_server.execution.registry import RuntimeRegistry, RegistryError
from n3_server.execution.executor import GraphExecutor, ExecutionContext, SpanType
from n3_server.converter.enhanced_converter import EnhancedN3ASTConverter, ConversionContext
from n3_server.converter.models import GraphJSON
from namel3ss.ast import AgentDefinition, Prompt, Chain, ChainStep
from namel3ss.ast.rag import RagPipelineDefinition


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM provider."""
    llm = MagicMock()
    llm.model_name = "gpt-4"
    llm.generate = AsyncMock(return_value="Mock response")
    return llm


@pytest.fixture
def mock_agent_runtime():
    """Mock AgentRuntime."""
    from namel3ss.agents.runtime import AgentResult, AgentTurn, AgentMessage
    
    runtime = MagicMock()
    runtime.name = "test_agent"
    runtime.llm = MagicMock()
    runtime.llm.model_name = "gpt-4"
    runtime.max_turns = 5
    
    # Create mock result with correct AgentTurn structure
    mock_result = AgentResult(
        status="success",
        final_response="Test output",
        turns=[
            AgentTurn(
                messages=[
                    AgentMessage(role="user", content="Test input"),
                    AgentMessage(role="assistant", content="Test output"),
                ],
                tool_calls=[],
                tool_results=[],
            )
        ],
        error=None,
    )
    runtime.execute = AsyncMock(return_value=mock_result)
    
    return runtime


@pytest.fixture
def mock_rag_pipeline():
    """Mock RagPipelineRuntime."""
    from namel3ss.rag.backends import ScoredDocument
    from namel3ss.rag.pipeline import RagResult
    
    pipeline = MagicMock()
    pipeline.name = "test_rag"
    pipeline.query_encoder = "text-embedding-ada-002"
    pipeline.top_k = 5
    pipeline.reranker = None
    
    # Create mock result
    mock_result = RagResult(
        documents=[
            ScoredDocument(
                id="doc1",
                content="Document 1 content",
                score=0.95,
                metadata={}
            ),
            ScoredDocument(
                id="doc2",
                content="Document 2 content",
                score=0.85,
                metadata={}
            ),
        ],
        query="test query",
        metadata={"retrieval_time_ms": 50.0}
    )
    pipeline.execute_query = AsyncMock(return_value=mock_result)
    
    return pipeline


@pytest.fixture
def conversion_context_with_agent():
    """ConversionContext with an agent."""
    context = ConversionContext(
        project_id="test",
        graph_name="test",
        visited_nodes=set(),
        agent_registry={
            "test_agent": AgentDefinition(
                name="test_agent",
                llm_name="gpt-4",
                system_prompt="You are a test agent.",
                tool_names=["search"],
                max_turns=5
            )
        },
        prompt_registry={},
        rag_registry={},
        tool_registry={},
    )
    return context


@pytest.fixture
def conversion_context_with_prompt():
    """ConversionContext with a prompt."""
    from namel3ss.ast.ai.prompts import PromptArgument
    
    context = ConversionContext(
        project_id="test",
        graph_name="test",
        visited_nodes=set(),
        agent_registry={},
        prompt_registry={
            "test_prompt": Prompt(
                name="test_prompt",
                model="gpt-4",
                template="Say {{input}}",
                args=[PromptArgument(name="input", arg_type="string", required=True)],
                output_schema={"result": "string"},
            )
        },
        rag_registry={},
        tool_registry={},
    )
    return context


@pytest.fixture
def conversion_context_with_rag():
    """ConversionContext with RAG dataset."""
    context = ConversionContext(
        project_id="test",
        graph_name="test",
        visited_nodes=set(),
        agent_registry={},
        prompt_registry={},
        rag_registry={
            "test_rag": RagPipelineDefinition(
                name="test_rag",
                query_encoder="text-embedding-ada-002",
                index="test_index",
                top_k=5,
                distance_metric="cosine",
            )
        },
        tool_registry={},
    )
    return context


# ============================================================================
# RuntimeRegistry Tests
# ============================================================================

class TestRuntimeRegistry:
    """Tests for RuntimeRegistry."""
    
    @pytest.mark.asyncio
    async def test_build_registry_with_agent(
        self, conversion_context_with_agent, mock_llm
    ):
        """Test building registry from context with agent."""
        llm_registry = {"gpt-4": mock_llm}
        tool_registry = {"search": lambda x: "search result"}
        
        with patch("n3_server.execution.registry.AgentRuntime") as MockAgentRuntime:
            MockAgentRuntime.return_value = MagicMock()
            
            registry = await RuntimeRegistry.from_conversion_context(
                context=conversion_context_with_agent,
                llm_registry=llm_registry,
                tool_registry=tool_registry,
            )
            
            assert "test_agent" in registry.agents
            assert len(registry.llms) == 1
            assert len(registry.tools) == 1
    
    @pytest.mark.asyncio
    async def test_build_registry_with_prompt(
        self, conversion_context_with_prompt, mock_llm
    ):
        """Test building registry from context with prompt."""
        llm_registry = {"gpt-4": mock_llm}
        
        registry = await RuntimeRegistry.from_conversion_context(
            context=conversion_context_with_prompt,
            llm_registry=llm_registry,
            tool_registry={},
        )
        
        assert "test_prompt" in registry.prompts
        assert registry.prompts["test_prompt"].name == "test_prompt"
    
    @pytest.mark.asyncio
    async def test_registry_missing_llm(self, conversion_context_with_agent):
        """Test that missing LLM raises RegistryError."""
        # Empty LLM registry
        llm_registry = {}
        
        with pytest.raises(RegistryError, match="not found in LLM registry"):
            await RuntimeRegistry.from_conversion_context(
                context=conversion_context_with_agent,
                llm_registry=llm_registry,
                tool_registry={},
            )
    
    def test_get_agent_not_found(self):
        """Test that getting non-existent agent raises error."""
        registry = RuntimeRegistry()
        
        with pytest.raises(RegistryError, match="Agent.*not found"):
            registry.get_agent("nonexistent")
    
    def test_get_prompt_not_found(self):
        """Test that getting non-existent prompt raises error."""
        registry = RuntimeRegistry()
        
        with pytest.raises(RegistryError, match="Prompt.*not found"):
            registry.get_prompt("nonexistent")
    
    def test_get_rag_pipeline_not_found(self):
        """Test that getting non-existent RAG pipeline raises error."""
        registry = RuntimeRegistry()
        
        with pytest.raises(RegistryError, match="RAG pipeline.*not found"):
            registry.get_rag_pipeline("nonexistent")


# ============================================================================
# GraphExecutor Tests
# ============================================================================

class TestGraphExecutor:
    """Tests for GraphExecutor with mocked runtime components."""
    
    @pytest.mark.asyncio
    async def test_execute_prompt_step(self, mock_llm):
        """Test executing a prompt step."""
        from namel3ss.prompts.executor import StructuredPromptResult
        
        # Setup registry
        from namel3ss.ast.ai.prompts import PromptArgument
        
        registry = RuntimeRegistry(
            prompts={
                "test_prompt": Prompt(
                    name="test_prompt",
                    model="gpt-4",
                    template="Say {{input}}",
                    args=[PromptArgument(name="input", arg_type="string", required=True)],
                    output_schema={"result": "string"},
                )
            },
            llms={"gpt-4": mock_llm},
        )
        
        # Create executor
        executor = GraphExecutor(registry=registry)
        
        # Create chain step
        step = ChainStep(
            kind="prompt",
            target="test_prompt",
            options={"args": {"input": "hello"}},
        )
        
        # Create execution context
        context = ExecutionContext(
            project_id="test",
            entry_node="start",
            input_data={"input": "hello"},
        )
        
        # Mock execute_structured_prompt
        mock_result = StructuredPromptResult(
            output={"result": "hello"},
            raw_response="hello",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4",
        )
        
        with patch(
            "n3_server.execution.executor.execute_structured_prompt",
            return_value=mock_result
        ):
            result = await executor._execute_prompt_step(
                step, {"input": "hello"}, context, None
            )
            
            assert result == {"result": "hello"}
            assert len(context.spans) == 1
            assert context.spans[0].type == SpanType.PROMPT
            assert context.spans[0].status == "ok"
            assert context.spans[0].attributes.tokens_prompt == 10
            assert context.spans[0].attributes.tokens_completion == 5
    
    @pytest.mark.asyncio
    async def test_execute_agent_step(self, mock_agent_runtime):
        """Test executing an agent step."""
        # Setup registry
        registry = RuntimeRegistry(
            agents={"test_agent": mock_agent_runtime},
        )
        
        # Create executor
        executor = GraphExecutor(registry=registry)
        
        # Create chain step
        step = ChainStep(
            kind="agent",
            target="test_agent",
            options={"input": "test", "max_turns": 3},
        )
        
        # Create execution context
        context = ExecutionContext(
            project_id="test",
            entry_node="start",
            input_data={"input": "test"},
        )
        
        # Execute
        result = await executor._execute_agent_step(
            step, {"input": "test"}, context, None
        )
        
        assert result["status"] == "success"  # AgentResult uses "success" not "completed"
        assert result["turns_executed"] == 1
        assert len(context.spans) >= 1  # At least agent span
    
    @pytest.mark.asyncio
    async def test_execute_rag_step(self, mock_rag_pipeline):
        """Test executing a RAG step."""
        # Setup registry
        registry = RuntimeRegistry(
            rag_pipelines={"test_rag": mock_rag_pipeline},
        )
        
        # Create executor
        executor = GraphExecutor(registry=registry)
        
        # Create chain step
        step = ChainStep(
            kind="knowledge_query",
            target="test_rag",
            options={"top_k": 5},
        )
        
        # Create execution context
        context = ExecutionContext(
            project_id="test",
            entry_node="start",
            input_data={"query": "test query"},
        )
        
        # Execute
        result = await executor._execute_rag_step(
            step, {"query": "test query"}, context, None
        )
        
        assert result["query"] == "test query"
        assert len(result["documents"]) == 2
        assert result["documents"][0]["id"] == "doc1"
        assert len(context.spans) == 1
        assert context.spans[0].type == SpanType.RAG_QUERY
    
    @pytest.mark.asyncio
    async def test_execute_chain(self, mock_llm):
        """Test executing a full chain."""
        from namel3ss.prompts.executor import StructuredPromptResult
        
        # Setup registry
        from namel3ss.ast.ai.prompts import PromptArgument
        
        registry = RuntimeRegistry(
            prompts={
                "greeting": Prompt(
                    name="greeting",
                    model="gpt-4",
                    template="Say hello to {{name}}",
                    args=[PromptArgument(name="name", arg_type="string", required=True)],
                    output_schema={"greeting": "string"},
                )
            },
            llms={"gpt-4": mock_llm},
        )
        
        # Create chain
        chain = Chain(
            name="test_chain",
            steps=[
                ChainStep(
                    kind="prompt",
                    target="greeting",
                    options={},
                    name="greeting_step",
                )
            ],
        )
        
        # Create executor
        executor = GraphExecutor(registry=registry)
        
        # Create execution context
        context = ExecutionContext(
            project_id="test",
            entry_node="start",
            input_data={"name": "Alice"},
        )
        
        # Mock execute_structured_prompt
        mock_result = StructuredPromptResult(
            output={"greeting": "Hello Alice"},
            raw_response="Hello Alice",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-4",
        )
        
        with patch(
            "n3_server.execution.executor.execute_structured_prompt",
            return_value=mock_result
        ):
            result = await executor.execute_chain(
                chain, {"name": "Alice"}, context
            )
            
            # Chain returns merged output, step result is the greeting dict
            assert result is not None
            assert "greeting" in result
            assert result["greeting"] == "Hello Alice"
            assert len(context.spans) >= 2  # Chain span + prompt span
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        registry = RuntimeRegistry()
        executor = GraphExecutor(registry=registry)
        
        # GPT-4 pricing
        cost = executor._estimate_cost("gpt-4", 1000, 500)
        assert cost == pytest.approx(0.03 + 0.03, rel=0.01)  # 1K prompt + 500 completion
        
        # GPT-3.5 pricing
        cost = executor._estimate_cost("gpt-3.5-turbo", 1000, 500)
        assert cost == pytest.approx(0.0015 + 0.001, rel=0.01)
        
        # Unknown model (uses GPT-4 pricing)
        cost = executor._estimate_cost("unknown-model", 1000, 500)
        assert cost == pytest.approx(0.03 + 0.03, rel=0.01)


class TestExecutionContext:
    """Tests for ExecutionContext."""
    
    def test_add_span(self):
        """Test adding spans to context."""
        from n3_server.execution.executor import ExecutionSpan, SpanAttribute
        
        context = ExecutionContext(
            project_id="test",
            entry_node="start",
            input_data={},
        )
        
        span = ExecutionSpan(
            span_id="span-1",
            parent_span_id=None,
            name="test_span",
            type=SpanType.CHAIN,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_ms=100.0,
            status="ok",
            attributes=SpanAttribute(),
        )
        
        context.add_span(span)
        
        assert len(context.spans) == 1
        assert context.spans[0].span_id == "span-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
