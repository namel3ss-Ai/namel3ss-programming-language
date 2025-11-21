"""Runtime component registry for graph execution.

This module provides centralized management of runtime components needed
for graph execution: agents, prompts, RAG pipelines, tools, and LLMs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable

from namel3ss.agents.runtime import AgentRuntime
from namel3ss.ast import Prompt, AgentDefinition
from namel3ss.ast.rag import RagPipelineDefinition
from namel3ss.llm.base import BaseLLM
from namel3ss.rag.pipeline import RagPipelineRuntime
from n3_server.converter.enhanced_converter import ConversionContext

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Errors during registry initialization or lookup."""
    pass


@dataclass
class RuntimeRegistry:
    """
    Central registry for all runtime components needed for graph execution.
    
    This registry holds instantiated runtime components (agents, prompts, RAG
    pipelines, tools, LLMs) that are referenced by graph nodes. It is built
    from the ConversionContext after graph-to-AST conversion.
    
    Attributes:
        agents: AgentRuntime instances keyed by agent name
        prompts: Prompt AST definitions keyed by prompt name
        rag_pipelines: RagPipelineRuntime instances keyed by dataset name
        tools: Tool functions keyed by tool name
        llms: LLM provider instances keyed by model name/alias
    """
    
    agents: Dict[str, AgentRuntime] = field(default_factory=dict)
    prompts: Dict[str, Prompt] = field(default_factory=dict)
    rag_pipelines: Dict[str, RagPipelineRuntime] = field(default_factory=dict)
    tools: Dict[str, Callable] = field(default_factory=dict)
    llms: Dict[str, BaseLLM] = field(default_factory=dict)
    
    @classmethod
    async def from_conversion_context(
        cls,
        context: ConversionContext,
        llm_registry: Dict[str, BaseLLM],
        tool_registry: Optional[Dict[str, Callable]] = None,
    ) -> RuntimeRegistry:
        """
        Build runtime registry from conversion context and LLM registry.
        
        This method:
        1. Instantiates AgentRuntime for each agent in context
        2. Stores Prompt AST definitions directly
        3. Instantiates RagPipelineRuntime for each RAG dataset
        4. Registers tools from tool_registry
        5. Stores LLM instances from llm_registry
        
        Args:
            context: ConversionContext from graph-to-AST conversion
            llm_registry: LLM instances keyed by model name/alias
            tool_registry: Optional tool functions keyed by tool name
        
        Returns:
            RuntimeRegistry with all components instantiated
        
        Raises:
            RegistryError: If component instantiation fails
        """
        registry = cls(llms=llm_registry, tools=tool_registry or {})
        
        # Register prompts (store AST definitions directly)
        for prompt_name, prompt_ast in context.prompt_registry.items():
            registry.prompts[prompt_name] = prompt_ast
            logger.debug(f"Registered prompt: {prompt_name}")
        
        # Instantiate and register agents
        for agent_name, agent_ast in context.agent_registry.items():
            try:
                agent_runtime = await registry._create_agent_runtime(agent_ast)
                registry.agents[agent_name] = agent_runtime
                logger.info(f"Registered agent: {agent_name} (model: {agent_ast.llm_name})")
            except Exception as e:
                raise RegistryError(f"Failed to instantiate agent '{agent_name}': {e}") from e
        
        # Instantiate and register RAG pipelines
        for rag_name, rag_ast in context.rag_registry.items():
            try:
                rag_pipeline = await registry._create_rag_pipeline(rag_ast)
                registry.rag_pipelines[rag_name] = rag_pipeline
                logger.info(
                    f"Registered RAG pipeline: {rag_name} "
                    f"(encoder: {rag_ast.embeddings.query_encoder}, "
                    f"backend: {rag_ast.vector_index.backend})"
                )
            except Exception as e:
                raise RegistryError(f"Failed to instantiate RAG pipeline '{rag_name}': {e}") from e
        
        # Register tools (from tool_registry)
        for tool_name, tool_fn in (tool_registry or {}).items():
            registry.tools[tool_name] = tool_fn
            logger.debug(f"Registered tool: {tool_name}")
        
        return registry
    
    async def _create_agent_runtime(self, agent_ast: AgentDefinition) -> AgentRuntime:
        """
        Create AgentRuntime instance from AgentDefinition AST.
        
        Args:
            agent_ast: AgentDefinition AST definition
        
        Returns:
            Instantiated AgentRuntime
        
        Raises:
            RegistryError: If LLM not found or instantiation fails
        """
        # Get LLM instance
        llm = self.llms.get(agent_ast.llm_name)
        if not llm:
            raise RegistryError(
                f"LLM model '{agent_ast.llm_name}' not found in LLM registry. "
                f"Available models: {list(self.llms.keys())}"
            )
        
        # Resolve tools
        agent_tools = []
        for tool_name in agent_ast.tool_names:
            tool_fn = self.tools.get(tool_name)
            if not tool_fn:
                logger.warning(
                    f"Tool '{tool_name}' not found in tool registry for agent '{agent_ast.name}'. "
                    f"Available tools: {list(self.tools.keys())}"
                )
                continue
            agent_tools.append(tool_fn)
        
        # Create runtime
        return AgentRuntime(
            name=agent_ast.name,
            llm=llm,
            tools=agent_tools,
            system_prompt=agent_ast.system_prompt,
            max_turns=agent_ast.max_turns or 10,
            temperature=agent_ast.temperature or 0.7,
        )
    
    async def _create_rag_pipeline(self, rag_ast: RagPipelineDefinition) -> RagPipelineRuntime:
        """
        Create RagPipelineRuntime instance from RagPipelineDefinition AST.
        
        Args:
            rag_ast: RagPipelineDefinition AST definition
        
        Returns:
            Instantiated RagPipelineRuntime
        
        Raises:
            RegistryError: If backend initialization fails
        """
        from namel3ss.rag.backends import get_vector_backend
        
        # Get vector backend instance
        backend = get_vector_backend(
            backend_type=rag_ast.vector_index.backend,
            config={
                "index_name": rag_ast.name,
                "dimension": rag_ast.vector_index.dimension,
                **rag_ast.vector_index.options,
            },
        )
        
        # Create pipeline runtime
        reranker = rag_ast.reranking.model if rag_ast.reranking else None
        reranker_config = rag_ast.reranking.options if rag_ast.reranking else {}
        
        return RagPipelineRuntime(
            name=rag_ast.name,
            query_encoder=rag_ast.embeddings.query_encoder,
            index_backend=backend,
            top_k=rag_ast.retrieval.top_k,
            reranker=reranker,
            distance_metric=rag_ast.vector_index.distance_metric,
            config={"reranker_config": reranker_config},
        )
    
    def get_agent(self, name: str) -> AgentRuntime:
        """Get agent runtime by name, raising if not found."""
        agent = self.agents.get(name)
        if not agent:
            raise RegistryError(
                f"Agent '{name}' not found. Available agents: {list(self.agents.keys())}"
            )
        return agent
    
    def get_prompt(self, name: str) -> Prompt:
        """Get prompt AST by name, raising if not found."""
        prompt = self.prompts.get(name)
        if not prompt:
            raise RegistryError(
                f"Prompt '{name}' not found. Available prompts: {list(self.prompts.keys())}"
            )
        return prompt
    
    def get_rag_pipeline(self, name: str) -> RagPipelineRuntime:
        """Get RAG pipeline by name, raising if not found."""
        rag = self.rag_pipelines.get(name)
        if not rag:
            raise RegistryError(
                f"RAG pipeline '{name}' not found. Available pipelines: {list(self.rag_pipelines.keys())}"
            )
        return rag
    
    def get_tool(self, name: str) -> Callable:
        """Get tool function by name, raising if not found."""
        tool = self.tools.get(name)
        if not tool:
            raise RegistryError(
                f"Tool '{name}' not found. Available tools: {list(self.tools.keys())}"
            )
        return tool
    
    def get_llm(self, model: str) -> BaseLLM:
        """Get LLM instance by model name, raising if not found."""
        llm = self.llms.get(model)
        if not llm:
            raise RegistryError(
                f"LLM model '{model}' not found. Available models: {list(self.llms.keys())}"
            )
        return llm
