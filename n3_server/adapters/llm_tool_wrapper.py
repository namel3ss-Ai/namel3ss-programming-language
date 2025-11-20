"""
LLM tool wrapper for creating LLM-powered tools.

Supports all N3 LLM providers: OpenAI, Anthropic, Vertex AI, Azure OpenAI, Ollama.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

from namel3ss.llm import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    create_llm,
    get_registry as get_llm_registry,
)
from namel3ss.llm.openai_llm import OpenAILLM
from namel3ss.llm.anthropic_llm import AnthropicLLM
from namel3ss.llm.vertex_llm import VertexLLM
from namel3ss.llm.azure_openai_llm import AzureOpenAILLM
from namel3ss.llm.ollama_llm import OllamaLLM


@dataclass
class LLMToolConfig:
    """Configuration for LLM-powered tools."""
    llm_name: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    response_format: str = "text"  # text, json
    output_schema: Optional[Dict[str, Any]] = None


class LLMToolWrapper:
    """
    Wrapper for creating LLM-powered tools.
    
    Enables using LLMs as tools that can be called by agents or chains.
    Supports all N3 LLM providers:
    - OpenAI (GPT-3.5, GPT-4, GPT-4 Turbo)
    - Anthropic (Claude 2, Claude 3)
    - Google Vertex AI (PaLM 2, Gemini)
    - Azure OpenAI
    - Ollama (local models)
    
    Example:
        >>> wrapper = LLMToolWrapper()
        >>> summarizer = wrapper.create_tool(
        ...     name="summarize",
        ...     description="Summarize text",
        ...     llm_name="gpt4",
        ...     system_prompt="You are a summarization expert."
        ... )
        >>> result = await summarizer(text="Long article...")
    """
    
    def __init__(self):
        self.llm_registry = get_llm_registry()
        self._tools: Dict[str, Callable] = {}
    
    def create_tool(
        self,
        name: str,
        description: str,
        llm_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: str = "text",
        output_schema: Optional[Dict[str, Any]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Create an LLM-powered tool.
        
        Args:
            name: Tool name
            description: Tool description
            llm_name: Name of registered LLM to use
            system_prompt: System prompt for LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: "text" or "json"
            output_schema: JSON schema for output (if response_format="json")
            input_schema: JSON schema for inputs
        
        Returns:
            Async callable tool function
        """
        # Get LLM instance
        llm = self.llm_registry.get(llm_name)
        if llm is None:
            raise ValueError(
                f"LLM '{llm_name}' not registered. "
                f"Available: {', '.join(self.llm_registry.list())}"
            )
        
        async def tool_func(**kwargs) -> Any:
            """LLM-powered tool function."""
            # Build prompt from inputs
            user_prompt = self._build_user_prompt(kwargs)
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            messages.append(ChatMessage(role="user", content=user_prompt))
            
            # Call LLM
            response = await self._call_llm(
                llm=llm,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                output_schema=output_schema,
            )
            
            return response
        
        # Set function metadata
        tool_func.__name__ = name
        tool_func.__doc__ = description
        
        # Default input schema
        if input_schema is None:
            input_schema = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input text",
                    }
                },
                "required": ["input"],
            }
        
        # Attach metadata
        tool_func._tool_metadata = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema or {"type": "string"},
            "tags": ["llm", llm.provider],
            "source": "llm",
            "llm_name": llm_name,
        }
        
        self._tools[name] = tool_func
        return tool_func
    
    async def _call_llm(
        self,
        llm: BaseLLM,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        response_format: str,
        output_schema: Optional[Dict[str, Any]],
    ) -> Any:
        """Call LLM with proper handling for each provider."""
        # Handle structured output for supported providers
        if response_format == "json" and output_schema:
            # OpenAI supports response_format
            if isinstance(llm, OpenAILLM):
                response = await llm.generate_chat_async(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response", "text": response.text}
            
            # Anthropic requires JSON in prompt
            elif isinstance(llm, AnthropicLLM):
                # Add JSON instruction to last message
                messages[-1].content += "\n\nRespond with valid JSON only."
                response = await llm.generate_chat_async(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response", "text": response.text}
            
            # Vertex AI (Gemini supports JSON mode)
            elif isinstance(llm, VertexLLM):
                messages[-1].content += "\n\nRespond with valid JSON only."
                response = await llm.generate_chat_async(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response", "text": response.text}
            
            # Azure OpenAI (same as OpenAI)
            elif isinstance(llm, AzureOpenAILLM):
                response = await llm.generate_chat_async(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response", "text": response.text}
            
            # Ollama (add JSON instruction)
            elif isinstance(llm, OllamaLLM):
                messages[-1].content += "\n\nRespond with valid JSON only."
                response = await llm.generate_chat_async(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response", "text": response.text}
        
        # Default text response
        response = await llm.generate_chat_async(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.text
    
    def _build_user_prompt(self, kwargs: Dict[str, Any]) -> str:
        """Build user prompt from input kwargs."""
        # If single "input" key, use it directly
        if len(kwargs) == 1 and "input" in kwargs:
            return str(kwargs["input"])
        
        # If "text" key, use it
        if "text" in kwargs:
            return str(kwargs["text"])
        
        # Otherwise, format all kwargs
        parts = []
        for key, value in kwargs.items():
            parts.append(f"{key}: {value}")
        return "\n".join(parts)
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get all created tools."""
        return self._tools.copy()


def create_llm_tool(
    name: str,
    description: str,
    llm_name: str,
    system_prompt: Optional[str] = None,
    **config,
) -> Callable:
    """
    Convenience function to create an LLM-powered tool.
    
    Args:
        name: Tool name
        description: Tool description
        llm_name: Name of registered LLM
        system_prompt: System prompt
        **config: Additional config (temperature, max_tokens, etc.)
    
    Returns:
        Async callable tool function
    """
    wrapper = LLMToolWrapper()
    return wrapper.create_tool(
        name=name,
        description=description,
        llm_name=llm_name,
        system_prompt=system_prompt,
        **config,
    )


# Example LLM tool creators for each provider
def create_openai_tool(
    name: str,
    description: str,
    model: str = "gpt-4",
    system_prompt: Optional[str] = None,
    **config,
) -> Callable:
    """Create tool using OpenAI LLM."""
    llm = OpenAILLM(name=f"{name}_llm", model=model, config=config)
    get_llm_registry().register(llm)
    return create_llm_tool(name, description, f"{name}_llm", system_prompt, **config)


def create_anthropic_tool(
    name: str,
    description: str,
    model: str = "claude-3-sonnet-20240229",
    system_prompt: Optional[str] = None,
    **config,
) -> Callable:
    """Create tool using Anthropic LLM."""
    llm = AnthropicLLM(name=f"{name}_llm", model=model, config=config)
    get_llm_registry().register(llm)
    return create_llm_tool(name, description, f"{name}_llm", system_prompt, **config)


def create_vertex_tool(
    name: str,
    description: str,
    model: str = "gemini-pro",
    system_prompt: Optional[str] = None,
    **config,
) -> Callable:
    """Create tool using Vertex AI LLM."""
    llm = VertexLLM(name=f"{name}_llm", model=model, config=config)
    get_llm_registry().register(llm)
    return create_llm_tool(name, description, f"{name}_llm", system_prompt, **config)


def create_azure_tool(
    name: str,
    description: str,
    deployment_name: str,
    system_prompt: Optional[str] = None,
    **config,
) -> Callable:
    """Create tool using Azure OpenAI LLM."""
    llm = AzureOpenAILLM(name=f"{name}_llm", model=deployment_name, config=config)
    get_llm_registry().register(llm)
    return create_llm_tool(name, description, f"{name}_llm", system_prompt, **config)


def create_ollama_tool(
    name: str,
    description: str,
    model: str = "llama2",
    system_prompt: Optional[str] = None,
    **config,
) -> Callable:
    """Create tool using Ollama LLM."""
    llm = OllamaLLM(name=f"{name}_llm", model=model, config=config)
    get_llm_registry().register(llm)
    return create_llm_tool(name, description, f"{name}_llm", system_prompt, **config)
