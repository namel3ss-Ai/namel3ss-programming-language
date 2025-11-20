"""Integration adapters for N3Provider with existing Namel3ss runtime."""

from typing import Any, AsyncIterable, Dict, List, Optional
import asyncio

from namel3ss.llm.base import BaseLLM, ChatMessage, LLMResponse
from namel3ss.providers.base import N3Provider, ProviderMessage, ProviderResponse
from namel3ss.templates import get_default_engine, TemplateError


class ProviderLLMBridge(BaseLLM):
    """
    Bridge adapter that wraps N3Provider as BaseLLM.
    
    This allows N3Provider instances to be used with existing agent
    and chain execution code that expects BaseLLM interface.
    
    Example:
        ```python
        from namel3ss.providers import create_provider_from_spec
        from namel3ss.providers.integration import ProviderLLMBridge
        
        # Create provider
        provider = create_provider_from_spec("openai", "gpt-4")
        
        # Wrap as BaseLLM
        llm = ProviderLLMBridge(provider)
        
        # Use with agents
        agent_runtime = AgentRuntime(agent_def, llm, tools)
        ```
    """
    
    def __init__(
        self,
        provider: N3Provider,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ):
        """
        Initialize bridge.
        
        Args:
            provider: N3Provider instance to wrap
            default_temperature: Default temperature for generation
            default_max_tokens: Default max tokens for generation
        """
        self.provider = provider
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from prompt.
        
        Converts simple string prompt to provider message format.
        """
        messages = [ProviderMessage(role="user", content=prompt)]
        
        # Run async provider in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self.provider.generate(messages, **self._merge_kwargs(kwargs))
            )
        finally:
            loop.close()
        
        return self._convert_response(response)
    
    def generate_chat(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        """
        Generate chat completion from messages.
        
        Converts ChatMessage to ProviderMessage format.
        """
        provider_messages = [
            ProviderMessage(role=msg.role, content=msg.content)
            for msg in messages
        ]
        
        # Run async provider in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self.provider.generate(provider_messages, **self._merge_kwargs(kwargs))
            )
        finally:
            loop.close()
        
        return self._convert_response(response)
    
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Async generate text from prompt.
        
        Preferred method for async execution contexts.
        """
        messages = [ProviderMessage(role="user", content=prompt)]
        response = await self.provider.generate(messages, **self._merge_kwargs(kwargs))
        return self._convert_response(response)
    
    async def agenerate_chat(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        """
        Async generate chat completion from messages.
        
        Preferred method for async execution contexts.
        """
        provider_messages = [
            ProviderMessage(role=msg.role, content=msg.content)
            for msg in messages
        ]
        response = await self.provider.generate(provider_messages, **self._merge_kwargs(kwargs))
        return self._convert_response(response)
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return self.provider.supports_streaming()
    
    async def stream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncIterable[str]:
        """
        Stream chat completion from messages.
        
        Yields text chunks as they are generated.
        """
        if not self.supports_streaming():
            raise NotImplementedError(f"Provider {self.provider.name} does not support streaming")
        
        provider_messages = [
            ProviderMessage(role=msg.role, content=msg.content)
            for msg in messages
        ]
        
        async for chunk in self.provider.stream(provider_messages, **self._merge_kwargs(kwargs)):
            yield chunk
    
    def _merge_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge kwargs with defaults."""
        merged = {}
        
        if self.default_temperature is not None and "temperature" not in kwargs:
            merged["temperature"] = self.default_temperature
        
        if self.default_max_tokens is not None and "max_tokens" not in kwargs:
            merged["max_tokens"] = self.default_max_tokens
        
        merged.update(kwargs)
        return merged
    
    def _convert_response(self, response: ProviderResponse) -> LLMResponse:
        """Convert ProviderResponse to LLMResponse."""
        return LLMResponse(
            text=response.output_text,
            model=response.model,
            finish_reason=response.finish_reason or "stop",
            usage=response.usage,
            raw=response.raw,
        )


async def run_chain_with_provider(
    chain_steps: List[Any],
    provider: N3Provider,
    initial_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a chain using N3Provider for LLM steps.
    
    This is a helper function for integrating providers with chain execution.
    
    Args:
        chain_steps: List of ChainStep objects from AST
        provider: N3Provider instance to use for generation
        initial_input: Initial input variables for chain
        context: Additional context variables
    
    Returns:
        Final output from chain execution
    
    Example:
        ```python
        from namel3ss.providers import create_provider_from_spec
        from namel3ss.providers.integration import run_chain_with_provider
        
        provider = create_provider_from_spec("anthropic", "claude-3-sonnet")
        
        result = await run_chain_with_provider(
            chain_steps=chain.steps,
            provider=provider,
            initial_input={"question": "What is AI?"},
        )
        ```
    """
    context = context or {}
    state = {**initial_input, **context}
    
    for step in chain_steps:
        # Handle different step types
        if step.kind == "llm":
            # Build prompt from step target (template or direct text)
            prompt = _build_prompt_from_step(step, state)
            
            # Generate with provider
            messages = [ProviderMessage(role="user", content=prompt)]
            response = await provider.generate(messages, **step.options)
            
            # Store result in state
            output_var = step.options.get("output", "response")
            state[output_var] = response.output_text
            state[f"{output_var}_raw"] = response.raw
            
        elif step.kind == "tool":
            # Execute tool from target
            tool_name = step.target
            tool_fn = context.get(f"tool_{tool_name}")
            if tool_fn:
                tool_input = _build_tool_input(step, state)
                result = tool_fn(**tool_input)
                output_var = step.options.get("output", "tool_result")
                state[output_var] = result
        
        else:
            # Unknown step kind - skip or log
            pass
    
    return state


async def run_agent_with_provider(
    agent_def: Any,
    provider: N3Provider,
    user_input: str,
    tools: Optional[Dict[str, Any]] = None,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute an agent using N3Provider for reasoning.
    
    This is a helper function for integrating providers with agent execution.
    
    Args:
        agent_def: AgentDefinition from AST
        provider: N3Provider instance to use for reasoning
        user_input: User's input message
        tools: Dict mapping tool names to callable functions
        max_turns: Maximum reasoning turns
    
    Returns:
        Agent execution result
    
    Example:
        ```python
        from namel3ss.providers import create_provider_from_spec
        from namel3ss.providers.integration import run_agent_with_provider
        
        provider = create_provider_from_spec("openai", "gpt-4")
        
        result = await run_agent_with_provider(
            agent_def=agent,
            provider=provider,
            user_input="Analyze sales data",
            tools={"query_db": query_db_fn},
        )
        ```
    """
    tools = tools or {}
    max_turns = max_turns or getattr(agent_def, "max_turns", 10)
    
    messages = [
        ProviderMessage(role="system", content=agent_def.system_prompt or "You are a helpful assistant."),
        ProviderMessage(role="user", content=user_input),
    ]
    
    turns = []
    
    for turn_num in range(max_turns):
        # Generate agent response
        response = await provider.generate(
            messages,
            temperature=getattr(agent_def, "temperature", 0.7),
        )
        
        # Parse tool calls from response
        tool_calls = _parse_tool_calls(response.output_text)
        
        if not tool_calls:
            # Agent is done
            return {
                "status": "success",
                "response": response.output_text,
                "turns": turns,
            }
        
        # Execute tools
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_fn = tools.get(tool_name)
            
            if tool_fn:
                try:
                    result = tool_fn(**tool_call.get("arguments", {}))
                    tool_results.append({"name": tool_name, "result": result})
                except Exception as e:
                    tool_results.append({"name": tool_name, "error": str(e)})
        
        # Add tool results to messages
        for result in tool_results:
            messages.append(
                ProviderMessage(
                    role="assistant",
                    content=f"Tool {result['name']}: {result.get('result', result.get('error'))}",
                )
            )
        
        turns.append({
            "response": response.output_text,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        })
    
    return {
        "status": "max_turns",
        "response": response.output_text if turns else "",
        "turns": turns,
    }


def _build_prompt_from_step(step: Any, state: Dict[str, Any]) -> str:
    """
    Build prompt from chain step and current state using template engine.
    
    Uses the unified template engine for secure, production-grade template rendering.
    Templates support Jinja2 syntax with variables, conditionals, loops, and filters.
    
    Args:
        step: Chain step with target template string
        state: Current state variables for template context
        
    Returns:
        Rendered prompt string
        
    Raises:
        TemplateError: If template compilation or rendering fails
    """
    template_source = step.target
    engine = get_default_engine()
    
    try:
        # Compile and render template with state variables
        compiled = engine.compile(template_source, name=f"chain_step_{step.name or 'unnamed'}")
        return compiled.render(state)
    except TemplateError:
        # Re-raise template errors as-is
        raise
    except Exception as e:
        # Wrap other errors
        raise TemplateError(
            f"Failed to build prompt from chain step: {e}",
            template_name=f"chain_step_{step.name or 'unnamed'}",
            original_error=e,
        )


def _build_tool_input(step: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Build tool input from step options and state."""
    inputs = step.options.get("inputs", {})
    resolved = {}
    
    for key, value in inputs.items():
        if isinstance(value, str) and value.startswith("$"):
            # Variable reference
            var_name = value[1:]
            resolved[key] = state.get(var_name)
        else:
            resolved[key] = value
    
    return resolved


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from LLM response text.
    
    Looks for patterns like:
    - <tool>name</tool><args>{"key": "value"}</args>
    - Function: name(arg1="value", arg2="value")
    
    Returns list of tool call dicts with 'name' and 'arguments' keys.
    """
    import re
    import json
    
    tool_calls = []
    
    # Pattern 1: XML-style tags
    xml_pattern = r'<tool>(.*?)</tool><args>(.*?)</args>'
    for match in re.finditer(xml_pattern, text, re.DOTALL):
        name = match.group(1).strip()
        try:
            args = json.loads(match.group(2).strip())
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({"name": name, "arguments": args})
    
    # Pattern 2: Function call syntax (simplified)
    func_pattern = r'Function:\s*(\w+)\((.*?)\)'
    for match in re.finditer(func_pattern, text):
        name = match.group(1)
        # Simple arg parsing - would need more robust parser for production
        args = {}
        tool_calls.append({"name": name, "arguments": args})
    
    return tool_calls


__all__ = [
    "ProviderLLMBridge",
    "run_chain_with_provider",
    "run_agent_with_provider",
]
