"""Base N3Provider interface and response types.

This module defines the unified provider abstraction for all LLM backends.
N3Provider wraps the existing BaseLLM interface with additional capabilities
for streaming, batching, and unified configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Dict, List, Optional, Union

from namel3ss.llm.base import BaseLLM, ChatMessage, LLMResponse, LLMError


# Type alias for provider messages (compatible with existing ChatMessage)
ProviderMessage = ChatMessage


@dataclass
class ProviderResponse:
    """
    Normalized response from any provider.
    
    This wraps LLMResponse with additional provider-level metadata and
    provides a consistent interface across all backends.
    """
    
    model: str
    """The model that generated the response."""
    
    output_text: str
    """Primary text output from the model."""
    
    raw: Any
    """Raw provider response object."""
    
    usage: Optional[Dict[str, Any]] = None
    """Token usage and cost information."""
    
    finish_reason: Optional[str] = None
    """Why the generation stopped (e.g., 'stop', 'length', 'content_filter')."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional provider-specific metadata."""
    
    @classmethod
    def from_llm_response(cls, response: LLMResponse) -> ProviderResponse:
        """Create a ProviderResponse from an LLMResponse."""
        return cls(
            model=response.model,
            output_text=response.text,
            raw=response.raw,
            usage=response.usage,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
        )
    
    @property
    def prompt_tokens(self) -> int:
        """Number of tokens in the prompt."""
        if self.usage:
            return self.usage.get("prompt_tokens", 0)
        return 0
    
    @property
    def completion_tokens(self) -> int:
        """Number of tokens in the completion."""
        if self.usage:
            return self.usage.get("completion_tokens", 0)
        return 0
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        if self.usage:
            return self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens)
        return 0


class ProviderError(LLMError):
    """Base exception for provider-related errors."""
    pass


class N3Provider(ABC):
    """
    Unified interface for LLM-like providers.
    
    This is the primary abstraction for all LLM backends in Namel3ss.
    It provides async-first generation, streaming, and batching capabilities.
    
    All concrete provider implementations must subclass this and implement
    the required methods.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the N3Provider.
        
        Args:
            name: Logical provider instance name (e.g., "chat_gpt_4o")
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            config: Provider-specific configuration including:
                - temperature: Sampling temperature (default 0.7)
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - timeout: Request timeout in seconds
                - api_key: API key (or use environment variable)
                - base_url: Override default API base URL
                - Additional provider-specific parameters
        """
        self.name = name
        self.model = model
        self.config = config or {}
    
    @abstractmethod
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Single request/response chat-style generation.
        
        This is the primary generation method. All providers must implement this.
        
        Args:
            messages: List of chat messages (system, user, assistant)
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
                     These override instance-level config for this call only.
        
        Returns:
            ProviderResponse with generated text and metadata
        
        Raises:
            ProviderError: If generation fails
        """
        pass
    
    async def generate_batch(
        self,
        batch: List[List[ProviderMessage]],
        **kwargs: Any,
    ) -> List[ProviderResponse]:
        """
        Batch generation for multiple message sequences.
        
        Default implementation: Loop over generate() with optional concurrency.
        Providers can override for more efficient batch processing.
        
        Args:
            batch: List of message sequences
            **kwargs: Additional generation parameters
        
        Returns:
            List of ProviderResponse objects
        
        Raises:
            ProviderError: If any generation fails
        """
        import asyncio
        
        # Default: concurrent execution of individual generate calls
        tasks = [self.generate(messages, **kwargs) for messages in batch]
        return await asyncio.gather(*tasks)
    
    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming generation.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        return False
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """
        Stream generation tokens as they arrive.
        
        Providers that support streaming should override this and set
        supports_streaming() to return True.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
        
        Yields:
            Incremental text chunks as they are generated
        
        Raises:
            NotImplementedError: If streaming is not supported
            ProviderError: If streaming fails
        """
        if False:
            yield  # Make this an async generator
        raise NotImplementedError(
            f"Streaming not supported by provider '{self.name}' ({self.__class__.__name__})"
        )
    
    def get_model(self) -> str:
        """Get the model identifier."""
        return self.model
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"


class BaseLLMAdapter(N3Provider):
    """
    Adapter that wraps existing BaseLLM implementations as N3Providers.
    
    This allows gradual migration from BaseLLM to N3Provider while maintaining
    backwards compatibility.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        name: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize adapter from a BaseLLM instance.
        
        Args:
            llm: Existing BaseLLM instance to wrap
            name: Override name (defaults to llm.name)
            model: Override model (defaults to llm.model)
            config: Override config (defaults to llm.config)
        """
        super().__init__(
            name=name or llm.name,
            model=model or llm.model,
            config=config or llm.config,
        )
        self.llm = llm
    
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate using the wrapped BaseLLM."""
        # Use async wrapper around sync generate_chat
        import asyncio
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_chat(messages, **kwargs)
            )
            return ProviderResponse.from_llm_response(response)
        except LLMError as e:
            raise ProviderError(str(e), provider=e.provider, model=e.model) from e
    
    def supports_streaming(self) -> bool:
        """Check if the wrapped LLM supports streaming."""
        return self.llm.supports_streaming()
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """Stream using the wrapped BaseLLM."""
        if not self.supports_streaming():
            raise NotImplementedError(f"Streaming not supported by {self.llm}")
        
        # Convert sync generator to async
        import asyncio
        loop = asyncio.get_event_loop()
        
        gen = await loop.run_in_executor(
            None,
            lambda: self.llm.stream_chat(messages, **kwargs)
        )
        
        for chunk in gen:
            yield chunk


__all__ = [
    "ProviderMessage",
    "ProviderResponse",
    "ProviderError",
    "N3Provider",
    "BaseLLMAdapter",
]
