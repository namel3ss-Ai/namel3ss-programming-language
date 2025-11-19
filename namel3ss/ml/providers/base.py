"""Base LLM provider interface and response types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Base exception for LLM provider errors."""
    
    def __init__(self, message: str, *, provider: Optional[str] = None, 
                 status_code: Optional[int] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.original_error = original_error


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All concrete providers (OpenAI, Anthropic, etc.) must implement this interface.
    """
    
    def __init__(self, *, model: str, temperature: float = 0.7, max_tokens: int = 1024,
                 top_p: Optional[float] = None, frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None, **config):
        """
        Initialize the LLM provider.
        
        Args:
            model: The model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            **config: Additional provider-specific configuration
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The user prompt
            system: Optional system message
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with the generated content
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def agenerate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async version of generate()."""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, *, system: Optional[str] = None, 
                       **kwargs) -> AsyncIterator[str]:
        """
        Stream a completion for the given prompt.
        
        Args:
            prompt: The user prompt
            system: Optional system message
            **kwargs: Additional generation parameters
            
        Yields:
            Content chunks as they are generated
            
        Raises:
            LLMError: If streaming fails
        """
        pass
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Generate a completion for a conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with the generated content
        """
        # Default implementation converts to simple prompt
        # Subclasses can override for native chat support
        if not messages:
            raise ValueError("messages cannot be empty")
        
        system = None
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        return self.generate(prompt, system=system, **kwargs)
