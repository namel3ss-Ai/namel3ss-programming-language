"""Base LLM interface and response types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    
    role: str  # "system", "user", "assistant", "function"
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API calls."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        return result


@dataclass
class LLMResponse:
    """Response from an LLM generation call."""
    
    text: str
    """The primary generated text response."""
    
    raw: Any
    """The raw response object from the provider."""
    
    model: str
    """The model that generated the response."""
    
    usage: Optional[Dict[str, Any]] = None
    """Token usage and cost information if available."""
    
    finish_reason: Optional[str] = None
    """Why the generation stopped (e.g., 'stop', 'length', 'content_filter')."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional provider-specific metadata."""
    
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


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.original_error = original_error


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    All concrete provider implementations (OpenAI, Anthropic, Vertex AI, etc.)
    must subclass this and implement the required methods.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the LLM instance.
        
        Args:
            name: Logical name for this LLM instance (e.g., "chat_gpt_4o")
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
            config: Provider-specific configuration including:
                - temperature: Sampling temperature (default 0.7)
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
                - timeout: Request timeout in seconds
                - api_key: API key (or use environment variable)
                - base_url: Override default API base URL
                - Additional provider-specific parameters
        """
        self.name = name
        self.model = model
        self.config = config or {}
        
        # Common parameters with defaults
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.timeout = self.config.get("timeout", 60.0)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Generate a completion for a single text prompt.
        
        Args:
            prompt: The text prompt to complete
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
                     These override instance-level config for this call only.
        
        Returns:
            LLMResponse with generated text and metadata
        
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_chat(
        self,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion for a chat conversation.
        
        Args:
            messages: List of chat messages (system, user, assistant)
            **kwargs: Additional generation parameters
        
        Returns:
            LLMResponse with generated text and metadata
        
        Raises:
            LLMError: If generation fails
        """
        pass
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> List[LLMResponse]:
        """
        Generate completions for multiple prompts.
        
        Default implementation: Loop over generate(). Providers can override
        for more efficient batch processing.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional generation parameters
        
        Returns:
            List of LLMResponse objects
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming generation.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        pass
    
    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterable[str]:
        """
        Stream a completion for a text prompt.
        
        Args:
            prompt: The text prompt to complete
            **kwargs: Additional generation parameters
        
        Yields:
            Text chunks as they are generated
        
        Raises:
            NotImplementedError: If streaming is not supported
            LLMError: If streaming fails
        """
        if not self.supports_streaming():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support streaming"
            )
        raise NotImplementedError("stream() must be implemented by subclass")
    
    def stream_chat(
        self,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> Iterable[str]:
        """
        Stream a completion for a chat conversation.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
        
        Yields:
            Text chunks as they are generated
        
        Raises:
            NotImplementedError: If streaming is not supported
            LLMError: If streaming fails
        """
        if not self.supports_streaming():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support streaming"
            )
        raise NotImplementedError("stream_chat() must be implemented by subclass")
    
    def get_provider_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            Provider name (e.g., "openai", "anthropic")
        """
        return self.__class__.__name__.replace("LLM", "").lower()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
